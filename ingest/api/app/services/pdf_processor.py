import io
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkDraft:
    page_number: int
    ordinal: int
    content: str
    word_count: int


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _has_enough_text(text: str) -> bool:
    settings = get_settings()
    return len(_normalize_text(text)) >= settings.docling_ocr_min_text_chars


def _pdf_page_count(content: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
        return len(reader.pages)
    except Exception as error:
        raise ValueError("Não foi possível abrir o PDF.") from error


def _ocr_languages(settings: Settings) -> list[str]:
    return [language.strip() for language in settings.docling_ocr_languages.split(",") if language.strip()]


def _docling_converter(settings: Settings, use_ocr: bool):
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError:
        return DocumentConverter()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    if use_ocr:
        try:
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=_ocr_languages(settings),
                force_full_page_ocr=settings.docling_ocr_force_full_page,
            )
        except TypeError:
            ocr_options = EasyOcrOptions()
            if hasattr(ocr_options, "lang"):
                ocr_options.lang = _ocr_languages(settings)
            if hasattr(ocr_options, "force_full_page_ocr"):
                ocr_options.force_full_page_ocr = settings.docling_ocr_force_full_page
            pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _extract_with_docling(content: bytes, use_ocr: bool = False) -> str:
    settings = get_settings()
    artifacts_path = settings.docling_artifacts_path.strip()
    if artifacts_path:
        os.environ["DOCLING_ARTIFACTS_PATH"] = artifacts_path
    else:
        os.environ.pop("DOCLING_ARTIFACTS_PATH", None)
    try:
        converter = _docling_converter(settings, use_ocr)
    except ImportError as error:
        raise ValueError("Docling não está instalado na imagem da API de ingestão.") from error

    with tempfile.TemporaryDirectory() as temporary_dir:
        pdf_path = Path(temporary_dir) / "document.pdf"
        pdf_path.write_bytes(content)
        try:
            result = converter.convert(str(pdf_path))
            document = result.document
            markdown_getter = getattr(document, "export_to_markdown", None)
            if markdown_getter is not None:
                text = markdown_getter()
            else:
                text_getter = getattr(document, "export_to_text", None)
                text = text_getter() if text_getter is not None else str(document)
        except Exception as error:
            detail = " com OCR" if use_ocr else ""
            logger.exception("Docling falhou ao extrair PDF%s", detail)
            error_detail = str(error).strip()
            if len(error_detail) > 500:
                error_detail = f"{error_detail[:500]}..."
            message = f"Docling não conseguiu extrair o conteúdo do PDF{detail}."
            if error_detail:
                message = f"{message} Causa: {error.__class__.__name__}: {error_detail}"
            else:
                message = f"{message} Causa: {error.__class__.__name__}."
            raise ValueError(message) from error

    return text.strip()


def _extract_text(content: bytes) -> tuple[int, str]:
    page_count = _pdf_page_count(content)
    settings = get_settings()
    errors: list[str] = []
    text = ""

    if settings.docling_enabled:
        try:
            text = _extract_with_docling(content, use_ocr=False)
        except ValueError as error:
            errors.append(str(error))

    if _has_enough_text(text):
        return page_count, text

    if settings.docling_enabled and settings.docling_ocr_enabled:
        try:
            text = _extract_with_docling(content, use_ocr=True)
        except ValueError as error:
            errors.append(str(error))

    if text:
        return page_count, text

    if errors:
        raise ValueError(
            "Não foi possível extrair texto do PDF. "
            "Se ele for digitalizado, verifique se o OCR do Docling está disponível. "
            f"Detalhes: {' | '.join(errors)}"
        )
    raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")


def _paragraph_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [_normalize_text(block) for block in re.split(r"\n\s*\n+", normalized)]
    if len(blocks) <= 1:
        blocks = [_normalize_text(block) for block in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])", normalized)]
    return [block for block in blocks if block]


def _split_large_block(block: str, max_words: int) -> list[str]:
    words = block.split()
    if len(words) <= max_words:
        return [block]
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def _overlap_tail(text: str, overlap_words: int) -> str:
    if overlap_words <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else " ".join(words)


def _dynamic_chunks(text: str, page_count: int) -> list[ChunkDraft]:
    settings = get_settings()
    max_words = max(settings.chunk_size_words, 1)
    min_words = min(max(settings.chunk_min_words, 1), max_words)
    overlap_words = min(max(settings.chunk_overlap_words, 0), max_words - 1)

    blocks: list[str] = []
    for block in _paragraph_blocks(text):
        blocks.extend(_split_large_block(block, max_words))

    total_words = sum(_word_count(block) for block in blocks)
    estimated_total_chunks = max(1, math.ceil(total_words / max(1, max_words - overlap_words)))
    chunks: list[ChunkDraft] = []
    current: list[str] = []
    current_words = 0
    current_is_overlap_only = False

    def flush() -> None:
        nonlocal current, current_words, current_is_overlap_only
        if not current:
            return
        content = _normalize_text("\n\n".join(current))
        if not content:
            current = []
            current_words = 0
            current_is_overlap_only = False
            return
        ordinal = len(chunks)
        page_number = max(1, min(page_count, math.floor((ordinal / estimated_total_chunks) * page_count) + 1))
        chunks.append(
            ChunkDraft(
                page_number=page_number,
                ordinal=ordinal,
                content=content,
                word_count=_word_count(content),
            )
        )
        overlap = _overlap_tail(content, overlap_words)
        current = [overlap] if overlap else []
        current_words = _word_count(overlap)
        current_is_overlap_only = bool(overlap)

    for block in blocks:
        block_words = _word_count(block)
        if current and current_words >= min_words and current_words + block_words > max_words:
            flush()
        current.append(block)
        current_words += block_words
        current_is_overlap_only = False
        if current_words >= max_words:
            flush()

    if current and not current_is_overlap_only:
        flush()

    if not chunks:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")

    return chunks


def chunk_text(text: str, page_count: int = 1) -> list[ChunkDraft]:
    return _dynamic_chunks(text, max(page_count, 1))


def extract_pdf_chunks(content: bytes) -> tuple[int, list[ChunkDraft]]:
    page_count, text = _extract_text(content)
    return page_count, chunk_text(text, page_count)
