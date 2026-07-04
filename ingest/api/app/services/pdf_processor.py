import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_settings


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


def _pdf_page_count(content: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
        return len(reader.pages)
    except Exception as error:
        raise ValueError("Não foi possível abrir o PDF.") from error


def _extract_with_pypdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception as error:
        raise ValueError("Não foi possível abrir o PDF.") from error

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_text(page.extract_text() or "")
        except Exception as error:
            raise ValueError(f"Não foi possível extrair o texto da página {page_number}.") from error
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_with_docling(content: bytes) -> str:
    settings = get_settings()
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", settings.docling_artifacts_path)
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise ValueError("Docling não está instalado na imagem da API de ingestão.") from error

    with tempfile.TemporaryDirectory() as temporary_dir:
        pdf_path = Path(temporary_dir) / "document.pdf"
        pdf_path.write_bytes(content)
        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            document = result.document
            markdown_getter = getattr(document, "export_to_markdown", None)
            if markdown_getter is not None:
                text = markdown_getter()
            else:
                text_getter = getattr(document, "export_to_text", None)
                text = text_getter() if text_getter is not None else str(document)
        except Exception as error:
            raise ValueError("Docling não conseguiu extrair o conteúdo do PDF.") from error

    return text.strip()


def _extract_text(content: bytes) -> tuple[int, str]:
    page_count = _pdf_page_count(content)
    settings = get_settings()
    if settings.docling_enabled:
        text = _extract_with_docling(content)
    else:
        text = _extract_with_pypdf(content)

    if not text:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")
    return page_count, text


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


def extract_pdf_chunks(content: bytes) -> tuple[int, list[ChunkDraft]]:
    page_count, text = _extract_text(content)
    return page_count, _dynamic_chunks(text, page_count)
