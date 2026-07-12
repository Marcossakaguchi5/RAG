import io
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

CHUNKING_STRATEGIES = {
    "fixed_token": "Baseline 1: fixed_token",
    "recursive_text": "Baseline 2: recursive_text",
    "docling_hierarchical": "Estrutural 1: docling_hierarchical",
    "docling_hybrid": "Estrutural 2: docling_hybrid",
    "docling_hybrid_parent_child": "Avançada 1: docling_hybrid_parent_child",
    "docling_hybrid_contextual": "Avançada 2: docling_hybrid_contextual",
}
_STRATEGY_ALIASES = {
    "dynamic_blocks": "recursive_text",
    "default": "recursive_text",
}


@dataclass(frozen=True)
class ChunkDraft:
    page_number: int
    ordinal: int
    content: str
    word_count: int
    chunking_strategy: str = "recursive_text"


@dataclass(frozen=True)
class ExtractedContent:
    page_count: int
    text: str
    dl_doc: Any | None = None


def normalize_chunking_strategy(strategy: str | None) -> str:
    value = (strategy or get_settings().chunking_strategy).strip().lower()
    value = _STRATEGY_ALIASES.get(value, value)
    if value not in CHUNKING_STRATEGIES:
        accepted = ", ".join(CHUNKING_STRATEGIES)
        raise ValueError(f"Estratégia de chunking inválida: {strategy}. Use uma destas: {accepted}.")
    return value


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


def _document_to_text(document: Any) -> str:
    markdown_getter = getattr(document, "export_to_markdown", None)
    if markdown_getter is not None:
        return str(markdown_getter())
    text_getter = getattr(document, "export_to_text", None)
    return str(text_getter()) if text_getter is not None else str(document)


def _extract_with_docling(content: bytes, use_ocr: bool = False) -> tuple[str, Any]:
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
            text = _document_to_text(document)
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

    return text.strip(), document


def _extract_with_pdftotext(content: bytes) -> str:
    """Extract PDF text in the reading order used by the text baselines."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        pdf_path = Path(temporary_dir) / "document.pdf"
        pdf_path.write_bytes(content)
        try:
            completed = subprocess.run(
                ["pdftotext", "-raw", str(pdf_path), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise ValueError(
                "pdftotext não está instalado na imagem da API de ingestão."
            ) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").strip()
            raise ValueError(
                "pdftotext não conseguiu extrair o conteúdo do PDF"
                + (f": {detail}" if detail else ".")
            ) from error
    text = completed.stdout.strip()
    if not text:
        raise ValueError("O PDF não possui texto extraível pelo pdftotext.")
    return text


def _extract_content(content: bytes) -> ExtractedContent:
    page_count = _pdf_page_count(content)
    settings = get_settings()
    errors: list[str] = []
    text = ""
    dl_doc: Any | None = None

    if settings.docling_enabled:
        try:
            text, dl_doc = _extract_with_docling(content, use_ocr=False)
        except ValueError as error:
            errors.append(str(error))

    if _has_enough_text(text):
        return ExtractedContent(page_count=page_count, text=text, dl_doc=dl_doc)

    if settings.docling_enabled and settings.docling_ocr_enabled:
        try:
            text, dl_doc = _extract_with_docling(content, use_ocr=True)
        except ValueError as error:
            errors.append(str(error))

    if text:
        return ExtractedContent(page_count=page_count, text=text, dl_doc=dl_doc)

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


def _recursive_split_block(block: str, max_words: int, separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")) -> list[str]:
    block = block.strip()
    if not block or _word_count(block) <= max_words:
        return [block] if block else []
    if not separators:
        return _split_large_block(block, max_words)

    separator = separators[0]
    parts = block.split(separator)
    if len(parts) <= 1:
        return _recursive_split_block(block, max_words, separators[1:])

    chunks: list[str] = []
    current = ""
    joiner = separator if separator != " " else " "
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current}{joiner}{part}" if current else part
        if _word_count(candidate) <= max_words:
            current = candidate
            continue
        if current:
            chunks.extend(_recursive_split_block(current, max_words, separators[1:]))
        current = part
    if current:
        chunks.extend(_recursive_split_block(current, max_words, separators[1:]))
    return chunks


def _overlap_tail(text: str, overlap_words: int) -> str:
    if overlap_words <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else " ".join(words)


def _page_for_ordinal(ordinal: int, estimated_total_chunks: int, page_count: int) -> int:
    return max(1, min(page_count, math.floor((ordinal / estimated_total_chunks) * page_count) + 1))


def _build_chunk_drafts(texts: list[str], page_count: int, strategy: str) -> list[ChunkDraft]:
    normalized_texts = [_normalize_text(text) for text in texts if _normalize_text(text)]
    if not normalized_texts:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")
    total_words = sum(_word_count(text) for text in normalized_texts)
    settings = get_settings()
    estimated_total_chunks = max(1, math.ceil(total_words / max(1, settings.chunk_size_words)))
    return [
        ChunkDraft(
            page_number=_page_for_ordinal(index, estimated_total_chunks, page_count),
            ordinal=index,
            content=text,
            word_count=_word_count(text),
            chunking_strategy=strategy,
        )
        for index, text in enumerate(normalized_texts)
    ]


def _dynamic_chunks(text: str, page_count: int, strategy: str = "recursive_text") -> list[ChunkDraft]:
    settings = get_settings()
    max_words = max(settings.chunk_size_words, 1)
    min_words = min(max(settings.chunk_min_words, 1), max_words)
    overlap_words = min(max(settings.chunk_overlap_words, 0), max_words - 1)

    blocks: list[str] = []
    for block in _paragraph_blocks(text):
        blocks.extend(_recursive_split_block(block, max_words))

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
        page_number = _page_for_ordinal(ordinal, estimated_total_chunks, page_count)
        chunks.append(
            ChunkDraft(
                page_number=page_number,
                ordinal=ordinal,
                content=content,
                word_count=_word_count(content),
                chunking_strategy=strategy,
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


def _load_transformers_tokenizer() -> Any | None:
    settings = get_settings()
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            settings.embedding_model,
            revision=settings.embedding_model_revision.strip() or None,
        )
    except Exception:
        logger.warning("Tokenizer HuggingFace indisponível; usando fallback por palavras.", exc_info=True)
        return None


def _fixed_token_chunks(text: str, page_count: int, strategy: str = "fixed_token") -> list[ChunkDraft]:
    settings = get_settings()
    max_tokens = max(settings.chunk_size_tokens, 1)
    overlap_tokens = min(max(settings.chunk_overlap_tokens, 0), max_tokens - 1)
    step = max(1, max_tokens - overlap_tokens)
    tokenizer = _load_transformers_tokenizer()
    pieces: list[str] = []

    if tokenizer is not None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(token_ids), step):
            decoded = tokenizer.decode(token_ids[start : start + max_tokens], skip_special_tokens=True).strip()
            if decoded:
                pieces.append(decoded)
    else:
        words = text.split()
        for start in range(0, len(words), step):
            pieces.append(" ".join(words[start : start + max_tokens]))

    return _build_chunk_drafts(pieces, page_count, strategy)


def _docling_chunker_classes() -> tuple[type[Any], type[Any]]:
    try:
        from docling.chunking import HierarchicalChunker, HybridChunker

        return HierarchicalChunker, HybridChunker
    except ImportError:
        from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        return HierarchicalChunker, HybridChunker


def _hybrid_chunker() -> Any:
    _, hybrid_chunker_class = _docling_chunker_classes()
    settings = get_settings()
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        tokenizer = _load_transformers_tokenizer()
        if tokenizer is not None:
            return hybrid_chunker_class(
                tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=settings.chunk_size_tokens),
                merge_peers=True,
            )
    except Exception:
        logger.warning("HybridChunker sem tokenizer customizado; usando configuração padrão do Docling.", exc_info=True)
    return hybrid_chunker_class(merge_peers=True)


def _chunk_text(chunk: Any) -> str:
    return _normalize_text(str(getattr(chunk, "text", "") or chunk))


def _contextualized_text(chunker: Any, chunk: Any) -> str:
    contextualize = getattr(chunker, "contextualize", None)
    if contextualize is None:
        return _chunk_text(chunk)
    return _normalize_text(str(contextualize(chunk=chunk)))


def _docling_page_number(chunk: Any, fallback: int) -> int:
    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", None) or []
    for item in doc_items:
        provenances = getattr(item, "prov", None) or []
        for provenance in provenances:
            page_no = getattr(provenance, "page_no", None) or getattr(provenance, "page_number", None)
            if page_no:
                return max(1, int(page_no))
    return fallback


def _docling_chunks(dl_doc: Any, page_count: int, strategy: str) -> list[ChunkDraft]:
    hierarchical_chunker_class, _ = _docling_chunker_classes()
    if strategy == "docling_hierarchical":
        chunker = hierarchical_chunker_class()
        source_chunks = list(chunker.chunk(dl_doc=dl_doc))
        texts = [_chunk_text(chunk) for chunk in source_chunks]
    else:
        chunker = _hybrid_chunker()
        source_chunks = list(chunker.chunk(dl_doc=dl_doc))
        contextual = strategy == "docling_hybrid_contextual"
        texts = [
            _contextualized_text(chunker, chunk) if contextual else _chunk_text(chunk)
            for chunk in source_chunks
        ]

    chunk_pairs = [
        (chunk, _normalize_text(text))
        for chunk, text in zip(source_chunks, texts, strict=True)
        if _normalize_text(text)
    ]
    if not chunk_pairs:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")
    total_words = sum(_word_count(text) for _, text in chunk_pairs)
    estimated_total_chunks = max(1, math.ceil(total_words / max(1, get_settings().chunk_size_words)))
    return [
        ChunkDraft(
            page_number=_docling_page_number(chunk, _page_for_ordinal(index, estimated_total_chunks, page_count)),
            ordinal=index,
            content=text,
            word_count=_word_count(text),
            chunking_strategy=strategy,
        )
        for index, (chunk, text) in enumerate(chunk_pairs)
    ]


def _context_prefix(contextualized: str, raw_text: str) -> str:
    contextualized = contextualized.strip()
    raw_text = raw_text.strip()
    if raw_text and contextualized.endswith(raw_text):
        return contextualized[: -len(raw_text)].strip()
    return ""


def _word_windows(text: str, size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    size_words = max(size_words, 1)
    overlap_words = min(max(overlap_words, 0), size_words - 1)
    step = max(1, size_words - overlap_words)
    return [" ".join(words[start : start + size_words]) for start in range(0, len(words), step)]


def _docling_parent_child_chunks(dl_doc: Any, page_count: int) -> list[ChunkDraft]:
    settings = get_settings()
    strategy = "docling_hybrid_parent_child"
    chunker = _hybrid_chunker()
    parent_chunks = list(chunker.chunk(dl_doc=dl_doc))
    drafts: list[ChunkDraft] = []
    for parent in parent_chunks:
        raw_text = _chunk_text(parent)
        contextualized = _contextualized_text(chunker, parent)
        prefix = _context_prefix(contextualized, raw_text)
        child_texts = _word_windows(
            raw_text,
            settings.parent_child_size_words,
            settings.parent_child_overlap_words,
        )
        if not child_texts and raw_text:
            child_texts = [raw_text]
        for child_text in child_texts:
            content = f"{prefix}\n\n{child_text}" if prefix else child_text
            ordinal = len(drafts)
            fallback_page = _page_for_ordinal(ordinal, max(1, len(parent_chunks)), page_count)
            drafts.append(
                ChunkDraft(
                    page_number=_docling_page_number(parent, fallback_page),
                    ordinal=ordinal,
                    content=_normalize_text(content),
                    word_count=_word_count(content),
                    chunking_strategy=strategy,
                )
            )
    if not drafts:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")
    return drafts


def chunk_text(
    text: str,
    page_count: int = 1,
    chunking_strategy: str | None = None,
) -> list[ChunkDraft]:
    strategy = normalize_chunking_strategy(chunking_strategy)
    page_count = max(page_count, 1)
    if strategy == "fixed_token":
        return _fixed_token_chunks(text, page_count, strategy)
    if strategy == "recursive_text":
        return _dynamic_chunks(text, page_count, strategy)
    raise ValueError(f"A estratégia {strategy} exige um DoclingDocument; use extract_pdf_chunks para PDFs.")


def extract_pdf_chunks(content: bytes, chunking_strategy: str | None = None) -> tuple[int, list[ChunkDraft]]:
    strategy = normalize_chunking_strategy(chunking_strategy)
    if strategy in {"fixed_token", "recursive_text"}:
        # Docling's Markdown export can reorder columns in born-digital PDFs.
        # The baseline needs a stable linear reading order for its qrels.
        page_count = _pdf_page_count(content)
        return page_count, chunk_text(_extract_with_pdftotext(content), page_count, strategy)
    extracted = _extract_content(content)
    if extracted.dl_doc is None:
        raise ValueError(f"A estratégia {strategy} exige Docling habilitado para preservar a estrutura do documento.")
    if strategy == "docling_hybrid_parent_child":
        return extracted.page_count, _docling_parent_child_chunks(extracted.dl_doc, extracted.page_count)
    return extracted.page_count, _docling_chunks(extracted.dl_doc, extracted.page_count, strategy)
