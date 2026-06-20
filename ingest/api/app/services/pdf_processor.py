import io
import re
from dataclasses import dataclass

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


def extract_pdf_chunks(content: bytes) -> tuple[int, list[ChunkDraft]]:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception as error:
        raise ValueError("Não foi possível abrir o PDF.") from error

    settings = get_settings()
    step = max(settings.chunk_size_words - settings.chunk_overlap_words, 1)
    chunks: list[ChunkDraft] = []
    ordinal = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_text(page.extract_text() or "")
        except Exception as error:
            raise ValueError(f"Não foi possível extrair o texto da página {page_number}.") from error

        words = text.split()
        for start in range(0, len(words), step):
            selected = words[start : start + settings.chunk_size_words]
            if not selected:
                continue
            chunks.append(
                ChunkDraft(
                    page_number=page_number,
                    ordinal=ordinal,
                    content=" ".join(selected),
                    word_count=len(selected),
                )
            )
            ordinal += 1
            if start + settings.chunk_size_words >= len(words):
                break

    if not chunks:
        raise ValueError("O PDF não possui texto extraível. PDFs digitalizados exigem OCR.")

    return len(reader.pages), chunks
