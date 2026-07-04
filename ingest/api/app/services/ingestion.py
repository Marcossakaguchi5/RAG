from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, Document
from app.schemas import DocumentOut
from app.services.collections import normalize_collection_name
from app.services.indexer import index_document_chunks
from app.services.pdf_processor import extract_pdf_chunks
from app.services.storage import delete_object, put_pdf
from app.services.vector_store import delete_chunks


def document_to_schema(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        original_name=document.original_name,
        collection_name=document.collection_name,
        size_bytes=document.size_bytes,
        page_count=document.page_count,
        chunks_count=len(document.chunks),
        created_at=document.created_at,
    )


def ingest_pdf(upload: UploadFile, session: Session, collection_name: str) -> DocumentOut:
    collection_name = normalize_collection_name(collection_name)
    filename = Path(upload.filename or "documento.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Apenas arquivos PDF são aceitos nesta etapa.")

    data = upload.file.read()
    if len(data) > get_settings().max_upload_bytes:
        raise ValueError("O arquivo excede o limite de 50 MB.")
    if not data.startswith(b"%PDF"):
        raise ValueError("O arquivo enviado não parece ser um PDF válido.")

    page_count, drafts = extract_pdf_chunks(data)
    document_id = str(uuid4())
    object_name = f"documents/{collection_name}/{document_id}/{filename}"
    document = Document(
        id=document_id,
        original_name=filename,
        collection_name=collection_name,
        object_name=object_name,
        content_type="application/pdf",
        size_bytes=len(data),
        page_count=page_count,
    )
    chunks = [
        Chunk(
            id=str(uuid4()),
            document_id=document_id,
            ordinal=draft.ordinal,
            page_number=draft.page_number,
            content=draft.content,
            word_count=draft.word_count,
        )
        for draft in drafts
    ]
    stored_file = False
    indexed_points = False
    try:
        put_pdf(object_name, data)
        stored_file = True
        index_document_chunks(document, chunks)
        indexed_points = True
        session.add(document)
        session.add_all(chunks)
        session.commit()
        session.refresh(document)
        return document_to_schema(document)
    except Exception:
        session.rollback()
        if indexed_points:
            try:
                delete_chunks([chunk.id for chunk in chunks], collection_name)
            except Exception:
                pass
        if stored_file:
            try:
                delete_object(object_name)
            except Exception:
                pass
        raise
