from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from qdrant_client import models

from app.core.config import get_settings
from app.models import Chunk, Document
from app.services.embeddings import get_embedding_service
from app.services.sparse_embeddings import get_sparse_embedding_service
from app.services.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, upsert_chunks


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _document_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".") or "pdf"


def _chunk_payload(
    document: Document,
    chunk: Chunk,
    *,
    collection_name: str,
    chunk_total: int,
    indexed_at: str,
) -> dict[str, object]:
    settings = get_settings()
    char_count = len(chunk.content)
    page_number = max(chunk.page_number, 1)
    payload = {
        "schema_version": 2,
        "source_type": "pdf",
        "collection_name": collection_name,
        "chunk_id": chunk.id,
        "point_id": chunk.id,
        "document_id": document.id,
        "document_name": document.original_name,
        "file_name": document.original_name,
        "file_extension": _document_extension(document.original_name),
        "content_type": document.content_type,
        "object_name": document.object_name,
        "minio_bucket": settings.minio_bucket,
        "document_size_bytes": document.size_bytes,
        "document_page_count": document.page_count,
        "page_count": document.page_count,
        "page_number": page_number,
        "page_start": page_number,
        "page_end": page_number,
        "ordinal": chunk.ordinal,
        "chunk_ordinal": chunk.ordinal,
        "chunk_index": chunk.ordinal + 1,
        "chunk_total": chunk_total,
        "is_first_chunk": chunk.ordinal == 0,
        "is_last_chunk": chunk.ordinal == chunk_total - 1,
        "word_count": chunk.word_count,
        "chunk_word_count": chunk.word_count,
        "char_count": char_count,
        "chunk_char_count": char_count,
        "chunking_strategy": "dynamic_blocks",
        "embedding_model": settings.embedding_model,
        "sparse_model": settings.sparse_model,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "sparse_vector_name": SPARSE_VECTOR_NAME,
        "indexed_at": indexed_at,
        "document_created_at": _iso_datetime(document.created_at),
        "chunk_created_at": _iso_datetime(chunk.created_at),
        "content": chunk.content,
    }
    return {key: value for key, value in payload.items() if value is not None}


def index_document_chunks(document: Document, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    collection_name = document.collection_name
    dense_vectors = get_embedding_service().encode([chunk.content for chunk in chunks])
    sparse_vectors = get_sparse_embedding_service().encode_documents([chunk.content for chunk in chunks])
    indexed_at = datetime.now(timezone.utc).isoformat()
    chunk_total = len(chunks)
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload=_chunk_payload(
                document,
                chunk,
                collection_name=collection_name,
                chunk_total=chunk_total,
                indexed_at=indexed_at,
            ),
        )
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ]
    upsert_chunks(points, collection_name)


def rebuild_index_from_mysql(session: Session, collection_name: str | None = None, batch_size: int = 20) -> int:
    """Reconstrói o índice derivado em lotes de documentos, sem carregar a base inteira."""
    indexed_chunks = 0
    last_document_id = ""
    while True:
        query = (
            select(Document)
            .where(Document.id > last_document_id)
            .options(selectinload(Document.chunks))
            .order_by(Document.id)
            .limit(batch_size)
        )
        if collection_name:
            query = query.where(Document.collection_name == collection_name)
        documents = list(
            session.scalars(
                query
            )
        )
        if not documents:
            return indexed_chunks
        for document in documents:
            chunks = list(document.chunks)
            index_document_chunks(document, chunks)
            indexed_chunks += len(chunks)
        last_document_id = documents[-1].id
