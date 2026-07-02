from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from qdrant_client import models

from app.models import Chunk, Document
from app.services.embeddings import get_embedding_service
from app.services.sparse_embeddings import get_sparse_embedding_service
from app.services.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, upsert_chunks


def index_document_chunks(document: Document, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    collection_name = document.collection_name
    dense_vectors = get_embedding_service().encode([chunk.content for chunk in chunks])
    sparse_vectors = get_sparse_embedding_service().encode_documents([chunk.content for chunk in chunks])
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload={
                "chunk_id": chunk.id,
                "collection_name": collection_name,
                "document_id": document.id,
                "document_name": document.original_name,
                "page_number": chunk.page_number,
                "ordinal": chunk.ordinal,
                "content": chunk.content,
            },
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
