from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.services.embeddings import get_embedding_service

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
PAYLOAD_INDEXES = {
    "collection_name": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "document_name": models.PayloadSchemaType.TEXT,
    "file_name": models.PayloadSchemaType.TEXT,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "page_number": models.PayloadSchemaType.INTEGER,
    "ordinal": models.PayloadSchemaType.INTEGER,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "chunk_total": models.PayloadSchemaType.INTEGER,
    "word_count": models.PayloadSchemaType.INTEGER,
    "indexed_at": models.PayloadSchemaType.DATETIME,
}
_payload_indexes_checked: set[str] = set()


@lru_cache
def get_vector_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _create_collection(collection_name: str) -> None:
    client = get_vector_client()
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=get_embedding_service().dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    ensure_payload_indexes(collection_name, force=True)


def ensure_payload_indexes(collection_name: str, force: bool = False) -> None:
    if not force and collection_name in _payload_indexes_checked:
        return
    client = get_vector_client()
    for field_name, field_schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception:
            # O índice pode já existir ou versões antigas do Qdrant podem rejeitar algum tipo.
            # A busca vetorial continua funcionando; os índices só aceleram filtros futuros.
            pass
    _payload_indexes_checked.add(collection_name)


def _is_hybrid_collection(collection_name: str) -> bool:
    collection = get_vector_client().get_collection(collection_name)
    vectors = collection.config.params.vectors
    sparse_vectors = collection.config.params.sparse_vectors or {}
    return (
        isinstance(vectors, dict)
        and DENSE_VECTOR_NAME in vectors
        and SPARSE_VECTOR_NAME in sparse_vectors
    )


def ensure_collection(collection_name: str | None = None) -> bool:
    """Garante a coleção hybrid; retorna True quando ela foi criada/migrada."""
    settings = get_settings()
    resolved_collection = collection_name or settings.qdrant_collection
    client = get_vector_client()
    if not client.collection_exists(resolved_collection):
        _create_collection(resolved_collection)
        return True

    if _is_hybrid_collection(resolved_collection):
        ensure_payload_indexes(resolved_collection)
        return False

    # O Qdrant é índice derivado: os chunks canônicos estão no MySQL e serão reindexados.
    client.recreate_collection(
        collection_name=resolved_collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=get_embedding_service().dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    ensure_payload_indexes(resolved_collection, force=True)
    return True


def upsert_chunks(points: list[models.PointStruct], collection_name: str | None = None) -> None:
    if not points:
        return
    settings = get_settings()
    get_vector_client().upsert(collection_name=collection_name or settings.qdrant_collection, points=points, wait=True)


def delete_chunks(chunk_ids: list[str], collection_name: str | None = None) -> None:
    if not chunk_ids:
        return
    settings = get_settings()
    get_vector_client().delete(
        collection_name=collection_name or settings.qdrant_collection,
        points_selector=models.PointIdsList(points=chunk_ids),
        wait=True,
    )


def search_dense(query_vector: list[float], limit: int, collection_name: str | None = None) -> list[Any]:
    settings = get_settings()
    response = get_vector_client().query_points(
        collection_name=collection_name or settings.qdrant_collection,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def search_sparse(query_vector: models.SparseVector, limit: int, collection_name: str | None = None) -> list[Any]:
    settings = get_settings()
    response = get_vector_client().query_points(
        collection_name=collection_name or settings.qdrant_collection,
        query=query_vector,
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def search_hybrid(
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    limit: int,
    collection_name: str | None = None,
) -> list[Any]:
    settings = get_settings()
    candidate_limit = min(limit * 4, 200)
    response = get_vector_client().query_points(
        collection_name=collection_name or settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=candidate_limit),
            models.Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=candidate_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def preview_points(limit: int, collection_name: str | None = None) -> tuple[int, list[Any]]:
    settings = get_settings()
    resolved_collection = collection_name or settings.qdrant_collection
    client = get_vector_client()
    total = client.count(collection_name=resolved_collection, exact=True).count
    points, _ = client.scroll(
        collection_name=resolved_collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return total, list(points)
