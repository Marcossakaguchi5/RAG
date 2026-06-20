from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.core.config import get_settings


class SparseEmbeddingService:
    """Gera vetores esparsos BM25 compatíveis com o índice IDF do Qdrant."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = SparseTextEmbedding(
            model_name=settings.sparse_model,
            cache_dir=settings.fastembed_cache_dir,
            language=settings.sparse_language,
        )

    @staticmethod
    def _to_qdrant_vector(embedding: object) -> models.SparseVector:
        indices = [int(value) for value in embedding.indices.tolist()]
        values = [float(value) for value in embedding.values.tolist()]
        return models.SparseVector(indices=indices, values=values)

    def encode_documents(self, texts: list[str]) -> list[models.SparseVector]:
        if not texts:
            return []
        return [self._to_qdrant_vector(item) for item in self._model.embed(texts)]

    def encode_query(self, text: str) -> models.SparseVector | None:
        embedding = next(self._model.query_embed(text), None)
        if embedding is None or len(embedding.indices) == 0:
            return None
        return self._to_qdrant_vector(embedding)


@lru_cache
def get_sparse_embedding_service() -> SparseEmbeddingService:
    return SparseEmbeddingService()
