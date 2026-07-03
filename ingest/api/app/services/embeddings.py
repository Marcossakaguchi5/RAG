from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self._model.get_sentence_embedding_dimension
        dimension = dimension_getter()
        if dimension is None:
            raise RuntimeError("Não foi possível determinar a dimensão do modelo de embeddings.")
        return dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
