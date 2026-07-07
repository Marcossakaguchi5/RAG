from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Ingest API"
    environment: str = "development"
    cors_origins: str = "http://localhost:8080"
    ingest_app_password: str = Field(default="alterar-esta-senha", min_length=1)
    ingest_auth_token_ttl_seconds: int = Field(default=28800, gt=0)

    mysql_host: str = "mysql"
    mysql_port: int = 3306
    mysql_database: str = "rag_ingest"
    mysql_user: str = "rag"
    mysql_password: str = "rag_secret"

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "portuguese"
    fastembed_cache_dir: str = "/models/fastembed"

    docling_enabled: bool = True
    docling_artifacts_path: str = ""
    docling_ocr_enabled: bool = True
    docling_ocr_force_full_page: bool = False
    docling_ocr_languages: str = "pt,en"
    docling_ocr_min_text_chars: int = Field(default=80, ge=0)
    chunking_strategy: str = "recursive_text"
    chunk_min_words: int = Field(default=180, ge=1)
    chunk_size_words: int = 700
    chunk_overlap_words: int = 100
    chunk_size_tokens: int = Field(default=512, ge=32)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    parent_child_size_words: int = Field(default=320, ge=32)
    parent_child_overlap_words: int = Field(default=60, ge=0)
    max_upload_bytes: int = 50 * 1024 * 1024
    hybrid_dense_weight: float = 0.5

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
