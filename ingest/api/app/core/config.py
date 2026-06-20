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

    chunk_size_words: int = 700
    chunk_overlap_words: int = 100
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
