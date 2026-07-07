from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "chat/.env"),
        extra="ignore",
    )

    app_name: str = "RAG Chat API"
    environment: str = "development"
    cors_origins: str = "http://localhost:8081"

    chat_app_password: str = Field(default="alterar-esta-senha", min_length=1)
    chat_auth_token_ttl_seconds: int = Field(default=28800, gt=0)

    ingest_api_url: str = "http://localhost:8010"
    ingest_app_password: str = Field(default="alterar-esta-senha", min_length=1)
    ingest_request_timeout_seconds: float = Field(default=90, gt=0)

    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek/deepseek-v4-flash"
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=1200, ge=128, le=8000)

    max_context_characters: int = Field(default=24000, ge=2000)
    ragas_model: str = ""
    ragas_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ragas_embedding_device: str = ""
    ragas_max_context_characters: int = Field(default=3000, ge=500)
    ragas_max_sources: int = Field(default=3, ge=1, le=20)
    ragas_parallel_workers: int = Field(default=4, ge=1, le=8)

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_ragas_model(self) -> str:
        return self.ragas_model or self.llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
