from functools import lru_cache

import httpx
from fastapi import HTTPException

from app.core.config import get_settings


class LlmClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.llm_base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.settings.llm_api_key:
            raise HTTPException(status_code=503, detail="LLM_API_KEY nao configurada para o Chat RAG.")
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "RAG Chat",
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        except httpx.RequestError as error:
            raise HTTPException(status_code=503, detail="Nao foi possivel conectar ao provedor LLM.") from error

        body = response.json() if response.content else {}
        if response.is_error:
            detail = body.get("error", {}).get("message") if isinstance(body, dict) else None
            raise HTTPException(status_code=response.status_code, detail=detail or "Falha ao gerar resposta.")
        return str(body["choices"][0]["message"]["content"]).strip()


@lru_cache
def get_llm_client() -> LlmClient:
    return LlmClient()
