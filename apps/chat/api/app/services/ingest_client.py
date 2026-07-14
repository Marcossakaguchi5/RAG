import time
from functools import lru_cache
from typing import Any

import httpx
from fastapi import HTTPException

from app.core.config import get_settings
from app.schemas import CollectionOut, SearchHit


class IngestClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ingest_api_url.rstrip("/")
        self._token: str | None = None
        self._expires_at = 0

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = self.settings.ingest_request_timeout_seconds
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.RequestError as error:
            raise HTTPException(
                status_code=503,
                detail="Nao foi possivel conectar ao modulo de ingestao.",
            ) from error

        body = response.json() if response.content else None
        if response.is_error:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise HTTPException(status_code=response.status_code, detail=detail or "Falha no modulo de ingestao.")
        return body

    def _authenticate(self) -> str:
        if self._token and self._expires_at > int(time.time()) + 10:
            return self._token
        body = self._request(
            "POST",
            "/api/auth/login",
            json={"password": self.settings.ingest_app_password},
        )
        self._token = str(body["access_token"])
        self._expires_at = int(body["expires_at"])
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._authenticate()}"}

    def collections(self) -> list[CollectionOut]:
        body = self._request("GET", "/api/collections", headers=self._headers())
        return [CollectionOut(**item) for item in body]

    def search(self, query: str, collection_name: str, method: str, top_k: int) -> list[SearchHit]:
        body = self._request(
            "POST",
            "/api/search",
            headers=self._headers(),
            json={
                "query": query,
                "collection_name": collection_name,
                "method": method,
                "top_k": top_k,
                "relevant_chunk_ids": [],
            },
        )
        return [SearchHit(**item) for item in body.get("results", [])]


@lru_cache
def get_ingest_client() -> IngestClient:
    return IngestClient()
