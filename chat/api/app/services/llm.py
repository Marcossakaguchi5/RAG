import json
import re
from functools import lru_cache
from typing import Any

import httpx
from fastapi import HTTPException

from app.core.config import get_settings


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _json_candidates(content: str) -> list[str]:
    stripped = _strip_code_fences(content)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates


def _loads_json_object(content: str) -> dict[str, Any]:
    for candidate in _json_candidates(content):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


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
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
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

    def json_chat(self, messages: list[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        content = ""
        try:
            content = self.chat(messages, model=model, max_tokens=900, response_format={"type": "json_object"})
        except HTTPException as error:
            if error.status_code not in {400, 422}:
                raise
            content = self.chat(messages, model=model, max_tokens=900)

        data = _loads_json_object(content)
        if data:
            return data

        repair_prompt = (
            "Converta a avaliacao abaixo para um unico objeto JSON valido. "
            "Use somente estas chaves: faithfulness, faithfulness_reason, answer_relevancy, "
            "answer_relevancy_reason, context_precision, context_precision_reason, context_recall, "
            "context_recall_reason, answer_correctness, answer_correctness_reason. "
            "Notas devem ser numeros de 0 a 1 ou null. Responda somente JSON.\n\n"
            f"Avaliacao original:\n{content}"
        )
        repaired = self.chat(
            [
                {"role": "system", "content": "Voce reescreve avaliacoes em JSON estrito. Responda somente JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            model=model,
            max_tokens=700,
        )
        return _loads_json_object(repaired)


@lru_cache
def get_llm_client() -> LlmClient:
    return LlmClient()
