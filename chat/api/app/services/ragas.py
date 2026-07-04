import math
import os
from functools import lru_cache
from typing import Any, Callable

from app.core.config import get_settings
from app.schemas import RagasMetric, RagasReport, RagSource


def _contexts(sources: list[RagSource], max_chars: int) -> list[str]:
    return [source.content.strip()[:max_chars] for source in sources if source.content.strip()]


def _score_value(result: Any) -> tuple[float | None, str | None]:
    value = getattr(result, "value", result)
    reason = getattr(result, "reason", None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if math.isnan(numeric):
            numeric = None
        else:
            numeric = max(0.0, min(1.0, numeric))
    return numeric, str(reason) if reason else None


def _metric(name: str, result: Any, reason: str | None = None) -> RagasMetric:
    value, result_reason = _score_value(result)
    return RagasMetric(name=name, value=value, reason=reason or result_reason)


class OfficialRagasEvaluator:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY nao configurada para calcular RAGAS oficial.")

        try:
            from openai import AsyncOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                ContextUtilization,
                FactualCorrectness,
                Faithfulness,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dependencias oficiais do RAGAS nao instaladas. "
                "Rebuild o chat-api com requirements.txt atualizado."
            ) from error

        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        self.max_context_chars = settings.ragas_max_context_characters
        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:8081",
                "X-Title": "RAG Chat",
            },
        )
        self.llm = llm_factory(
            settings.resolved_ragas_model,
            client=client,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model=settings.ragas_embedding_model,
            device=settings.ragas_embedding_device or None,
            normalize_embeddings=True,
        )
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_utilization = ContextUtilization(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        self.factual_correctness = FactualCorrectness(llm=self.llm)

    def evaluate(
        self,
        query: str,
        answer: str,
        sources: list[RagSource],
        reference_answer: str = "",
    ) -> RagasReport:
        contexts = _contexts(sources, self.max_context_chars)
        if not contexts:
            return RagasReport(evaluated=False, message="Sem contextos recuperados para avaliar com RAGAS oficial.")

        reference = reference_answer.strip()
        calls: list[tuple[str, Callable[[], Any], str | None]] = [
            (
                "Faithfulness",
                lambda: self.faithfulness.score(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                None,
            ),
            (
                "Answer relevancy",
                lambda: self.answer_relevancy.score(
                    user_input=query,
                    response=answer,
                ),
                None,
            ),
        ]

        if reference:
            calls.extend(
                [
                    (
                        "Context precision",
                        lambda: self.context_precision.score(
                            user_input=query,
                            reference=reference,
                            retrieved_contexts=contexts,
                        ),
                        None,
                    ),
                    (
                        "Context recall",
                        lambda: self.context_recall.score(
                            user_input=query,
                            reference=reference,
                            retrieved_contexts=contexts,
                        ),
                        None,
                    ),
                    (
                        "Factual correctness",
                        lambda: self.factual_correctness.score(
                            response=answer,
                            reference=reference,
                        ),
                        None,
                    ),
                ]
            )
        else:
            calls.append(
                (
                    "Context precision",
                    lambda: self.context_utilization.score(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=contexts,
                    ),
                    "Sem resposta de referencia; RAGAS oficial calculou ContextUtilization como proxy.",
                )
            )
            calls.extend(
                [
                    (
                        "Context recall",
                        None,
                        "Requer resposta de referencia no RAGAS oficial.",
                    ),
                    (
                        "Factual correctness",
                        None,
                        "Requer resposta de referencia no RAGAS oficial.",
                    ),
                ]
            )

        metrics = []
        for name, scorer, reason in calls:
            if scorer is None:
                metrics.append(RagasMetric(name=name, value=None, reason=reason))
                continue
            metrics.append(_metric(name, scorer(), reason))

        return RagasReport(evaluated=True, message="Metricas calculadas com a biblioteca oficial ragas.", metrics=metrics)


@lru_cache
def get_official_ragas_evaluator() -> OfficialRagasEvaluator:
    return OfficialRagasEvaluator()


def evaluate_ragas(query: str, answer: str, sources: list[RagSource], reference_answer: str = "") -> RagasReport:
    return get_official_ragas_evaluator().evaluate(query, answer, sources, reference_answer)
