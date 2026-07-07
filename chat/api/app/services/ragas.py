import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable

from app.core.config import get_settings
from app.schemas import RagasMetric, RagasReport, RagSource


def _contexts(sources: list[RagSource], max_chars: int, max_sources: int) -> list[str]:
    return [
        source.content.strip()[:max_chars]
        for source in sources[:max_sources]
        if source.content.strip()
    ]


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
        self.max_sources = settings.ragas_max_sources
        self.parallel_workers = settings.ragas_parallel_workers
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
        started_at = time.perf_counter()
        contexts = _contexts(sources, self.max_context_chars, self.max_sources)
        if not contexts:
            return RagasReport(evaluated=False, message="Sem contextos recuperados para avaliar com RAGAS oficial.")

        reference = reference_answer.strip()
        calls: list[tuple[str, Callable[[], Any] | None, str | None]] = [
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

        metrics_by_name: dict[str, RagasMetric] = {}
        executable_calls: list[tuple[str, Callable[[], Any], str | None]] = []
        for name, scorer, reason in calls:
            if scorer is None:
                metrics_by_name[name] = RagasMetric(name=name, value=None, reason=reason)
            else:
                executable_calls.append((name, scorer, reason))

        with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(executable_calls) or 1)) as executor:
            futures = {
                executor.submit(scorer): (name, reason)
                for name, scorer, reason in executable_calls
            }
            for future in as_completed(futures):
                name, reason = futures[future]
                try:
                    metrics_by_name[name] = _metric(name, future.result(), reason)
                except Exception as error:
                    metrics_by_name[name] = RagasMetric(
                        name=name,
                        value=None,
                        reason=f"Falha ao calcular esta metrica: {error}",
                    )

        metrics = [metrics_by_name[name] for name, _, _ in calls]
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        return RagasReport(
            evaluated=True,
            message=(
                "Metricas calculadas com a biblioteca oficial ragas "
                f"em {elapsed_ms} ms usando {len(contexts)} contexto(s)."
            ),
            metrics=metrics,
        )


@lru_cache
def get_official_ragas_evaluator() -> OfficialRagasEvaluator:
    return OfficialRagasEvaluator()


def evaluate_ragas(query: str, answer: str, sources: list[RagSource], reference_answer: str = "") -> RagasReport:
    return get_official_ragas_evaluator().evaluate(query, answer, sources, reference_answer)
