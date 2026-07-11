import math
import os
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

from app.core.config import get_settings
from app.schemas import RagasMetric, RagasReport, RagSource
from app.services.answering import select_generation_sources


def _ragas_version() -> str:
    try:
        return version("ragas")
    except PackageNotFoundError:
        return "unknown"


def _contexts(sources: list[RagSource]) -> list[str]:
    return [source.content.strip() for source in sources if source.content.strip()]


def _generation_contexts(
    sources: list[RagSource],
    generation_source_ids: list[str] | None,
) -> list[str]:
    if generation_source_ids is None:
        return _contexts(select_generation_sources(sources))

    sources_by_id = {source.chunk_id: source for source in sources}
    missing_ids = [chunk_id for chunk_id in generation_source_ids if chunk_id not in sources_by_id]
    if missing_ids:
        raise ValueError(
            "generation_source_ids contem chunks ausentes em sources: "
            + ", ".join(missing_ids)
        )
    return _contexts([sources_by_id[chunk_id] for chunk_id in generation_source_ids])


def _score_value(result: Any) -> tuple[float | None, str | None]:
    value = getattr(result, "value", result)
    reason = getattr(result, "reason", None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and not math.isfinite(numeric):
        numeric = None
    return numeric, str(reason) if reason else None


def _metric(name: str, result: Any, reason: str | None = None) -> RagasMetric:
    value, result_reason = _score_value(result)
    return RagasMetric(name=name, value=value, reason=reason or result_reason)


class OfficialRagasEvaluator:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.resolved_ragas_api_key:
            raise RuntimeError("RAGAS_LLM_API_KEY ou LLM_API_KEY nao configurada para calcular RAGAS oficial.")

        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
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

        self.ragas_version = _ragas_version()
        self.evaluator_model = settings.resolved_ragas_model
        self.embedding_model = settings.ragas_embedding_model
        client = AsyncOpenAI(
            api_key=settings.resolved_ragas_api_key,
            base_url=settings.resolved_ragas_base_url,
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
        generation_source_ids: list[str] | None = None,
    ) -> RagasReport:
        started_at = time.perf_counter()
        retrieved_contexts = _contexts(sources)
        generation_contexts = _generation_contexts(sources, generation_source_ids)
        report_metadata = {
            "ragas_version": self.ragas_version,
            "evaluator_model": self.evaluator_model,
            "embedding_model": self.embedding_model,
            "retrieved_contexts_count": len(retrieved_contexts),
            "generation_contexts_count": len(generation_contexts),
        }
        if not retrieved_contexts:
            return RagasReport(
                evaluated=False,
                message="Sem contextos recuperados para avaliar com RAGAS oficial.",
                **report_metadata,
            )

        reference = reference_answer.strip()
        calls: list[tuple[str, Callable[[], Any] | None, str | None]] = [
            (
                "Faithfulness",
                (
                    lambda: self.faithfulness.score(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=generation_contexts,
                    )
                )
                if generation_contexts
                else None,
                None if generation_contexts else "Sem contextos enviados ao gerador para avaliar Faithfulness.",
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
                            retrieved_contexts=retrieved_contexts,
                        ),
                        None,
                    ),
                    (
                        "Context recall",
                        lambda: self.context_recall.score(
                            user_input=query,
                            reference=reference,
                            retrieved_contexts=retrieved_contexts,
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
                    "Context utilization",
                    lambda: self.context_utilization.score(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=retrieved_contexts,
                    ),
                    "Sem resposta de referencia; calculado com ContextUtilization.",
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

        for name, scorer, reason in executable_calls:
            try:
                metrics_by_name[name] = _metric(name, scorer(), reason)
            except Exception as error:
                metrics_by_name[name] = RagasMetric(
                    name=name,
                    value=None,
                    reason=f"Falha ao calcular esta metrica: {error}",
                )

        metrics = [metrics_by_name[name] for name, _, _ in calls]
        successful_metrics = sum(
            metrics_by_name[name].value is not None
            for name, _, _ in executable_calls
        )
        attempted_metrics = len(executable_calls)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        return RagasReport(
            evaluated=successful_metrics > 0,
            message=(
                f"{successful_metrics}/{attempted_metrics} metricas executaveis calculadas "
                f"com ragas {self.ragas_version} em {elapsed_ms} ms; "
                f"{len(retrieved_contexts)} contexto(s) recuperado(s) e "
                f"{len(generation_contexts)} enviado(s) ao gerador."
            ),
            metrics=metrics,
            **report_metadata,
        )


@lru_cache
def get_official_ragas_evaluator() -> OfficialRagasEvaluator:
    return OfficialRagasEvaluator()


def evaluate_ragas(
    query: str,
    answer: str,
    sources: list[RagSource],
    reference_answer: str = "",
    generation_source_ids: list[str] | None = None,
) -> RagasReport:
    return get_official_ragas_evaluator().evaluate(
        query,
        answer,
        sources,
        reference_answer,
        generation_source_ids,
    )
