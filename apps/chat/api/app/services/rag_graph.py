import logging
import time
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.schemas import RagRequest, RagResponse, RagSource, RagasEvaluationRequest, RagasReport, SearchHit
from app.services.answering import generate_answer, select_generation_sources
from app.services.ingest_client import get_ingest_client
from app.services.ragas import evaluate_ragas
from app.services.reranker import rerank

logger = logging.getLogger(__name__)


class RagGraphState(TypedDict, total=False):
    payload: RagRequest
    started_at: float
    candidate_k: int
    hits: list[SearchHit]
    sources: list[RagSource]
    generation_source_ids: list[str]
    answer: str
    ragas: RagasReport


def _prepare(state: RagGraphState) -> RagGraphState:
    payload = state["payload"]
    return {
        "started_at": time.perf_counter(),
        "candidate_k": max(payload.top_k, payload.candidate_k),
    }


def _retrieve(state: RagGraphState) -> RagGraphState:
    payload = state["payload"]
    hits = get_ingest_client().search(
        query=payload.query,
        collection_name=payload.collection_name,
        method=payload.method,
        top_k=state["candidate_k"],
    )
    return {"hits": hits}


def _rerank(state: RagGraphState) -> RagGraphState:
    payload = state["payload"]
    sources = rerank(payload.query, state.get("hits", []), payload.top_k, payload.use_reranker)
    return {"sources": sources}


def _answer(state: RagGraphState) -> RagGraphState:
    payload = state["payload"]
    generation_sources = select_generation_sources(state.get("sources", []))
    answer = generate_answer(payload.query, generation_sources)
    return {
        "answer": answer,
        "generation_source_ids": [source.chunk_id for source in generation_sources],
    }


def run_ragas_evaluation(payload: RagasEvaluationRequest) -> RagasReport:
    try:
        return evaluate_ragas(
            payload.query,
            payload.answer,
            payload.sources,
            payload.reference_answer.strip(),
            payload.generation_source_ids,
        )
    except Exception as error:
        logger.exception("Falha ao avaliar RAGAS")
        return RagasReport(evaluated=False, message=f"Nao foi possivel calcular RAGAS: {error}")


def _evaluate(state: RagGraphState) -> RagGraphState:
    payload = state["payload"]
    if not payload.evaluate_ragas:
        return {"ragas": RagasReport(evaluated=False, message="A avaliacao RAGAS foi desativada.")}

    ragas = run_ragas_evaluation(
        RagasEvaluationRequest(
            query=payload.query,
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            generation_source_ids=state.get("generation_source_ids", []),
            reference_answer=payload.reference_answer,
        )
    )
    return {"ragas": ragas}


@lru_cache
def get_rag_graph():
    graph = StateGraph(RagGraphState)
    graph.add_node("prepare", _prepare)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("answer", _answer)
    graph.add_node("evaluate", _evaluate)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()


def run_rag_graph(payload: RagRequest) -> RagResponse:
    state: RagGraphState = get_rag_graph().invoke({"payload": payload})
    started_at = state.get("started_at", time.perf_counter())
    return RagResponse(
        answer=state.get("answer", ""),
        collection_name=payload.collection_name,
        method=payload.method,
        top_k=payload.top_k,
        candidate_k=state.get("candidate_k", max(payload.top_k, payload.candidate_k)),
        used_reranker=payload.use_reranker,
        latency_ms=round((time.perf_counter() - started_at) * 1000),
        sources=state.get("sources", []),
        generation_source_ids=state.get("generation_source_ids", []),
        ragas=state.get("ragas", RagasReport(evaluated=False, message="A avaliacao RAGAS nao foi executada.")),
    )
