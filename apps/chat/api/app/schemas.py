from typing import Literal

from pydantic import BaseModel, Field


RetrievalMethod = Literal["bm25", "dense", "hybrid"]


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: int


class CollectionOut(BaseModel):
    name: str
    documents_count: int = 0


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    ordinal: int
    content: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None


class RagSource(SearchHit):
    rank: int
    retrieval_rank: int
    rerank_score: float | None = None


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    collection_name: str = Field(default="rag_chunks", min_length=1, max_length=64)
    method: RetrievalMethod = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=50)
    use_reranker: bool = True
    evaluate_ragas: bool = False
    reference_answer: str = Field(default="", max_length=12000)


class RagasMetric(BaseModel):
    name: str
    value: float | None = None
    reason: str | None = None


class RagasReport(BaseModel):
    evaluated: bool
    message: str | None = None
    metrics: list[RagasMetric] = Field(default_factory=list)
    ragas_version: str | None = None
    evaluator_model: str | None = None
    embedding_model: str | None = None
    retrieved_contexts_count: int = 0
    generation_contexts_count: int = 0


class RagResponse(BaseModel):
    answer: str
    collection_name: str
    method: RetrievalMethod
    top_k: int
    candidate_k: int
    used_reranker: bool
    latency_ms: int
    sources: list[RagSource]
    generation_source_ids: list[str] = Field(default_factory=list)
    ragas: RagasReport


class RagasEvaluationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)
    sources: list[RagSource] = Field(default_factory=list)
    generation_source_ids: list[str] | None = None
    reference_answer: str = Field(default="", max_length=12000)
