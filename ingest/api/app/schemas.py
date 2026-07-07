from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RetrievalMethod = Literal["bm25", "dense", "hybrid"]


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: int


class DocumentOut(BaseModel):
    id: str
    original_name: str
    collection_name: str
    chunking_strategy: str = "recursive_text"
    size_bytes: int
    page_count: int
    chunks_count: int
    created_at: datetime


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class CollectionOut(BaseModel):
    name: str
    documents_count: int = 0


class UploadError(BaseModel):
    filename: str
    detail: str


class UploadResponse(BaseModel):
    documents: list[DocumentOut]
    errors: list[UploadError] = []


class PointPreview(BaseModel):
    id: str
    document_id: str
    document_name: str
    file_name: str = ""
    page_number: int
    ordinal: int
    chunk_index: int = 0
    chunk_total: int = 0
    word_count: int = 0
    char_count: int = 0
    chunking_strategy: str = "recursive_text"
    content: str


class PointsResponse(BaseModel):
    total: int
    points: list[PointPreview]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    collection_name: str = Field(default="rag_chunks", min_length=1, max_length=64)
    method: RetrievalMethod = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    relevant_chunk_ids: list[str] = Field(default_factory=list)


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


class EvaluationMetrics(BaseModel):
    evaluated: bool
    message: str | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    map: float | None = None
    ndcg_at_k: float | None = None
    mrr: float | None = None


class SearchResponse(BaseModel):
    method: RetrievalMethod
    top_k: int
    results: list[SearchHit]
    metrics: EvaluationMetrics
