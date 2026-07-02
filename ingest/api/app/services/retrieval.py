from dataclasses import dataclass

from app.schemas import SearchHit
from app.services.collections import normalize_collection_name
from app.services.embeddings import get_embedding_service
from app.services.sparse_embeddings import get_sparse_embedding_service
from app.services.vector_store import search_dense, search_hybrid, search_sparse


@dataclass
class RetrievedRecord:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    ordinal: int
    content: str
    dense_score: float | None = None
    bm25_score: float | None = None
    score: float = 0.0

    def to_schema(self) -> SearchHit:
        return SearchHit(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            document_name=self.document_name,
            page_number=self.page_number,
            ordinal=self.ordinal,
            content=self.content,
            score=round(self.score, 6),
            dense_score=round(self.dense_score, 6) if self.dense_score is not None else None,
            bm25_score=round(self.bm25_score, 6) if self.bm25_score is not None else None,
        )


def _records_from_points(points: list[object], score_type: str) -> list[RetrievedRecord]:
    records: list[RetrievedRecord] = []
    for point in points:
        payload = point.payload or {}
        raw_score = float(point.score)
        records.append(
            RetrievedRecord(
                chunk_id=str(payload.get("chunk_id", point.id)),
                document_id=str(payload.get("document_id", "")),
                document_name=str(payload.get("document_name", "Documento sem nome")),
                page_number=int(payload.get("page_number", 0)),
                ordinal=int(payload.get("ordinal", 0)),
                content=str(payload.get("content", "")),
                dense_score=raw_score if score_type == "dense" else None,
                bm25_score=raw_score if score_type == "bm25" else None,
                score=raw_score,
            )
        )
    return records


def retrieve(query: str, method: str, top_k: int, collection_name: str) -> list[SearchHit]:
    collection_name = normalize_collection_name(collection_name)
    dense_vector = get_embedding_service().encode([query])[0]
    if method == "bm25":
        sparse_vector = get_sparse_embedding_service().encode_query(query)
        if sparse_vector is None:
            return []
        return [
            record.to_schema()
            for record in _records_from_points(search_sparse(sparse_vector, top_k, collection_name), "bm25")
        ]

    if method == "dense":
        return [
            record.to_schema()
            for record in _records_from_points(search_dense(dense_vector, top_k, collection_name), "dense")
        ]

    sparse_vector = get_sparse_embedding_service().encode_query(query)
    if sparse_vector is None:
        return [
            record.to_schema()
            for record in _records_from_points(search_dense(dense_vector, top_k, collection_name), "dense")
        ]
    points = search_hybrid(dense_vector, sparse_vector, top_k, collection_name)
    return [record.to_schema() for record in _records_from_points(points, "hybrid")]
