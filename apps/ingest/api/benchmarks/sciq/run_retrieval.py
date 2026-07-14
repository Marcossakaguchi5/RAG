from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_LANGUAGE,
    configure_benchmark_environment,
    qdrant_connection_hint,
    read_jsonl,
    write_jsonl,
)


def retrieve_with_loaded_services(
    query: str,
    method: str,
    top_k: int,
    collection_name: str,
    embedding_service: object | None,
    sparse_service: object | None,
) -> list[object]:
    from app.services.retrieval import _records_from_points
    from app.services.vector_store import search_dense, search_hybrid, search_sparse

    if method == "bm25":
        if sparse_service is None:
            raise RuntimeError("Serviço sparse não foi carregado para busca BM25.")
        sparse_vector = sparse_service.encode_query(query)
        if sparse_vector is None:
            return []
        return [
            record.to_schema()
            for record in _records_from_points(search_sparse(sparse_vector, top_k, collection_name), "bm25")
        ]

    if embedding_service is None:
        raise RuntimeError("Serviço dense não foi carregado para busca dense/hybrid.")
    dense_vector = embedding_service.encode([query])[0]

    if method == "dense":
        return [
            record.to_schema()
            for record in _records_from_points(search_dense(dense_vector, top_k, collection_name), "dense")
        ]

    if sparse_service is None:
        raise RuntimeError("Serviço sparse não foi carregado para busca hybrid.")
    sparse_vector = sparse_service.encode_query(query)
    if sparse_vector is None:
        return [
            record.to_schema()
            for record in _records_from_points(search_dense(dense_vector, top_k, collection_name), "dense")
        ]
    return [
        record.to_schema()
        for record in _records_from_points(search_hybrid(dense_vector, sparse_vector, top_k, collection_name), "hybrid")
    ]


def run_retrieval(
    queries_path: Path,
    output_path: Path,
    collection_name: str,
    method: str,
    split: str,
    top_k: int,
    limit_queries: int | None,
) -> dict[str, float | int | str]:
    from app.services.embeddings import get_embedding_service
    from app.services.sparse_embeddings import get_sparse_embedding_service
    from app.services.vector_store import ensure_collection

    try:
        ensure_collection(collection_name)
    except Exception as error:
        from app.core.config import get_settings

        raise RuntimeError(qdrant_connection_hint(get_settings().qdrant_url)) from error

    print("Carregando modelos necessários uma vez para esta execução...")
    embedding_service = get_embedding_service() if method in {"dense", "hybrid"} else None
    sparse_service = get_sparse_embedding_service() if method in {"bm25", "hybrid"} else None

    rows = [row for row in read_jsonl(queries_path) if row["split"] == split]
    if limit_queries is not None:
        rows = rows[:limit_queries]

    run_rows = []
    started_at = perf_counter()
    for position, row in enumerate(rows, start=1):
        query_started_at = perf_counter()
        candidates = retrieve_with_loaded_services(
            str(row["question"]),
            method,
            top_k * 4,
            collection_name,
            embedding_service,
            sparse_service,
        )
        results = []
        seen_documents = set()
        for candidate in candidates:
            if candidate.document_id in seen_documents:
                continue
            seen_documents.add(candidate.document_id)
            results.append(candidate)
            if len(results) >= top_k:
                break
        latency_ms = round((perf_counter() - query_started_at) * 1000, 3)
        for rank, result in enumerate(results, start=1):
            run_rows.append(
                {
                    "query_id": row["query_id"],
                    "split": split,
                    "method": method,
                    "doc_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "rank": rank,
                    "score": result.score,
                    "latency_ms": latency_ms,
                }
            )
        if position % 50 == 0 or position == len(rows):
            print(f"{method} {split}: {position}/{len(rows)} queries processadas...")

    write_jsonl(output_path, run_rows)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    return {
        "method": method,
        "split": split,
        "queries": len(rows),
        "rows": len(run_rows),
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda retrieval SciQ e salva o run em JSONL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--collection", default="sciq_baseline")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--sparse-language", default=DEFAULT_SPARSE_LANGUAGE)
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    output_path = args.output or args.data_dir / "runs" / f"{args.method}_{args.split}.jsonl"
    summary = run_retrieval(
        queries_path=args.data_dir / "processed" / "queries.jsonl",
        output_path=output_path,
        collection_name=args.collection,
        method=args.method,
        split=args.split,
        top_k=args.top_k,
        limit_queries=args.limit_queries,
    )
    print(f"Run salvo em {output_path}. Resumo: {summary}")


if __name__ == "__main__":
    main()
