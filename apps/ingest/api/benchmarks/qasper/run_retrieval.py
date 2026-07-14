from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_LANGUAGE,
    configure_benchmark_environment,
)
from benchmarks.sciq.run_retrieval import run_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda retrieval QASPER e salva o run em JSONL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--collection", default="qasper_text_baseline")
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
