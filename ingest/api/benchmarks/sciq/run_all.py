from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_LANGUAGE,
    configure_benchmark_environment,
    parse_k_values,
    write_json,
)
from evaluate_retrieval import evaluate, write_csv_summary
from ingest_corpus import ingest
from prepare_sciq import prepare
from run_retrieval import run_retrieval


def parse_methods(raw: str) -> list[str]:
    allowed = {"bm25", "dense", "hybrid"}
    methods = [method.strip() for method in raw.split(",") if method.strip()]
    invalid = sorted(set(methods) - allowed)
    if invalid:
        raise ValueError(f"Métodos inválidos: {', '.join(invalid)}. Use bm25,dense,hybrid.")
    if not methods:
        raise ValueError("Informe ao menos um método.")
    return methods


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    started_at = perf_counter()
    methods = parse_methods(args.methods)
    k_values = parse_k_values(args.k)
    summary: dict[str, Any] = {
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "sparse_language": args.sparse_language,
        "fastembed_cache_dir": str(args.fastembed_cache_dir),
        "split": args.split,
        "methods": methods,
        "top_k": args.top_k,
        "k_values": k_values,
        "steps": {},
    }

    if args.skip_prepare:
        print("Etapa prepare ignorada.")
    else:
        print("Preparando SciQ...")
        summary["steps"]["prepare"] = prepare(
            data_dir=args.data_dir,
            min_support_words=args.min_support_words,
            seed=args.seed,
        )

    if args.skip_ingest:
        print("Etapa ingest ignorada.")
    else:
        print(f"Indexando corpus na coleção {args.collection}...")
        indexed = ingest(
            corpus_path=args.data_dir / "processed" / "corpus.jsonl",
            collection_name=args.collection,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )
        summary["steps"]["ingest"] = {"indexed_documents": indexed}

    retrieval_summaries = {}
    evaluation_summaries = {}
    for method in methods:
        run_path = args.data_dir / "runs" / f"{method}_{args.split}.jsonl"
        print(f"Rodando retrieval: method={method}, split={args.split}, top_k={args.top_k}...")
        retrieval_summaries[method] = run_retrieval(
            queries_path=args.data_dir / "processed" / "queries.jsonl",
            output_path=run_path,
            collection_name=args.collection,
            method=method,
            split=args.split,
            top_k=args.top_k,
            limit_queries=args.limit_queries,
        )

        print(f"Avaliando run: {run_path}...")
        metrics_payload = evaluate(
            qrels_path=args.data_dir / "processed" / "qrels.jsonl",
            run_path=run_path,
            split=args.split,
            k_values=k_values,
        )
        metrics_json_path = args.data_dir / "results" / f"{method}_{args.split}_metrics.json"
        metrics_csv_path = args.data_dir / "results" / f"{method}_{args.split}_metrics.csv"
        write_json(metrics_json_path, metrics_payload)
        write_csv_summary(metrics_csv_path, metrics_payload)
        evaluation_summaries[method] = {
            "json": str(metrics_json_path),
            "csv": str(metrics_csv_path),
            "metrics": metrics_payload["metrics"],
        }

    summary["steps"]["retrieval"] = retrieval_summaries
    summary["steps"]["evaluation"] = evaluation_summaries
    summary["elapsed_seconds"] = round(perf_counter() - started_at, 3)

    summary_path = args.data_dir / "results" / f"summary_{args.split}.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o benchmark SciQ completo.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--collection", default="sciq_baseline")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--sparse-language", default=DEFAULT_SPARSE_LANGUAGE)
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--methods", default="bm25,dense,hybrid")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-support-words", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    summary = run_all(args)
    print(f"Benchmark finalizado em {summary['elapsed_seconds']}s.")
    print(f"Resumo salvo em {summary['summary_path']}.")
    for method, result in summary["steps"]["evaluation"].items():
        print(f"{method}: {result['metrics']}")


if __name__ == "__main__":
    main()
