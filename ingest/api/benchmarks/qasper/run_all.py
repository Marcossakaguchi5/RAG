from __future__ import annotations

import argparse
from datetime import datetime
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
from evaluate_retrieval import evaluate, load_expected_query_ids, write_csv_summary
from export_groundtruth import (
    DEFAULT_CHUNK_MIN_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_OUTPUT_PATH,
    build_groundtruth,
)
from ingest_corpus import SUPPORTED_TEXT_CHUNKING_STRATEGIES, ingest
from plot_results import plot
from prepare_qasper import DEFAULT_DATASET_REVISION, prepare
from run_retrieval import run_retrieval


def parse_methods(raw: str) -> list[str]:
    allowed = {"bm25", "dense", "hybrid"}
    methods = [method.strip() for method in raw.split(",") if method.strip()]
    invalid = sorted(set(methods) - allowed)
    if invalid or not methods:
        raise ValueError("Métodos inválidos. Use uma lista não vazia de bm25,dense,hybrid.")
    return methods


def default_run_dir(data_dir: Path) -> Path:
    return data_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    started_at = perf_counter()
    methods = parse_methods(args.methods)
    k_values = parse_k_values(args.k)
    run_dir = args.run_dir or default_run_dir(args.data_dir)
    summary: dict[str, Any] = {
        "dataset": "allenai/qasper",
        "run_dir": str(run_dir),
        "collection": args.collection,
        "split": args.split,
        "methods": methods,
        "top_k": args.top_k,
        "k_values": k_values,
        "chunking_strategy": args.chunking_strategy,
        "dataset_revision_requested": args.dataset_revision,
        "filters": {"answerable_only": args.answerable_only, "text_evidence_only": args.text_evidence_only},
        "steps": {},
    }
    if not args.skip_prepare:
        print("Preparando QASPER...")
        summary["steps"]["prepare"] = prepare(
            args.data_dir,
            args.dataset_revision,
            text_evidence_only=args.text_evidence_only,
            answerable_only=args.answerable_only,
        )
    if not args.skip_ingest:
        print(f"Indexando parágrafos na coleção {args.collection}...")
        summary["steps"]["ingest"] = {
            "indexed_chunks": ingest(
                args.data_dir / "processed" / "corpus.jsonl",
                args.collection,
                args.batch_size,
                args.recreate,
                args.chunking_strategy,
            )
        }
    if args.export_groundtruth:
        output_path = args.groundtruth_output or DEFAULT_OUTPUT_PATH
        summary["steps"]["groundtruth_export"] = build_groundtruth(
            args.data_dir, output_path, args.collection, args.split, args.chunking_strategy,
            args.chunk_min_words, args.chunk_size_words, args.chunk_overlap_words,
            args.limit_queries, args.top_k,
        )

    retrieval_summaries: dict[str, Any] = {}
    evaluation_summaries: dict[str, Any] = {}
    if not args.skip_retrieval:
        queries_path = args.data_dir / "processed" / "queries.jsonl"
        expected_query_ids = load_expected_query_ids(queries_path, args.split, args.limit_queries)
        for method in methods:
            run_path = run_dir / "retrieval" / f"{method}_{args.split}.jsonl"
            print(f"Rodando {method} para {len(expected_query_ids)} queries QASPER...")
            retrieval_summaries[method] = run_retrieval(
                queries_path, run_path, args.collection, method, args.split, args.top_k, args.limit_queries
            )
            payload = evaluate(
                args.data_dir / "processed" / "qrels.jsonl", run_path, args.split, k_values, expected_query_ids
            )
            json_path = run_dir / "results" / f"{method}_{args.split}_metrics.json"
            csv_path = run_dir / "results" / f"{method}_{args.split}_metrics.csv"
            write_json(json_path, payload)
            write_csv_summary(csv_path, payload)
            evaluation_summaries[method] = {"json": str(json_path), "csv": str(csv_path), "metrics": payload["metrics"]}
    summary["steps"]["retrieval"] = retrieval_summaries
    summary["steps"]["evaluation"] = evaluation_summaries
    if args.plot_results and not args.skip_retrieval:
        summary["steps"]["plots"] = plot(
            argparse.Namespace(data_dir=args.data_dir, run_dir=run_dir, output_dir=args.plots_output, k=max(k_values))
        )
    summary["elapsed_seconds"] = round(perf_counter() - started_at, 3)
    summary_path = run_dir / f"summary_{args.split}.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o benchmark textual QASPER completo.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--collection", default="qasper_text_baseline")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--sparse-language", default=DEFAULT_SPARSE_LANGUAGE)
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--methods", default="bm25,dense,hybrid")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunking-strategy", choices=sorted(SUPPORTED_TEXT_CHUNKING_STRATEGIES), default="recursive_text")
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--export-groundtruth", action="store_true")
    parser.add_argument("--groundtruth-output", type=Path, default=None)
    parser.add_argument("--plot-results", action="store_true")
    parser.add_argument("--plots-output", type=Path, default=None)
    parser.add_argument("--chunk-min-words", type=int, default=DEFAULT_CHUNK_MIN_WORDS)
    parser.add_argument("--chunk-size-words", type=int, default=DEFAULT_CHUNK_SIZE_WORDS)
    parser.add_argument("--chunk-overlap-words", type=int, default=DEFAULT_CHUNK_OVERLAP_WORDS)
    parser.add_argument("--include-unanswerable", dest="answerable_only", action="store_false")
    parser.add_argument("--include-float-evidence", dest="text_evidence_only", action="store_false")
    parser.set_defaults(answerable_only=True, text_evidence_only=True)
    summary = run_all(parser.parse_args())
    print(f"Benchmark QASPER finalizado em {summary['elapsed_seconds']}s. Resumo: {summary['summary_path']}")


if __name__ == "__main__":
    main()
