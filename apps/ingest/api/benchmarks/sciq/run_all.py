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
from plot_results import plot as plot_benchmark_results
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


def default_run_dir(data_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return data_dir / "runs" / timestamp


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    started_at = perf_counter()
    methods = parse_methods(args.methods)
    k_values = parse_k_values(args.k)
    run_dir = args.run_dir or default_run_dir(args.data_dir)
    retrieval_dir = run_dir / "retrieval"
    results_dir = run_dir / "results"
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "sparse_language": args.sparse_language,
        "fastembed_cache_dir": str(args.fastembed_cache_dir),
        "split": args.split,
        "methods": methods,
        "top_k": args.top_k,
        "k_values": k_values,
        "limit_queries": args.limit_queries,
        "chunking_strategy": args.chunking_strategy,
        "dataset_revision_requested": args.dataset_revision,
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
            dataset_revision=args.dataset_revision,
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
            chunking_strategy=args.chunking_strategy,
        )
        summary["steps"]["ingest"] = {"indexed_documents": indexed}

    if args.export_groundtruth:
        output_path = args.groundtruth_output or DEFAULT_OUTPUT_PATH
        print(f"Exportando ground truth de chunks em {output_path}...")
        summary["steps"]["groundtruth_export"] = build_groundtruth(
            data_dir=args.data_dir,
            output_path=output_path,
            collection_name=args.collection,
            split=args.split,
            chunking_strategy=args.chunking_strategy,
            chunk_min_words=args.chunk_min_words,
            chunk_size_words=args.chunk_size_words,
            chunk_overlap_words=args.chunk_overlap_words,
            limit_queries=args.limit_queries,
            top_k=args.top_k,
        )

    retrieval_summaries = {}
    evaluation_summaries = {}
    if args.skip_retrieval:
        print("Etapa retrieval/evaluation ignorada.")
    else:
        queries_path = args.data_dir / "processed" / "queries.jsonl"
        expected_query_ids = load_expected_query_ids(
            queries_path,
            split=args.split,
            limit_queries=args.limit_queries,
        )
        for method in methods:
            run_path = retrieval_dir / f"{method}_{args.split}.jsonl"
            print(f"Rodando retrieval: method={method}, split={args.split}, top_k={args.top_k}...")
            retrieval_summaries[method] = run_retrieval(
                queries_path=queries_path,
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
                expected_query_ids=expected_query_ids,
            )
            metrics_json_path = results_dir / f"{method}_{args.split}_metrics.json"
            metrics_csv_path = results_dir / f"{method}_{args.split}_metrics.csv"
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

    if args.plot_results:
        if args.skip_retrieval:
            print("Etapa plots ignorada porque retrieval/evaluation foi pulada.")
        else:
            print("Gerando graficos...")
            summary["steps"]["plots"] = plot_benchmark_results(
                argparse.Namespace(
                    data_dir=args.data_dir,
                    run_dir=run_dir,
                    output_dir=args.plots_output,
                    k=max(k_values),
                )
            )

    summary_path = run_dir / f"summary_{args.split}.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o benchmark SciQ completo.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
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
    parser.add_argument("--chunking-strategy", choices=sorted(SUPPORTED_TEXT_CHUNKING_STRATEGIES), default="recursive_text")
    parser.add_argument("--min-support-words", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default="main")
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
    args = parser.parse_args()

    summary = run_all(args)
    print(f"Benchmark finalizado em {summary['elapsed_seconds']}s.")
    print(f"Resumo salvo em {summary['summary_path']}.")
    for method, result in summary["steps"]["evaluation"].items():
        print(f"{method}: {result['metrics']}")


if __name__ == "__main__":
    main()
