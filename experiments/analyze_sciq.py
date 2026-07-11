#!/usr/bin/env python3
"""Add confidence intervals and paired comparisons to a SciQ retrieval run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCIQ_DIR = ROOT / "ingest" / "api" / "benchmarks" / "sciq"
if str(SCIQ_DIR) not in sys.path:
    sys.path.insert(0, str(SCIQ_DIR))

from evaluate_retrieval import (  # noqa: E402
    load_expected_query_ids,
    load_qrels,
    load_run,
)
from metrics import calculate_query_metrics  # noqa: E402
from plot_pdf_ir import bootstrap_mean_ci, paired_differences, write_csv  # noqa: E402


METRIC_GETTERS = {
    "hit_rate_at_k": lambda item: item.hit_rate,
    "precision_at_k": lambda item: item.precision,
    "recall_at_k": lambda item: item.recall,
    "map": lambda item: item.average_precision,
    "ndcg_at_k": lambda item: item.ndcg,
    "mrr": lambda item: item.mrr,
}


def analyze(
    *,
    run_dir: Path,
    queries_path: Path,
    qrels_path: Path,
    split: str,
    limit_queries: int | None,
    k_values: list[int],
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = load_expected_query_ids(queries_path, split, limit_queries)
    qrels = load_qrels(qrels_path, split)
    run_paths = sorted((run_dir / "retrieval").glob(f"*_{split}.jsonl"))
    if not run_paths:
        raise ValueError(f"Runs nao encontrados em {run_dir / 'retrieval'}.")

    method_runs = {
        path.name.removesuffix(f"_{split}.jsonl"): load_run(path, split)
        for path in run_paths
    }
    summaries: list[dict[str, Any]] = []
    paired_rows_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in k_values}

    for method_index, (method, run) in enumerate(sorted(method_runs.items())):
        for k_index, k in enumerate(k_values):
            query_metrics = {
                query_id: calculate_query_metrics(
                    run.get(query_id, []),
                    qrels.get(query_id, set()),
                    k,
                )
                for query_id in expected_ids
            }
            for metric_index, (metric, getter) in enumerate(METRIC_GETTERS.items()):
                values = [float(getter(query_metrics[query_id])) for query_id in expected_ids]
                point, low, high = bootstrap_mean_ci(
                    values,
                    repetitions=repetitions,
                    seed=seed + method_index * 1000 + k_index * 100 + metric_index,
                )
                summaries.append(
                    {
                        "method": method,
                        "k": k,
                        "metric": metric,
                        "queries": len(values),
                        "mean": round(point, 6),
                        "ci95_low": round(low, 6),
                        "ci95_high": round(high, 6),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": seed,
                    }
                )
            for query_id in expected_ids:
                item = query_metrics[query_id]
                paired_rows_by_k[k].append(
                    {
                        "chunking_strategy": f"sciq@{k}",
                        "method": method,
                        "case_id": query_id,
                        "status": "ok",
                        "metrics": {
                            metric: float(getter(item))
                            for metric, getter in METRIC_GETTERS.items()
                        },
                    }
                )

    comparisons: list[dict[str, Any]] = []
    for k_index, k in enumerate(k_values):
        comparisons.extend(
            paired_differences(
                paired_rows_by_k[k],
                repetitions=repetitions,
                seed=seed + k_index * 10000,
            )
        )
    return summaries, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise estatistica pareada de uma rodada SciQ.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio de saida; por padrao usa RUN_DIR/statistics.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "ingest" / "api" / "benchmarks" / "sciq" / "data",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise SystemExit("Use pelo menos 100 repeticoes bootstrap.")
    k_values = sorted({int(item.strip()) for item in args.k.split(",") if item.strip()})
    if not k_values or min(k_values) < 1:
        raise SystemExit("Valores de k devem ser positivos.")

    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    summaries, comparisons = analyze(
        run_dir=run_dir,
        queries_path=data_dir / "processed" / "queries.jsonl",
        qrels_path=data_dir / "processed" / "qrels.jsonl",
        split=args.split,
        limit_queries=args.limit_queries,
        k_values=k_values,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    output_dir = (args.output_dir or run_dir / "statistics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_with_ci.csv", summaries)
    if comparisons:
        write_csv(output_dir / "paired_differences.csv", comparisons)
    (output_dir / "metrics_with_ci.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_differences.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "split": args.split,
                "limit_queries": args.limit_queries,
                "k_values": k_values,
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "bootstrap_seed": args.seed,
                "aggregates": len(summaries),
                "paired_comparisons": len(comparisons),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
