from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_DATA_DIR, parse_k_values, read_jsonl, write_json
from metrics import average_metrics, calculate_query_metrics


def load_qrels(path: Path, split: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        if row["split"] != split:
            continue
        if int(row.get("relevance", 0)) > 0:
            qrels[str(row["query_id"])].add(str(row["doc_id"]))
    return dict(qrels)


def load_run(path: Path, split: str) -> dict[str, list[str]]:
    by_query: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("split") != split:
            continue
        by_query[str(row["query_id"])].append((int(row["rank"]), str(row["doc_id"])))
    return {
        query_id: [doc_id for _, doc_id in sorted(items, key=lambda item: item[0])]
        for query_id, items in by_query.items()
    }


def evaluate(qrels_path: Path, run_path: Path, split: str, k_values: list[int]) -> dict[str, Any]:
    qrels = load_qrels(qrels_path, split)
    run = load_run(run_path, split)

    metrics_by_k = {}
    for k in k_values:
        per_query = [
            calculate_query_metrics(run.get(query_id, []), relevant_ids, k)
            for query_id, relevant_ids in qrels.items()
        ]
        metrics_by_k[f"@{k}"] = average_metrics(per_query)

    return {
        "split": split,
        "qrels_queries": len(qrels),
        "run_queries": len(run),
        "missing_queries": len(set(qrels) - set(run)),
        "run_path": str(run_path),
        "metrics": metrics_by_k,
    }


def write_csv_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "k", "hit_rate", "precision", "recall", "map", "ndcg", "mrr"],
        )
        writer.writeheader()
        for k_label, metrics in payload["metrics"].items():
            writer.writerow({"split": payload["split"], "k": k_label.removeprefix("@"), **metrics})


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um run de retrieval SciQ contra qrels.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    k_values = parse_k_values(args.k)
    payload = evaluate(
        qrels_path=args.data_dir / "processed" / "qrels.jsonl",
        run_path=args.run,
        split=args.split,
        k_values=k_values,
    )

    stem = args.run.stem
    output_json = args.output_json or args.data_dir / "results" / f"{stem}_metrics.json"
    output_csv = args.output_csv or args.data_dir / "results" / f"{stem}_metrics.csv"
    write_json(output_json, payload)
    write_csv_summary(output_csv, payload)

    print(f"Métricas salvas em {output_json} e {output_csv}.")
    for k_label, metrics in payload["metrics"].items():
        print(f"{k_label}: {metrics}")


if __name__ == "__main__":
    main()
