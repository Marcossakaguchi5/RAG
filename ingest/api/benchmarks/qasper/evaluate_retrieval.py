from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_DIR, parse_k_values, write_json
from benchmarks.sciq.evaluate_retrieval import (
    evaluate,
    load_expected_query_ids,
    load_qrels,
    load_run,
    write_csv_summary,
)

__all__ = ["evaluate", "load_expected_query_ids", "load_qrels", "load_run", "write_csv_summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um run de retrieval QASPER contra qrels de evidência.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    queries_path = args.queries or args.data_dir / "processed" / "queries.jsonl"
    payload = evaluate(
        qrels_path=args.data_dir / "processed" / "qrels.jsonl",
        run_path=args.run,
        split=args.split,
        k_values=parse_k_values(args.k),
        expected_query_ids=load_expected_query_ids(queries_path, args.split, args.limit_queries),
    )
    output_json = args.output_json or args.data_dir / "results" / f"{args.run.stem}_metrics.json"
    output_csv = args.output_csv or args.data_dir / "results" / f"{args.run.stem}_metrics.csv"
    write_json(output_json, payload)
    write_csv_summary(output_csv, payload)
    print(f"Métricas salvas em {output_json} e {output_csv}.")


if __name__ == "__main__":
    main()
