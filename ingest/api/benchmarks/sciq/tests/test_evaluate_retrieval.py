from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCIQ_DIR = Path(__file__).resolve().parents[1]
if str(SCIQ_DIR) not in sys.path:
    sys.path.insert(0, str(SCIQ_DIR))

from evaluate_retrieval import (  # noqa: E402
    build_argument_parser,
    evaluate,
    load_expected_query_ids,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluateRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.queries_path = self.directory / "queries.jsonl"
        self.qrels_path = self.directory / "qrels.jsonl"
        self.run_path = self.directory / "run.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_limit_uses_expected_query_subset_and_missing_run_scores_zero(self) -> None:
        write_jsonl(
            self.queries_path,
            [
                {"query_id": "q1", "split": "test", "question": "one"},
                {"query_id": "q2", "split": "test", "question": "two"},
                {"query_id": "q3", "split": "test", "question": "three"},
            ],
        )
        write_jsonl(
            self.qrels_path,
            [
                {"query_id": "q1", "split": "test", "doc_id": "d1", "relevance": 1},
                {"query_id": "q2", "split": "test", "doc_id": "d2", "relevance": 1},
                {"query_id": "q3", "split": "test", "doc_id": "d3", "relevance": 1},
            ],
        )
        write_jsonl(
            self.run_path,
            [
                {"query_id": "q1", "split": "test", "doc_id": "d1", "rank": 1},
                {"query_id": "q3", "split": "test", "doc_id": "d3", "rank": 1},
            ],
        )

        expected_query_ids = load_expected_query_ids(self.queries_path, "test", limit_queries=2)
        payload = evaluate(
            self.qrels_path,
            self.run_path,
            split="test",
            k_values=[1],
            expected_query_ids=expected_query_ids,
        )

        self.assertEqual(expected_query_ids, ["q1", "q2"])
        self.assertEqual(payload["expected_queries"], 2)
        self.assertEqual(payload["qrels_queries"], 2)
        self.assertEqual(payload["run_queries"], 1)
        self.assertEqual(payload["missing_queries"], 1)
        self.assertEqual(payload["unexpected_run_queries"], 1)
        self.assertEqual(payload["metrics"]["@1"]["hit_rate"], 0.5)
        self.assertEqual(payload["metrics"]["@1"]["map"], 0.5)

    def test_no_limit_evaluates_the_complete_selected_split(self) -> None:
        write_jsonl(
            self.queries_path,
            [
                {"query_id": "q1", "split": "test", "question": "one"},
                {"query_id": "v1", "split": "validation", "question": "validation"},
                {"query_id": "q2", "split": "test", "question": "two"},
            ],
        )
        write_jsonl(
            self.qrels_path,
            [
                {"query_id": "q1", "split": "test", "doc_id": "d1", "relevance": 1},
                {"query_id": "q2", "split": "test", "doc_id": "d2", "relevance": 1},
                {"query_id": "v1", "split": "validation", "doc_id": "dv", "relevance": 1},
            ],
        )
        write_jsonl(
            self.run_path,
            [
                {"query_id": "q1", "split": "test", "doc_id": "d1", "rank": 1},
                {"query_id": "q2", "split": "test", "doc_id": "d2", "rank": 1},
                {"query_id": "v1", "split": "validation", "doc_id": "dv", "rank": 1},
            ],
        )

        expected_query_ids = load_expected_query_ids(self.queries_path, "test")
        payload = evaluate(
            self.qrels_path,
            self.run_path,
            split="test",
            k_values=[1],
            expected_query_ids=expected_query_ids,
        )

        self.assertEqual(expected_query_ids, ["q1", "q2"])
        self.assertEqual(payload["expected_queries"], 2)
        self.assertEqual(payload["missing_queries"], 0)
        self.assertEqual(payload["metrics"]["@1"]["hit_rate"], 1.0)

    def test_expected_query_without_qrel_stays_in_metric_denominator(self) -> None:
        write_jsonl(
            self.queries_path,
            [
                {"query_id": "q1", "split": "test", "question": "one"},
                {"query_id": "q2", "split": "test", "question": "two"},
            ],
        )
        write_jsonl(
            self.qrels_path,
            [{"query_id": "q1", "split": "test", "doc_id": "d1", "relevance": 1}],
        )
        write_jsonl(
            self.run_path,
            [{"query_id": "q1", "split": "test", "doc_id": "d1", "rank": 1}],
        )

        payload = evaluate(
            self.qrels_path,
            self.run_path,
            split="test",
            k_values=[1],
            expected_query_ids=load_expected_query_ids(self.queries_path, "test"),
        )

        self.assertEqual(payload["expected_queries"], 2)
        self.assertEqual(payload["qrels_queries"], 1)
        self.assertEqual(payload["missing_qrels_queries"], 1)
        self.assertEqual(payload["metrics"]["@1"]["recall"], 0.5)

    def test_cli_exposes_queries_path_and_limit(self) -> None:
        args = build_argument_parser().parse_args(
            [
                "--run",
                str(self.run_path),
                "--queries",
                str(self.queries_path),
                "--limit-queries",
                "7",
            ]
        )

        self.assertEqual(args.queries, self.queries_path)
        self.assertEqual(args.limit_queries, 7)


if __name__ == "__main__":
    unittest.main()
