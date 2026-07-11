import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_sciq import analyze  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class AnalyzeSciqTests(unittest.TestCase):
    def test_missing_run_query_is_zero_and_comparisons_remain_paired(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            queries = root / "processed" / "queries.jsonl"
            qrels = root / "processed" / "qrels.jsonl"
            run_dir = root / "run"
            write_jsonl(
                queries,
                [
                    {"query_id": "q1", "split": "test", "question": "um"},
                    {"query_id": "q2", "split": "test", "question": "dois"},
                ],
            )
            write_jsonl(
                qrels,
                [
                    {"query_id": "q1", "split": "test", "doc_id": "d1", "relevance": 1},
                    {"query_id": "q2", "split": "test", "doc_id": "d2", "relevance": 1},
                ],
            )
            write_jsonl(
                run_dir / "retrieval" / "bm25_test.jsonl",
                [{"query_id": "q1", "split": "test", "doc_id": "d1", "rank": 1}],
            )
            write_jsonl(
                run_dir / "retrieval" / "dense_test.jsonl",
                [
                    {"query_id": "q1", "split": "test", "doc_id": "irrelevant", "rank": 1},
                    {"query_id": "q1", "split": "test", "doc_id": "d1", "rank": 2},
                    {"query_id": "q2", "split": "test", "doc_id": "d2", "rank": 1},
                ],
            )

            summaries, comparisons = analyze(
                run_dir=run_dir,
                queries_path=queries,
                qrels_path=qrels,
                split="test",
                limit_queries=None,
                k_values=[1, 2],
                repetitions=100,
                seed=7,
            )

            bm25_hit_at_1 = next(
                row
                for row in summaries
                if row["method"] == "bm25"
                and row["k"] == 1
                and row["metric"] == "hit_rate_at_k"
            )
            dense_hit_at_2 = next(
                row
                for row in summaries
                if row["method"] == "dense"
                and row["k"] == 2
                and row["metric"] == "hit_rate_at_k"
            )
            hit_comparison_at_1 = next(
                row
                for row in comparisons
                if row["chunking_strategy"] == "sciq@1"
                and row["metric"] == "hit_rate_at_k"
            )

            self.assertEqual(bm25_hit_at_1["queries"], 2)
            self.assertEqual(bm25_hit_at_1["mean"], 0.5)
            self.assertEqual(dense_hit_at_2["mean"], 1.0)
            self.assertEqual(hit_comparison_at_1["queries"], 2)
            self.assertEqual(hit_comparison_at_1["left_wins"], 1)
            self.assertEqual(hit_comparison_at_1["right_wins"], 1)


if __name__ == "__main__":
    unittest.main()
