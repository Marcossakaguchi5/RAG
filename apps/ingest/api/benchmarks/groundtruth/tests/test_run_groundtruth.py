import math
import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from run_groundtruth import (  # noqa: E402
    calculate_ranked_metrics,
    relevance_grades_for_case,
)


class GroundTruthMetricsTests(unittest.TestCase):
    def test_binary_metrics_match_expected_ranking(self):
        metrics = calculate_ranked_metrics(
            ["irrelevant", "relevant-a", "relevant-b"],
            {"relevant-a": 1.0, "relevant-b": 1.0},
            top_k=3,
        )

        self.assertEqual(metrics["hit_rate_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], round(2 / 3, 6))
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["map"], round(((1 / 2) + (2 / 3)) / 2, 6))

    def test_graded_ndcg_rewards_high_grade_first(self):
        grades = {"partial": 1.0, "sufficient": 2.0}
        ideal = calculate_ranked_metrics(["sufficient", "partial"], grades, top_k=2)
        reversed_ranking = calculate_ranked_metrics(
            ["partial", "sufficient"], grades, top_k=2
        )

        self.assertEqual(ideal["ndcg_at_k"], 1.0)
        self.assertLess(reversed_ranking["ndcg_at_k"], 1.0)
        self.assertTrue(math.isclose(reversed_ranking["recall_at_k"], 1.0))

    def test_relevance_map_must_match_binary_ids(self):
        with self.assertRaisesRegex(ValueError, "mesmos IDs"):
            relevance_grades_for_case(
                {
                    "relevant_chunk_ids": ["a"],
                    "relevance_by_chunk": {"b": 2},
                }
            )

    def test_binary_ids_are_promoted_to_grade_one(self):
        self.assertEqual(
            relevance_grades_for_case({"relevant_chunk_ids": ["a", "b"]}),
            {"a": 1.0, "b": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
