import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_pdf_ir import (  # noqa: E402
    bootstrap_mean_ci,
    paired_chunking_differences,
    paired_differences,
    summarize,
)


def row(method: str, mrr: float, ndcg: float, recall: float):
    return {
        "chunking_strategy": "recursive_text",
        "method": method,
        "status": "ok",
        "metrics": {
            "mrr": mrr,
            "ndcg_at_k": ndcg,
            "recall_at_k": recall,
        },
    }


class PlotPdfIrTests(unittest.TestCase):
    def test_bootstrap_is_reproducible(self):
        first = bootstrap_mean_ci([0.0, 0.5, 1.0], repetitions=200, seed=42)
        second = bootstrap_mean_ci([0.0, 0.5, 1.0], repetitions=200, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.5)

    def test_summary_keeps_method_conditions_separate(self):
        summaries = summarize(
            [
                row("bm25", 1.0, 1.0, 1.0),
                row("bm25", 0.5, 0.7, 1.0),
                row("dense", 0.0, 0.0, 0.0),
                row("dense", 0.5, 0.4, 1.0),
            ],
            repetitions=200,
            seed=42,
        )

        methods = {(item["method"], item["metric"]) for item in summaries}
        self.assertEqual(len(summaries), 6)
        self.assertIn(("bm25", "mrr"), methods)
        self.assertIn(("dense", "ndcg_at_k"), methods)

    def test_paired_output_contains_mcnemar_and_holm(self):
        rows = []
        for case_id, bm25_hit, dense_hit in (("q1", 1.0, 0.0), ("q2", 1.0, 1.0)):
            for method, hit in (("bm25", bm25_hit), ("dense", dense_hit)):
                item = row(method, hit, hit, hit)
                item["case_id"] = case_id
                item["metrics"]["hit_rate_at_k"] = hit
                rows.append(item)

        comparisons = paired_differences(rows, repetitions=200, seed=42)
        hit = next(item for item in comparisons if item["metric"] == "hit_rate_at_k")
        self.assertEqual(hit["queries"], 2)
        self.assertIsNotNone(hit["mcnemar_exact_p"])
        self.assertIsNotNone(hit["mcnemar_holm_p"])

    def test_chunking_comparison_keeps_method_fixed(self):
        rows = []
        for case_id, recursive_hit, fixed_hit in (
            ("q1", 1.0, 0.0),
            ("q2", 1.0, 1.0),
        ):
            for strategy, hit in (
                ("recursive_text", recursive_hit),
                ("fixed_token", fixed_hit),
            ):
                item = row("hybrid", hit, hit, hit)
                item["chunking_strategy"] = strategy
                item["case_id"] = case_id
                item["metrics"]["hit_rate_at_k"] = hit
                rows.append(item)

        comparisons = paired_chunking_differences(
            rows,
            repetitions=200,
            seed=42,
        )
        hit = next(item for item in comparisons if item["metric"] == "hit_rate_at_k")
        self.assertEqual(hit["method"], "hybrid")
        self.assertEqual(hit["queries"], 2)
        self.assertEqual(hit["left_chunking_strategy"], "fixed_token")
        self.assertEqual(hit["right_chunking_strategy"], "recursive_text")
        self.assertEqual(hit["right_wins"], 1)


if __name__ == "__main__":
    unittest.main()
