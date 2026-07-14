import unittest
from types import SimpleNamespace

from benchmarks.ragas.evaluate_official import prepare_sample, score_value


class RagasBenchmarkTests(unittest.TestCase):
    def test_separates_retrieved_and_generation_contexts(self):
        sample = prepare_sample(
            {
                "query": "pergunta",
                "answer": "resposta",
                "reference_answer": "referencia",
                "generation_source_ids": ["c2"],
                "sources": [
                    {"chunk_id": "c1", "content": "contexto recuperado 1"},
                    {"chunk_id": "c2", "content": "contexto usado"},
                ],
            }
        )

        self.assertEqual(
            sample["retrieved_contexts"],
            ["contexto recuperado 1", "contexto usado"],
        )
        self.assertEqual(sample["generation_contexts"], ["contexto usado"])

    def test_requires_generation_source_ids(self):
        with self.assertRaisesRegex(ValueError, "generation_source_ids"):
            prepare_sample(
                {
                    "query": "pergunta",
                    "answer": "resposta",
                    "reference_answer": "referencia",
                    "sources": [{"chunk_id": "c1", "content": "contexto"}],
                }
            )

    def test_preserves_finite_raw_scores_and_rejects_non_finite(self):
        self.assertEqual(score_value(SimpleNamespace(value=-0.2, reason=None))[0], -0.2)
        self.assertIsNone(score_value(SimpleNamespace(value=float("inf"), reason=None))[0])


if __name__ == "__main__":
    unittest.main()
