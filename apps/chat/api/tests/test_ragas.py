import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas import RagSource
from app.services.answering import _format_source, select_generation_sources
from app.services.ragas import OfficialRagasEvaluator, _generation_contexts, _score_value


class ScoreMetric:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.calls: list[dict] = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value, reason=None)


class FailingMetric:
    def score(self, **kwargs):
        raise RuntimeError("falha simulada")


def source(chunk_id: str, content: str, rank: int) -> RagSource:
    return RagSource(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_name="documento",
        page_number=1,
        ordinal=rank - 1,
        content=content,
        score=1.0,
        rank=rank,
        retrieval_rank=rank,
    )


def evaluator(metric) -> OfficialRagasEvaluator:
    instance = object.__new__(OfficialRagasEvaluator)
    instance.ragas_version = "0.4.3"
    instance.evaluator_model = "judge-model"
    instance.embedding_model = "embedding-model"
    instance.faithfulness = metric
    instance.answer_relevancy = metric
    instance.context_precision = metric
    instance.context_utilization = metric
    instance.context_recall = metric
    instance.factual_correctness = metric
    return instance


class OfficialRagasEvaluatorTests(unittest.TestCase):
    def test_generation_source_selection_accounts_for_separators(self):
        first = source("c1", "primeiro contexto", 1)
        second = source("c2", "segundo contexto", 2)
        limit = len(_format_source(first)) + len("\n\n---\n\n") + len(_format_source(second)) - 1

        with patch(
            "app.services.answering.get_settings",
            return_value=SimpleNamespace(max_context_characters=limit),
        ):
            selected = select_generation_sources([first, second])

        self.assertEqual([item.chunk_id for item in selected], ["c1"])

    def test_preserves_finite_raw_scores(self):
        self.assertEqual(_score_value(SimpleNamespace(value=-0.4, reason=None))[0], -0.4)
        self.assertIsNone(_score_value(SimpleNamespace(value=float("nan"), reason=None))[0])

    def test_uses_generation_contexts_only_for_faithfulness(self):
        sources = [source("c1", "contexto recuperado 1", 1), source("c2", "contexto usado", 2)]
        instance = evaluator(ScoreMetric())
        instance.faithfulness = ScoreMetric()
        instance.context_precision = ScoreMetric()

        report = instance.evaluate("pergunta", "resposta", sources, "referencia", ["c2"])

        self.assertTrue(report.evaluated)
        self.assertEqual(
            instance.faithfulness.calls[0]["retrieved_contexts"],
            ["contexto usado"],
        )
        self.assertEqual(
            instance.context_precision.calls[0]["retrieved_contexts"],
            ["contexto recuperado 1", "contexto usado"],
        )
        self.assertEqual(report.generation_contexts_count, 1)
        self.assertEqual(report.retrieved_contexts_count, 2)

    def test_names_context_utilization_separately_without_reference(self):
        report = evaluator(ScoreMetric()).evaluate(
            "pergunta",
            "resposta",
            [source("c1", "contexto", 1)],
            generation_source_ids=["c1"],
        )

        self.assertIn("Context utilization", [metric.name for metric in report.metrics])
        self.assertNotIn("Context precision", [metric.name for metric in report.metrics])

    def test_does_not_mark_report_evaluated_when_all_calls_fail(self):
        report = evaluator(FailingMetric()).evaluate(
            "pergunta",
            "resposta",
            [source("c1", "contexto", 1)],
            "referencia",
            ["c1"],
        )

        self.assertFalse(report.evaluated)
        self.assertTrue(all(metric.value is None for metric in report.metrics))

    def test_rejects_generation_ids_not_present_in_sources(self):
        with self.assertRaisesRegex(ValueError, "chunks ausentes"):
            _generation_contexts([source("c1", "contexto", 1)], ["c2"])


if __name__ == "__main__":
    unittest.main()
