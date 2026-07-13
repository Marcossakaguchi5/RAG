from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


QASPER_DIR = Path(__file__).resolve().parents[1]
if str(QASPER_DIR) not in sys.path:
    sys.path.insert(0, str(QASPER_DIR))

from common import read_jsonl  # noqa: E402
from prepare_qasper import prepare_from_dataset  # noqa: E402


class PrepareQasperTests(unittest.TestCase):
    def test_parquet_shape_creates_paragraph_qrels_and_excludes_float_evidence(self) -> None:
        dataset = {
            "test": [
                {
                    "id": "paper-1",
                    "full_text": [
                        {"section_name": "Intro", "paragraphs": ["First evidence.", "Other paragraph."]},
                        {"section_name": "Results", "paragraphs": ["Second evidence."]},
                    ],
                    "qas": [
                        {
                            "question": "What was found?",
                            "question_id": "q-text",
                            "answers": [
                                {
                                    "answer": {
                                        "unanswerable": False,
                                        "free_form_answer": "A finding.",
                                        "extractive_spans": [],
                                        "yes_no": None,
                                        "evidence": ["First evidence.", "Second evidence."],
                                    }
                                }
                            ],
                        },
                        {
                            "question": "What is in Table 1?",
                            "question_id": "q-float",
                            "answers": [
                                {
                                    "answer": {
                                        "unanswerable": False,
                                        "free_form_answer": "A table finding.",
                                        "extractive_spans": [],
                                        "yes_no": None,
                                        "evidence": ["FLOAT SELECTED: Table 1"],
                                    }
                                }
                            ],
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = prepare_from_dataset(dataset, root)
            queries = list(read_jsonl(root / "processed" / "queries.jsonl"))
            qrels = list(read_jsonl(root / "processed" / "qrels.jsonl"))

        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["queries"], 1)
        self.assertEqual(summary["qrels"], 2)
        self.assertEqual(summary["skipped"]["float_evidence"], 1)
        self.assertEqual(queries[0]["reference_answer"], "A finding.")
        self.assertEqual(len(queries[0]["relevant_doc_ids"]), 2)
        self.assertEqual({row["relevance"] for row in qrels}, {2})


if __name__ == "__main__":
    unittest.main()
