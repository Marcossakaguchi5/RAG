import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_experiment import (  # noqa: E402
    DEFAULT_RUNS,
    ExperimentError,
    build_parser,
    main,
    require_reviewed_cases,
    safe_collection_name,
)


class RunExperimentTests(unittest.TestCase):
    def test_default_runs_are_centralized_under_novas(self):
        self.assertEqual(DEFAULT_RUNS.parts[-2:], ("runs", "novas"))

    def test_collection_name_is_valid_and_bounded(self):
        name = safe_collection_name("Livro com espaços", "docling/hybrid", "x" * 90)

        self.assertLessEqual(len(name), 64)
        self.assertRegex(name, r"^[A-Za-z0-9_-]+$")

    def test_rag_defaults_to_no_reranker(self):
        parser = build_parser()
        args = parser.parse_args(
            ["rag", "--cases", "cases.jsonl", "--collection", "collection"]
        )

        self.assertFalse(args.use_reranker)
        self.assertFalse(args.evaluate_ragas)
        self.assertEqual(args.methods, "bm25,dense,hybrid")
        self.assertFalse(args.allow_draft_cases)

    def test_draft_cases_require_explicit_pilot_flag(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cases = Path(temporary_directory) / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "provenance": {"review_status": "draft"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ExperimentError, "--allow-draft-cases"):
                require_reviewed_cases(cases, allow_draft=False)
            self.assertEqual(
                require_reviewed_cases(cases, allow_draft=True),
                {"draft": 1},
            )

    def test_sciq_dry_run_writes_a_non_destructive_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            exit_code = main(
                [
                    "sciq",
                    "--dry-run",
                    "--run-id",
                    "sciq-test",
                    "--run-dir",
                    str(run_dir),
                    "--limit-queries",
                    "5",
                ]
            )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "dry_run")
            self.assertEqual(manifest["parameters"]["limit_queries"], 5)
            self.assertTrue(manifest["parameters"]["recreate"])

    def test_pdf_dry_run_records_explicit_draft_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf = root / "material.pdf"
            cases = root / "cases.jsonl"
            run_dir = root / "run"
            pdf.write_bytes(b"%PDF-test")
            cases.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "query": "Pergunta?",
                        "reference_answer": "Resposta.",
                        "evidence": [
                            {
                                "document_name": "material.pdf",
                                "quote": "Evidencia.",
                                "relevance": 2,
                            }
                        ],
                        "provenance": {"review_status": "draft"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "pdf-ir",
                    "--dry-run",
                    "--run-dir",
                    str(run_dir),
                    "--pdf",
                    str(pdf),
                    "--cases",
                    str(cases),
                    "--allow-draft-cases",
                    "--bootstrap-repetitions",
                    "100",
                ]
            )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "dry_run")
            self.assertTrue(manifest["parameters"]["allow_draft_cases"])
            self.assertEqual(manifest["parameters"]["case_review_statuses"], {"draft": 1})


if __name__ == "__main__":
    unittest.main()
