import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from materialize import (  # noqa: E402
    UnmappedEvidenceError,
    materialize_files,
    materialize_records,
    normalize_text,
)


def case(quote="Recuperação de informação encontra documentos relevantes."):
    return {
        "id": "q1",
        "query": "O que faz a recuperação de informação?",
        "reference_answer": "Ela encontra documentos relevantes.",
        "evidence": [
            {
                "document_name": "ri.pdf",
                "quote": quote,
                "relevance": 2,
            }
        ],
    }


def chunk(content="Recuperação de informação encontra documentos relevantes."):
    return {
        "chunk_id": "chunk-1",
        "document_name": "ri.pdf",
        "content": content,
        "page_number": 3,
    }


class MaterializeTests(unittest.TestCase):
    def test_exact_match_maps_chunk_and_grade(self):
        ingest, ragas, report = materialize_records([case()], [chunk()])

        self.assertEqual(ingest[0]["relevant_chunk_ids"], ["chunk-1"])
        self.assertEqual(ingest[0]["relevance_by_chunk"], {"chunk-1": 2})
        self.assertEqual(ragas[0]["reference_answer"], "Ela encontra documentos relevantes.")
        match = report["cases"][0]["evidence"][0]["matches"][0]
        self.assertEqual(match["coverage"], 1.0)
        self.assertEqual(match["match_mode"], "normalized_exact")

    def test_normalization_handles_case_punctuation_whitespace_and_hyphenation(self):
        master = case("RECUPERAÇÃO de infor-\nmação: encontra documentos relevantes!")
        exported = chunk(
            "Nesta seção, recuperação   de informação encontra documentos relevantes."
        )

        ingest, _ragas, report = materialize_records([master], [exported])

        self.assertEqual(ingest[0]["relevant_chunk_ids"], ["chunk-1"])
        self.assertEqual(
            report["cases"][0]["evidence"][0]["normalized_quote"],
            "recuperação de informação encontra documentos relevantes",
        )

    def test_normalization_equates_pdf_hyphen_spacing_artifacts(self):
        self.assertEqual(
            normalize_text("pós- alfabetização e ex- ministra"),
            normalize_text("pós-\nalfabetização e ex-\nministra"),
        )

    def test_normalization_ignores_attached_pdf_footnote_markers(self):
        self.assertEqual(
            normalize_text("Fazer a História é estar representado"),
            normalize_text("Fazer a História é estar representado9"),
        )

    def test_unmatched_evidence_fails_strictly(self):
        with self.assertRaises(UnmappedEvidenceError) as raised:
            materialize_records([case("trecho ausente no PDF")], [chunk()])

        report = raised.exception.report
        self.assertEqual(report["status"], "failed_unmapped_evidence")
        self.assertEqual(report["summary"]["unmapped_evidence_count"], 1)
        self.assertEqual(report["cases"][0]["evidence"][0]["status"], "unmapped")

    def test_materialize_files_writes_compatible_outputs_and_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cases_path = directory / "cases.jsonl"
            chunks_path = directory / "chunks.jsonl"
            ingest_path = directory / "out" / "ingest.jsonl"
            ragas_path = directory / "out" / "ragas.jsonl"
            report_path = directory / "out" / "report.json"
            cases_path.write_text(json.dumps(case(), ensure_ascii=False) + "\n", encoding="utf-8")
            chunks_path.write_text(json.dumps(chunk(), ensure_ascii=False) + "\n", encoding="utf-8")

            report = materialize_files(
                cases_path, chunks_path, ingest_path, ragas_path, report_path
            )

            ingest_row = json.loads(ingest_path.read_text(encoding="utf-8"))
            ragas_row = json.loads(ragas_path.read_text(encoding="utf-8"))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(ingest_row),
                {
                    "id",
                    "query",
                    "reference_answer",
                    "relevant_chunk_ids",
                    "relevance_by_chunk",
                },
            )
            self.assertEqual(
                set(ragas_row), {"id", "query", "reference_answer"}
            )
            self.assertEqual(saved_report["status"], "ok")
            self.assertEqual(saved_report["inputs"]["cases"]["sha256"], report["inputs"]["cases"]["sha256"])

    def test_cli_returns_nonzero_and_writes_only_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cases_path = directory / "cases.jsonl"
            chunks_path = directory / "chunks.jsonl"
            ingest_path = directory / "ingest.jsonl"
            ragas_path = directory / "ragas.jsonl"
            report_path = directory / "report.json"
            cases_path.write_text(
                json.dumps(case("evidência inexistente"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            chunks_path.write_text(
                json.dumps(chunk(), ensure_ascii=False) + "\n", encoding="utf-8"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "materialize.py"),
                    "materialize",
                    "--cases",
                    str(cases_path),
                    "--chunks",
                    str(chunks_path),
                    "--ingest-out",
                    str(ingest_path),
                    "--ragas-out",
                    str(ragas_path),
                    "--report-out",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 3)
            self.assertFalse(ingest_path.exists())
            self.assertFalse(ragas_path.exists())
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["status"],
                "failed_unmapped_evidence",
            )


if __name__ == "__main__":
    unittest.main()
