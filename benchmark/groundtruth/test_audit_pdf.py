import hashlib
import unittest

from groundtruth.audit_pdf import audit_cases


HASH = hashlib.sha256(b"pdf").hexdigest()


def case(page: int = 2) -> dict:
    return {
        "id": "q1",
        "query": "Pergunta?",
        "reference_answer": "Resposta.",
        "evidence": [
            {
                "document_name": "material.pdf",
                "document_sha256": HASH,
                "page": page,
                "quote": "Evidência exata.",
                "relevance": 2,
            }
        ],
    }


def document(pages: dict[int, str]) -> dict:
    return {
        "path": "/tmp/material.pdf",
        "document_name": "material.pdf",
        "document_sha256": HASH,
        "extractor": "test",
        "pages": pages,
    }


class PdfAuditTests(unittest.TestCase):
    def test_accepts_quote_on_declared_page(self):
        report = audit_cases([case()], [document({1: "Outro texto", 2: "Evidência exata!"})])

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["matched_evidence_count"], 1)

    def test_reports_quote_found_only_on_another_page(self):
        report = audit_cases([case(page=1)], [document({1: "Outro texto", 2: "Evidência exata."})])

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failures"][0]["status"], "wrong_page")
        self.assertEqual(report["failures"][0]["matched_pages"], [2])

    def test_warns_when_quote_occurs_on_multiple_pages(self):
        report = audit_cases(
            [case(page=2)],
            [document({1: "Evidência exata.", 2: "Evidência exata."})],
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["multiple_match_warnings"], 1)


if __name__ == "__main__":
    unittest.main()
