#!/usr/bin/env python3
"""Audit canonical evidence quotes directly against their source PDF pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    from .materialize import GroundTruthError, load_jsonl, normalize_text, validate_cases
except ImportError:  # Direct execution: python experiments/groundtruth/audit_pdf.py
    from materialize import GroundTruthError, load_jsonl, normalize_text, validate_cases


class PdfAuditError(GroundTruthError):
    """Raised when PDFs cannot be resolved or their text cannot be extracted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_with_pypdf(path: Path) -> dict[int, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfAuditError("pypdf nao esta instalado") from exc
    reader = PdfReader(str(path))
    return {
        page_number: page.extract_text() or ""
        for page_number, page in enumerate(reader.pages, start=1)
    }


def extract_with_pdftotext(path: Path) -> dict[int, str]:
    try:
        completed = subprocess.run(
            ["pdftotext", "-raw", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PdfAuditError(
            "pdftotext nao encontrado; instale Poppler ou use um ambiente com pypdf"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise PdfAuditError(f"falha ao extrair {path}: {detail}") from exc
    pages = completed.stdout.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages:
        raise PdfAuditError(f"nenhuma pagina textual extraida de {path}")
    return {page_number: text for page_number, text in enumerate(pages, start=1)}


def extract_pdf(path: Path, extractor: str) -> tuple[dict[int, str], str]:
    if extractor == "pypdf":
        return extract_with_pypdf(path), "pypdf"
    if extractor == "pdftotext":
        return extract_with_pdftotext(path), "pdftotext"
    try:
        return extract_with_pypdf(path), "pypdf"
    except PdfAuditError:
        return extract_with_pdftotext(path), "pdftotext"


def load_documents(pdf_paths: Sequence[Path], extractor: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for input_path in pdf_paths:
        path = input_path.resolve()
        if not path.is_file():
            raise PdfAuditError(f"PDF inexistente: {path}")
        pages, used_extractor = extract_pdf(path, extractor)
        documents.append(
            {
                "path": str(path),
                "document_name": path.name,
                "document_sha256": sha256_file(path),
                "extractor": used_extractor,
                "pages": pages,
            }
        )
    if not documents:
        raise PdfAuditError("informe ao menos um --pdf")
    return documents


def _candidate_documents(
    evidence: dict[str, Any], documents: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        document
        for document in documents
        if all(
            document.get(selector) == evidence[selector]
            for selector in ("document_name", "document_sha256")
            if selector in evidence
        )
    ]


def audit_cases(
    cases: Sequence[dict[str, Any]], documents: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    case_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    evidence_count = 0

    normalized_documents: dict[str, dict[int, str]] = {
        str(document["path"]): {
            int(page): normalize_text(text)
            for page, text in document["pages"].items()
        }
        for document in documents
    }

    for case in cases:
        evidence_reports: list[dict[str, Any]] = []
        for evidence_index, evidence in enumerate(case["evidence"], start=1):
            evidence_count += 1
            candidates = _candidate_documents(evidence, documents)
            base = {
                "evidence_index": evidence_index,
                "declared_page": evidence.get("page"),
                "quote": evidence["quote"],
            }
            if len(candidates) != 1:
                status = "document_not_found" if not candidates else "ambiguous_document"
                item = {**base, "status": status, "candidate_documents": len(candidates)}
                evidence_reports.append(item)
                failures.append({"case_id": case["id"], **item})
                continue

            document = candidates[0]
            normalized_quote = normalize_text(evidence["quote"])
            normalized_pages = normalized_documents[str(document["path"])]
            matched_pages = [
                page
                for page, text in normalized_pages.items()
                if normalized_quote in text
            ]
            declared_page = evidence.get("page")
            if declared_page is None:
                valid = bool(matched_pages)
            else:
                valid = declared_page in matched_pages
            status = "ok" if valid else ("wrong_page" if matched_pages else "quote_not_found")
            occurrence_count = sum(
                normalized_pages[page].count(normalized_quote) for page in matched_pages
            )
            item = {
                **base,
                "status": status,
                "document_name": document["document_name"],
                "document_sha256": document["document_sha256"],
                "matched_pages": matched_pages,
                "occurrences": occurrence_count,
            }
            evidence_reports.append(item)
            if not valid:
                failures.append({"case_id": case["id"], **item})
            elif len(matched_pages) > 1 or occurrence_count > 1:
                warning = {"case_id": case["id"], **item, "warning": "multiple_matches"}
                warnings.append(warning)

        case_reports.append(
            {
                "id": case["id"],
                "query": case["query"],
                "status": (
                    "ok"
                    if all(item["status"] == "ok" for item in evidence_reports)
                    else "failed"
                ),
                "evidence": evidence_reports,
            }
        )

    return {
        "schema_version": "1.0",
        "status": "ok" if not failures else "failed",
        "summary": {
            "case_count": len(cases),
            "evidence_count": evidence_count,
            "matched_evidence_count": evidence_count - len(failures),
            "failed_evidence_count": len(failures),
            "multiple_match_warnings": len(warnings),
        },
        "documents": [
            {
                key: document[key]
                for key in (
                    "path",
                    "document_name",
                    "document_sha256",
                    "extractor",
                )
            }
            | {"page_count": len(document["pages"])}
            for document in documents
        ],
        "failures": failures,
        "warnings": warnings,
        "cases": case_reports,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confere hash, pagina e citacao dos casos diretamente nos PDFs."
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--extractor",
        choices=["auto", "pypdf", "pdftotext"],
        default="auto",
    )
    args = parser.parse_args(argv)
    try:
        cases = validate_cases(load_jsonl(args.cases))
        documents = load_documents(args.pdf, args.extractor)
        report = audit_cases(cases, documents)
        if args.report:
            write_json_atomic(args.report, report)
        print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
        return 0 if report["status"] == "ok" else 3
    except (OSError, GroundTruthError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
