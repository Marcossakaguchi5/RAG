#!/usr/bin/env python3
"""Validate and materialize a canonical PDF ground truth.

This module deliberately depends only on the Python standard library.  It maps
verbatim evidence quotes to exported chunks and emits the two projections used
by the retrieval and RAG benchmarks in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "1.0"
NORMALIZATION_VERSION = "nfkc-casefold-alnum-fragmented-v2"
MAX_FRAGMENT_ORDINAL_GAP = 2
MAX_EVIDENCE_FRAGMENTS = 3


class GroundTruthError(Exception):
    """Base class for user-facing ground-truth errors."""


class ValidationError(GroundTruthError):
    """Raised when a JSONL input does not follow the documented schema."""


class UnmappedEvidenceError(GroundTruthError):
    """Raised when at least one evidence quote cannot be mapped strictly."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def normalize_text(text: str) -> str:
    """Return the conservative representation used for quote matching.

    Unicode compatibility forms and case are normalized, word hyphenation at a
    physical line break is repaired, punctuation becomes a separator, and runs
    of whitespace collapse.  Diacritics are intentionally retained: removing
    them can turn a different word into a false-positive match.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "", normalized)
    # PDF extractors disagree about whether a hyphen at a visual line break is
    # preserved, separated from the next word, or removed altogether.  Treat
    # the remaining intra-word forms equivalently so that a verbatim quote such
    # as ``pós- alfabetização`` can match extracted ``pós-\nalfabetização``.
    # Poppler can preserve a superscript footnote marker directly after the
    # preceding word (for example, ``representado9``).  It is not textual
    # evidence and should not prevent an otherwise verbatim quote from mapping.
    normalized = re.sub(r"(?<=[^\W\d_])\d+\b", "", normalized)
    normalized = normalized.casefold()
    normalized = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(normalized.split())


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load a non-empty JSONL file and report precise line errors."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read {source}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValidationError(
                f"{source}:{line_number}: each JSONL line must be an object"
            )
        records.append(value)

    if not records:
        raise ValidationError(f"{source}: the JSONL file has no records")
    return records


def _non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{location} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, location: str) -> str:
    digest = _non_empty_string(value, location).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValidationError(f"{location} must be a 64-character SHA-256 hex digest")
    return digest


def validate_cases(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and return the canonical fields of master cases."""

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case_index, raw_case in enumerate(records, start=1):
        location = f"case #{case_index}"
        case_id = _non_empty_string(raw_case.get("id"), f"{location}.id")
        if case_id in seen_ids:
            raise ValidationError(f"{location}.id duplicates {case_id!r}")
        seen_ids.add(case_id)

        query = _non_empty_string(raw_case.get("query"), f"{location}.query")
        reference_answer = _non_empty_string(
            raw_case.get("reference_answer"), f"{location}.reference_answer"
        )
        raw_evidence = raw_case.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValidationError(f"{location}.evidence must be a non-empty list")

        evidence: list[dict[str, Any]] = []
        for evidence_index, raw_item in enumerate(raw_evidence, start=1):
            item_location = f"{location}.evidence[{evidence_index}]"
            if not isinstance(raw_item, dict):
                raise ValidationError(f"{item_location} must be an object")

            selector_keys = [
                key for key in ("document_name", "document_sha256") if key in raw_item
            ]
            if not selector_keys:
                raise ValidationError(
                    f"{item_location} must contain document_name, document_sha256, or both"
                )
            selectors: dict[str, str] = {}
            for selector_key in selector_keys:
                if selector_key == "document_sha256":
                    selectors[selector_key] = _sha256(
                        raw_item[selector_key], f"{item_location}.{selector_key}"
                    )
                else:
                    selectors[selector_key] = _non_empty_string(
                        raw_item[selector_key], f"{item_location}.{selector_key}"
                    )

            quote = _non_empty_string(raw_item.get("quote"), f"{item_location}.quote")
            if not normalize_text(quote):
                raise ValidationError(
                    f"{item_location}.quote has no searchable alphanumeric text"
                )
            relevance = raw_item.get("relevance")
            if isinstance(relevance, bool) or not isinstance(relevance, int):
                raise ValidationError(f"{item_location}.relevance must be integer 1 or 2")
            if relevance not in (1, 2):
                raise ValidationError(f"{item_location}.relevance must be 1 or 2")

            canonical_evidence: dict[str, Any] = {
                **selectors,
                "quote": quote,
                "relevance": relevance,
            }
            if "page" in raw_item:
                page = raw_item["page"]
                if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                    raise ValidationError(f"{item_location}.page must be an integer >= 1")
                canonical_evidence["page"] = page
            evidence.append(canonical_evidence)

        canonical_case: dict[str, Any] = {
            "id": case_id,
            "query": query,
            "reference_answer": reference_answer,
            "evidence": evidence,
        }
        for field in ("split", "category"):
            if field in raw_case:
                canonical_case[field] = _non_empty_string(
                    raw_case[field], f"{location}.{field}"
                )
        if "provenance" in raw_case:
            if not isinstance(raw_case["provenance"], dict):
                raise ValidationError(f"{location}.provenance must be an object")
            canonical_case["provenance"] = raw_case["provenance"]
        cases.append(canonical_case)
    if not cases:
        raise ValidationError("cases must contain at least one record")
    return cases


def validate_chunks(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate chunks exported by the ingestion pipeline."""

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for chunk_index, raw_chunk in enumerate(records, start=1):
        location = f"chunk #{chunk_index}"
        chunk_id = _non_empty_string(raw_chunk.get("chunk_id"), f"{location}.chunk_id")
        if chunk_id in seen_ids:
            raise ValidationError(f"{location}.chunk_id duplicates {chunk_id!r}")
        seen_ids.add(chunk_id)

        document_name = _non_empty_string(
            raw_chunk.get("document_name"), f"{location}.document_name"
        )
        content = _non_empty_string(raw_chunk.get("content"), f"{location}.content")
        page_number = raw_chunk.get("page_number")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValidationError(f"{location}.page_number must be an integer >= 1")

        chunk = {
            "chunk_id": chunk_id,
            "document_name": document_name,
            "content": content,
            "page_number": page_number,
            "_input_order": chunk_index,
            "_ordinal": raw_chunk.get("ordinal", chunk_index - 1),
        }
        if (
            isinstance(chunk["_ordinal"], bool)
            or not isinstance(chunk["_ordinal"], int)
            or chunk["_ordinal"] < 0
        ):
            raise ValidationError(f"{location}.ordinal must be an integer >= 0")
        if "document_sha256" in raw_chunk:
            chunk["document_sha256"] = _sha256(
                raw_chunk["document_sha256"], f"{location}.document_sha256"
            )
        chunks.append(chunk)
    if not chunks:
        raise ValidationError("chunks must contain at least one record")
    return chunks


def _score_quote(quote: str, content: str) -> dict[str, Any]:
    normalized_quote = normalize_text(quote)
    normalized_content = normalize_text(content)
    quote_tokens = normalized_quote.split()
    content_tokens = normalized_content.split()
    match = SequenceMatcher(
        None, quote_tokens, content_tokens, autojunk=False
    ).find_longest_match(0, len(quote_tokens), 0, len(content_tokens))
    coverage = match.size / len(quote_tokens)
    compact_quote = normalized_quote.replace(" ", "")
    compact_content = normalized_content.replace(" ", "")
    if match.size < len(quote_tokens) and compact_quote and compact_quote in compact_content:
        return {
            "coverage": 1.0,
            "matched_quote_tokens": len(quote_tokens),
            "quote_token_count": len(quote_tokens),
            "quote_token_start": 0,
            "quote_token_end_exclusive": len(quote_tokens),
            "chunk_token_start": 0,
            "matched_text_normalized": normalized_quote,
            "match_mode": "normalized_exact_spacing_insensitive",
        }
    return {
        "coverage": coverage,
        "matched_quote_tokens": match.size,
        "quote_token_count": len(quote_tokens),
        "quote_token_start": match.a,
        "quote_token_end_exclusive": match.a + match.size,
        "chunk_token_start": match.b,
        "matched_text_normalized": " ".join(
            content_tokens[match.b : match.b + match.size]
        ),
        "match_mode": (
            "normalized_exact" if match.size == len(quote_tokens) else "normalized_partial"
        ),
    }


def _round_coverage(value: float) -> float:
    return round(value, 6)


def _public_match(chunk: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "document_name": chunk["document_name"],
        "page_number": chunk["page_number"],
        "coverage": _round_coverage(score["coverage"]),
        "matched_quote_tokens": score["matched_quote_tokens"],
        "quote_token_count": score["quote_token_count"],
        "quote_token_start": score["quote_token_start"],
        "quote_token_end_exclusive": score["quote_token_end_exclusive"],
        "chunk_token_start": score["chunk_token_start"],
        "matched_text_normalized": score["matched_text_normalized"],
        "match_mode": score["match_mode"],
    }


def _fragmented_exact_match(
    scored: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    """Return a minimal, page-local sequence that exactly covers one quote.

    Structural PDF chunkers may split a source paragraph across adjacent document
    blocks.  We accept that only when the matched token spans reconstruct the quote
    exactly, in source order, on the same page, with no missing token.
    """
    fragments = [
        (chunk, score)
        for chunk, score in scored
        if score["matched_quote_tokens"] > 0
    ]
    if not fragments:
        return None
    quote_token_count = fragments[0][1]["quote_token_count"]
    fragments.sort(
        key=lambda item: (
            item[1]["quote_token_start"],
            item[0]["_ordinal"],
            -item[1]["matched_quote_tokens"],
        )
    )

    def extend(
        path: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
        last_chunk, last_score = path[-1]
        if last_score["quote_token_end_exclusive"] == quote_token_count:
            return path
        if len(path) >= MAX_EVIDENCE_FRAGMENTS:
            return None
        for candidate_chunk, candidate_score in fragments:
            if candidate_score["quote_token_start"] != last_score["quote_token_end_exclusive"]:
                continue
            if candidate_chunk["page_number"] != last_chunk["page_number"]:
                continue
            ordinal_gap = candidate_chunk["_ordinal"] - last_chunk["_ordinal"]
            if not 0 < ordinal_gap <= MAX_FRAGMENT_ORDINAL_GAP:
                continue
            result = extend([*path, (candidate_chunk, candidate_score)])
            if result is not None:
                return result
        return None

    for fragment in fragments:
        if fragment[1]["quote_token_start"] != 0:
            continue
        result = extend([fragment])
        if result is not None:
            return result
    return None


def _public_fragmented_match(
    fragments: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    first_score = fragments[0][1]
    return {
        "chunk_ids": [chunk["chunk_id"] for chunk, _ in fragments],
        "document_name": fragments[0][0]["document_name"],
        "page_number": fragments[0][0]["page_number"],
        "coverage": 1.0,
        "matched_quote_tokens": first_score["quote_token_count"],
        "quote_token_count": first_score["quote_token_count"],
        "matched_text_normalized": " ".join(
            score["matched_text_normalized"] for _, score in fragments
        ),
        "match_mode": "normalized_exact_fragmented",
    }


def materialize_records(
    cases: Iterable[dict[str, Any]],
    chunks: Iterable[dict[str, Any]],
    *,
    min_coverage: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build benchmark projections and an auditable matching report.

    ``min_coverage=1.0`` is the academically conservative default: every token
    in the normalized quote must occur contiguously in a chunk.  Lower values
    are opt-in and retain the measured coverage in the report.
    """

    if (
        isinstance(min_coverage, bool)
        or not isinstance(min_coverage, (int, float))
        or not math.isfinite(float(min_coverage))
        or not 0 < float(min_coverage) <= 1
    ):
        raise ValidationError("min_coverage must be a number in the interval (0, 1]")
    threshold = float(min_coverage)
    canonical_cases = validate_cases(cases)
    canonical_chunks = validate_chunks(chunks)

    ingest_rows: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []
    report_cases: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    total_evidence = 0

    for case in canonical_cases:
        relevance_by_chunk: dict[str, int] = {}
        evidence_reports: list[dict[str, Any]] = []

        for evidence_index, evidence in enumerate(case["evidence"], start=1):
            total_evidence += 1
            selector = {
                key: evidence[key]
                for key in ("document_name", "document_sha256")
                if key in evidence
            }
            candidates = [
                chunk
                for chunk in canonical_chunks
                if all(chunk.get(key) == value for key, value in selector.items())
            ]
            scored: list[tuple[dict[str, Any], dict[str, Any]]] = [
                (chunk, _score_quote(evidence["quote"], chunk["content"]))
                for chunk in candidates
            ]
            matches = [
                (chunk, score)
                for chunk, score in scored
                if score["coverage"] + 1e-12 >= threshold
            ]
            matches.sort(key=lambda item: item[0]["_input_order"])
            fragmented_match = None
            if not matches:
                fragmented_match = _fragmented_exact_match(scored)
            best = sorted(
                scored,
                key=lambda item: (-item[1]["coverage"], item[0]["_input_order"]),
            )[:5]

            for chunk, _score in matches:
                previous = relevance_by_chunk.get(chunk["chunk_id"], 0)
                relevance_by_chunk[chunk["chunk_id"]] = max(
                    previous, evidence["relevance"]
                )
            if fragmented_match is not None:
                for chunk, _score in fragmented_match:
                    previous = relevance_by_chunk.get(chunk["chunk_id"], 0)
                    relevance_by_chunk[chunk["chunk_id"]] = max(
                        previous, evidence["relevance"]
                    )

            evidence_report = {
                "evidence_index": evidence_index,
                "selector": selector,
                "page": evidence.get("page"),
                "quote": evidence["quote"],
                "normalized_quote": normalize_text(evidence["quote"]),
                "relevance": evidence["relevance"],
                "candidate_chunk_count": len(candidates),
                "matched_chunk_ids": (
                    [chunk["chunk_id"] for chunk, _ in matches]
                    if fragmented_match is None
                    else [chunk["chunk_id"] for chunk, _ in fragmented_match]
                ),
                "best_coverage": _round_coverage(
                    best[0][1]["coverage"] if best else 0.0
                ),
                "matches": (
                    [_public_match(chunk, score) for chunk, score in matches]
                    if fragmented_match is None
                    else [_public_fragmented_match(fragmented_match)]
                ),
                "best_candidates": [
                    _public_match(chunk, score) for chunk, score in best
                ],
                "status": "mapped" if matches or fragmented_match is not None else "unmapped",
            }
            evidence_reports.append(evidence_report)
            if not matches and fragmented_match is None:
                unmatched.append(
                    {
                        "case_id": case["id"],
                        "evidence_index": evidence_index,
                        "selector": selector,
                        "best_coverage": evidence_report["best_coverage"],
                    }
                )

        relevant_chunk_ids = list(relevance_by_chunk)
        shared_fields = {
            field: case[field]
            for field in ("split", "category", "provenance")
            if field in case
        }
        ingest_rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "reference_answer": case["reference_answer"],
                "relevant_chunk_ids": relevant_chunk_ids,
                "relevance_by_chunk": relevance_by_chunk,
                **shared_fields,
            }
        )
        ragas_rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "reference_answer": case["reference_answer"],
                **shared_fields,
            }
        )
        report_cases.append(
            {
                "id": case["id"],
                "query": case["query"],
                "relevant_chunk_ids": relevant_chunk_ids,
                "evidence": evidence_reports,
                **shared_fields,
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed_unmapped_evidence" if unmatched else "ok",
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode": "NFKC",
            "casefold": True,
            "repair_line_break_hyphenation": True,
            "punctuation_policy": "replace with spaces",
            "diacritics_policy": "retain",
            "coverage": "contiguous normalized token sequence, or exact page-local fragment reconstruction / quote tokens",
            "fragmented_evidence": {
                "enabled": True,
                "rule": "up to 3 ordered, page-local chunks may exactly cover a quote",
                "max_ordinal_gap": MAX_FRAGMENT_ORDINAL_GAP,
            },
        },
        "parameters": {"min_coverage": threshold},
        "summary": {
            "case_count": len(canonical_cases),
            "chunk_count": len(canonical_chunks),
            "evidence_count": total_evidence,
            "mapped_evidence_count": total_evidence - len(unmatched),
            "unmapped_evidence_count": len(unmatched),
        },
        "unmapped_evidence": unmatched,
        "cases": report_cases,
    }

    if unmatched:
        locations = ", ".join(
            f"{item['case_id']}[evidence {item['evidence_index']}]"
            for item in unmatched
        )
        raise UnmappedEvidenceError(
            f"{len(unmatched)} evidence quote(s) did not reach coverage "
            f"{threshold:g}: {locations}",
            report,
        )
    return ingest_rows, ragas_rows, report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _atomic_write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        raise GroundTruthError(f"cannot write {path}: {exc}") from exc


def _report_text(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def materialize_files(
    cases_path: str | os.PathLike[str],
    chunks_path: str | os.PathLike[str],
    ingest_out: str | os.PathLike[str],
    ragas_out: str | os.PathLike[str],
    report_out: str | os.PathLike[str],
    *,
    min_coverage: float = 1.0,
) -> dict[str, Any]:
    """Materialize files atomically; on mismatch, write only the audit report."""

    cases_source = Path(cases_path)
    chunks_source = Path(chunks_path)
    ingest_target = Path(ingest_out)
    ragas_target = Path(ragas_out)
    report_target = Path(report_out)
    targets = [
        target.resolve(strict=False)
        for target in (ingest_target, ragas_target, report_target)
    ]
    if len(set(targets)) != len(targets):
        raise ValidationError("ingest, RAGAS, and report output paths must be distinct")

    case_records = load_jsonl(cases_source)
    chunk_records = load_jsonl(chunks_source)
    input_metadata = {
        "cases": {
            "path": str(cases_source),
            "sha256": _file_sha256(cases_source),
            "record_count": len(case_records),
        },
        "chunks": {
            "path": str(chunks_source),
            "sha256": _file_sha256(chunks_source),
            "record_count": len(chunk_records),
        },
    }
    output_metadata = {
        "ingest": str(ingest_target),
        "ragas": str(ragas_target),
        "report": str(report_target),
    }

    try:
        ingest_rows, ragas_rows, report = materialize_records(
            case_records, chunk_records, min_coverage=min_coverage
        )
    except UnmappedEvidenceError as exc:
        exc.report["inputs"] = input_metadata
        exc.report["outputs"] = output_metadata
        _atomic_write(report_target, _report_text(exc.report))
        raise

    report["inputs"] = input_metadata
    report["outputs"] = output_metadata
    _atomic_write(ingest_target, _jsonl_text(ingest_rows))
    _atomic_write(ragas_target, _jsonl_text(ragas_rows))
    _atomic_write(report_target, _report_text(report))
    return report


def _coverage_argument(value: str) -> float:
    try:
        coverage = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]") from exc
    if not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]")
    return coverage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and materialize canonical PDF ground truth."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate master cases and, optionally, exported chunks"
    )
    validate_parser.add_argument("--cases", required=True, type=Path)
    validate_parser.add_argument("--chunks", type=Path)

    materialize_parser = subparsers.add_parser(
        "materialize", help="map evidence quotes and write benchmark projections"
    )
    materialize_parser.add_argument("--cases", required=True, type=Path)
    materialize_parser.add_argument("--chunks", required=True, type=Path)
    materialize_parser.add_argument("--ingest-out", required=True, type=Path)
    materialize_parser.add_argument("--ragas-out", required=True, type=Path)
    materialize_parser.add_argument("--report-out", required=True, type=Path)
    materialize_parser.add_argument(
        "--min-coverage",
        type=_coverage_argument,
        default=1.0,
        help="required contiguous normalized token coverage (default: 1.0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            cases = validate_cases(load_jsonl(args.cases))
            summary: dict[str, Any] = {
                "status": "ok",
                "case_count": len(cases),
                "evidence_count": sum(len(case["evidence"]) for case in cases),
            }
            if args.chunks:
                chunks = validate_chunks(load_jsonl(args.chunks))
                summary["chunk_count"] = len(chunks)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return 0

        report = materialize_files(
            args.cases,
            args.chunks,
            args.ingest_out,
            args.ragas_out,
            args.report_out,
            min_coverage=args.min_coverage,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
        return 0
    except UnmappedEvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except GroundTruthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
