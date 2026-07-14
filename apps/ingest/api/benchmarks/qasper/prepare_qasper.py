from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import DEFAULT_DATA_DIR, normalize_text, write_jsonl

DATASET_NAME = "allenai/qasper"
# Commit that contains the Parquet conversion published by the official dataset
# repository.  ``datasets>=4`` no longer executes the legacy qasper.py script.
DEFAULT_DATASET_REVISION = "0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46"
FLOAT_PREFIX = "FLOAT SELECTED"


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def make_doc_id(paper_id: str, section_index: int, paragraph_index: int) -> str:
    return f"qasper_{stable_id(paper_id)}_s{section_index:02d}_p{paragraph_index:03d}"


def load_qasper(revision: str | None = None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Dependência ausente: instale `pip install -r benchmarks/qasper/requirements-benchmark.txt`."
        ) from error
    selected_revision = revision or DEFAULT_DATASET_REVISION
    # Explicit Parquet paths bypass the legacy dataset script.  The latter is
    # intentionally unsupported by datasets 4+ and would otherwise make the
    # benchmark fail before preparation.
    data_files = {
        split: f"hf://datasets/{DATASET_NAME}@{selected_revision}/qasper/{split}-*.parquet"
        for split in ("train", "validation", "test")
    }
    return load_dataset("parquet", data_files=data_files)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def qas_for_paper(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both the native QASPER nested representation and HF columnar structs."""
    qas = example.get("qas", [])
    if isinstance(qas, list):
        return [item for item in qas if isinstance(item, dict)]
    if not isinstance(qas, dict):
        return []

    questions = as_list(qas.get("question", qas.get("questions", [])))
    question_ids = as_list(qas.get("question_id", qas.get("id", [])))
    answers = as_list(qas.get("answers", []))
    rows = []
    for index, question in enumerate(questions):
        rows.append(
            {
                "question": question,
                "question_id": question_ids[index] if index < len(question_ids) else f"q{index}",
                "answers": answers[index] if index < len(answers) else [],
            }
        )
    return rows


def answer_items(answer_group: Any) -> list[dict[str, Any]]:
    """Flatten QASPER's answer/annotation nesting without assuming a loader version."""
    if isinstance(answer_group, dict):
        # In the current Parquet schema each annotation is a struct with an
        # ``answer`` struct.  Older JSON loaders exposed a list here instead.
        if isinstance(answer_group.get("answer"), dict):
            return [answer_group["answer"]]
        if isinstance(answer_group.get("answer"), list):
            return [item for item in answer_group["answer"] if isinstance(item, dict)]
        if isinstance(answer_group.get("answers"), list):
            return [item for item in answer_group["answers"] if isinstance(item, dict)]
        return [answer_group]
    items: list[dict[str, Any]] = []
    for group in as_list(answer_group):
        items.extend(answer_items(group))
    return items


def answer_to_reference(answer: dict[str, Any]) -> tuple[str, str] | None:
    free_form = normalize_text(str(answer.get("free_form_answer") or ""))
    if free_form:
        return "free_form", free_form
    spans = [normalize_text(str(span)) for span in as_list(answer.get("extractive_spans"))]
    spans = [span for span in spans if span]
    if spans:
        return "extractive", " ".join(spans)
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes_no", "yes" if yes_no else "no"
    return None


def contains_float(evidence: Iterable[Any]) -> bool:
    return any(normalize_text(str(item)).startswith(FLOAT_PREFIX) for item in evidence)


def prepare_from_dataset(
    dataset: Any,
    data_dir: Path,
    *,
    text_evidence_only: bool = True,
    answerable_only: bool = True,
) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    skipped = defaultdict(int)

    for split, papers in dataset.items():
        for paper_index, paper in enumerate(papers):
            paper_id = str(paper.get("id") or paper.get("paper_id") or f"{split}-{paper_index}")
            full_text = paper.get("full_text") or {}
            # The original JSON represents this as parallel lists, whereas the
            # current Parquet dataset exposes it as a list of section structs.
            if isinstance(full_text, dict):
                sections = as_list(full_text.get("section_name"))
                paragraph_groups = as_list(full_text.get("paragraphs"))
            elif isinstance(full_text, list):
                sections = [item.get("section_name", "") for item in full_text if isinstance(item, dict)]
                paragraph_groups = [item.get("paragraphs", []) for item in full_text if isinstance(item, dict)]
            else:
                sections = []
                paragraph_groups = []
            evidence_to_docs: dict[str, set[str]] = defaultdict(set)

            for section_index, raw_paragraphs in enumerate(paragraph_groups):
                section = normalize_text(str(sections[section_index])) if section_index < len(sections) else ""
                for paragraph_index, raw_text in enumerate(as_list(raw_paragraphs)):
                    text = normalize_text(str(raw_text))
                    if not text or text.startswith(FLOAT_PREFIX):
                        continue
                    doc_id = make_doc_id(paper_id, section_index, paragraph_index)
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "paper_id": paper_id,
                        "section": section,
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                        "text": text,
                        "source": "qasper",
                        "source_field": "full_text.paragraphs",
                    }
                    evidence_to_docs[text].add(doc_id)

            for question_index, qa in enumerate(qas_for_paper(paper)):
                question = normalize_text(str(qa.get("question") or ""))
                if not question:
                    skipped["empty_question"] += 1
                    continue
                annotations = answer_items(qa.get("answers", qa.get("answer", [])))
                if not annotations:
                    skipped["no_answers"] += 1
                    continue
                unanswerable = [bool(answer.get("unanswerable", False)) for answer in annotations]
                if answerable_only and any(unanswerable):
                    skipped["unanswerable_or_disputed"] += 1
                    continue
                usable_answers = [answer for answer in annotations if not bool(answer.get("unanswerable", False))]
                evidence = [item for answer in usable_answers for item in as_list(answer.get("evidence"))]
                if text_evidence_only and contains_float(evidence):
                    skipped["float_evidence"] += 1
                    continue
                relevant_doc_ids = sorted(
                    {
                        doc_id
                        for item in evidence
                        for doc_id in evidence_to_docs.get(normalize_text(str(item)), set())
                    }
                )
                if not relevant_doc_ids:
                    skipped["no_mapped_text_evidence"] += 1
                    continue
                references = [answer_to_reference(answer) for answer in usable_answers]
                references = [reference for reference in references if reference]
                if not references:
                    skipped["no_reference_answer"] += 1
                    continue
                query_token = str(qa.get("question_id") or qa.get("id") or question_index)
                query_id = f"qasper_{stable_id(f'{paper_id}:{query_token}:{question}') }"
                unique_references = list(dict.fromkeys(text for _, text in references))
                answer_types = sorted({kind for kind, _ in references})
                queries.append(
                    {
                        "query_id": query_id,
                        "split": split,
                        "paper_id": paper_id,
                        "question": question,
                        "reference_answer": unique_references[0],
                        "reference_answers": unique_references,
                        "answer_types": answer_types,
                        "evidence_type": "text",
                        "relevant_doc_ids": relevant_doc_ids,
                    }
                )
                qrels.extend(
                    {"query_id": query_id, "split": split, "doc_id": doc_id, "relevance": 2}
                    for doc_id in relevant_doc_ids
                )

    processed_dir = data_dir / "processed"
    return {
        "dataset": DATASET_NAME,
        "documents": write_jsonl(processed_dir / "corpus.jsonl", docs.values()),
        "queries": write_jsonl(processed_dir / "queries.jsonl", queries),
        "qrels": write_jsonl(processed_dir / "qrels.jsonl", qrels),
        "filters": {"answerable_only": answerable_only, "text_evidence_only": text_evidence_only},
        "skipped": dict(sorted(skipped.items())),
        "splits": sorted(dataset.keys()),
    }


def prepare(
    data_dir: Path,
    dataset_revision: str | None = None,
    *,
    text_evidence_only: bool = True,
    answerable_only: bool = True,
) -> dict[str, Any]:
    dataset = load_qasper(dataset_revision)
    summary = prepare_from_dataset(
        dataset,
        data_dir,
        text_evidence_only=text_evidence_only,
        answerable_only=answerable_only,
    )
    summary.update(
        {
            "dataset_revision_requested": dataset_revision or DEFAULT_DATASET_REVISION,
            "dataset_version": str(getattr(getattr(dataset.get("train"), "info", None), "version", "")),
            "dataset_fingerprints": {split: str(getattr(data, "_fingerprint", "")) for split, data in dataset.items()},
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara corpus, queries e qrels textuais do QASPER.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--include-unanswerable", dest="answerable_only", action="store_false")
    parser.add_argument("--include-float-evidence", dest="text_evidence_only", action="store_false")
    parser.set_defaults(answerable_only=True, text_evidence_only=True)
    args = parser.parse_args()
    summary = prepare(
        args.data_dir,
        args.dataset_revision,
        text_evidence_only=args.text_evidence_only,
        answerable_only=args.answerable_only,
    )
    print(f"QASPER preparado: {summary['documents']} parágrafos, {summary['queries']} queries, {summary['qrels']} qrels.")


if __name__ == "__main__":
    main()
