from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from common import DEFAULT_DATA_DIR, normalize_text, write_jsonl


def make_doc_id(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"sciq_doc_{digest}"


def load_sciq(revision: str | None = None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Dependência ausente: instale as dependências do benchmark com "
            "`pip install -r benchmarks/sciq/requirements-benchmark.txt`."
        ) from error
    kwargs = {"revision": revision} if revision else {}
    return load_dataset("allenai/sciq", **kwargs)


def prepare(
    data_dir: Path,
    min_support_words: int,
    seed: int,
    dataset_revision: str | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    dataset = load_sciq(dataset_revision)

    docs: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        for index, example in enumerate(dataset[split]):
            support = normalize_text(example["support"])
            if len(support.split()) < min_support_words:
                continue

            doc_id = make_doc_id(support)
            docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "text": support,
                    "source": "sciq",
                    "source_field": "support",
                },
            )

            options = [
                normalize_text(example["correct_answer"]),
                normalize_text(example["distractor1"]),
                normalize_text(example["distractor2"]),
                normalize_text(example["distractor3"]),
            ]
            random.shuffle(options)

            query_id = f"sciq_{split}_{index:05d}"
            queries.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "question": normalize_text(example["question"]),
                    "correct_answer": normalize_text(example["correct_answer"]),
                    "options": options,
                    "relevant_doc_id": doc_id,
                }
            )
            qrels.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "doc_id": doc_id,
                    "relevance": 1,
                }
            )

    processed_dir = data_dir / "processed"
    doc_count = write_jsonl(processed_dir / "corpus.jsonl", docs.values())
    query_count = write_jsonl(processed_dir / "queries.jsonl", queries)
    qrel_count = write_jsonl(processed_dir / "qrels.jsonl", qrels)

    return {
        "dataset": "allenai/sciq",
        "dataset_revision_requested": dataset_revision or "main",
        "dataset_version": str(dataset["train"].info.version),
        "dataset_fingerprints": {
            split: str(getattr(dataset[split], "_fingerprint", ""))
            for split in ("train", "validation", "test")
        },
        "min_support_words": min_support_words,
        "seed": seed,
        "documents": doc_count,
        "queries": query_count,
        "qrels": qrel_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara corpus, queries e qrels do SciQ.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-support-words", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()

    counts = prepare(
        args.data_dir,
        args.min_support_words,
        args.seed,
        args.dataset_revision,
    )
    print(
        "SciQ preparado: "
        f"{counts['documents']} documentos, {counts['queries']} queries, {counts['qrels']} qrels."
    )


if __name__ == "__main__":
    main()
