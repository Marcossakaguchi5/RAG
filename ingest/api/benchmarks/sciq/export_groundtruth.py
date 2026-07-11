from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_DATA_DIR, read_jsonl, write_jsonl
from ingest_corpus import SUPPORTED_TEXT_CHUNKING_STRATEGIES, chunk_id_for_doc


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "groundtruth" / "sciq_ground_truth.jsonl"
DEFAULT_CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "180"))
DEFAULT_CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "700"))
DEFAULT_CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "100"))


def load_corpus(corpus_path: Path) -> dict[str, str]:
    return {str(row["doc_id"]): str(row["text"]) for row in read_jsonl(corpus_path)}


def load_qrels(qrels_path: Path, split: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(qrels_path):
        if str(row.get("split")) != split:
            continue
        qrels[str(row["query_id"])].add(str(row["doc_id"]))
    return dict(qrels)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def paragraph_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [normalize_text(block) for block in re.split(r"\n\s*\n+", normalized)]
    if len(blocks) <= 1:
        blocks = [normalize_text(block) for block in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])", normalized)]
    return [block for block in blocks if block]


def split_large_block(block: str, max_words: int) -> list[str]:
    words = block.split()
    if len(words) <= max_words:
        return [block]
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def recursive_split_block(
    block: str,
    max_words: int,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " "),
) -> list[str]:
    block = block.strip()
    if not block or word_count(block) <= max_words:
        return [block] if block else []
    if not separators:
        return split_large_block(block, max_words)

    separator = separators[0]
    parts = block.split(separator)
    if len(parts) <= 1:
        return recursive_split_block(block, max_words, separators[1:])

    chunks: list[str] = []
    current = ""
    joiner = separator if separator != " " else " "
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current}{joiner}{part}" if current else part
        if word_count(candidate) <= max_words:
            current = candidate
            continue
        if current:
            chunks.extend(recursive_split_block(current, max_words, separators[1:]))
        current = part
    if current:
        chunks.extend(recursive_split_block(current, max_words, separators[1:]))
    return chunks


def overlap_tail(text: str, overlap_words: int) -> str:
    if overlap_words <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else " ".join(words)


def recursive_text_chunk_count(text: str, chunk_min_words: int, chunk_size_words: int, chunk_overlap_words: int) -> int:
    max_words = max(chunk_size_words, 1)
    min_words = min(max(chunk_min_words, 1), max_words)
    overlap_words = min(max(chunk_overlap_words, 0), max_words - 1)
    blocks: list[str] = []
    for block in paragraph_blocks(text):
        blocks.extend(recursive_split_block(block, max_words))

    estimated_total_chunks = max(1, math.ceil(sum(word_count(block) for block in blocks) / max(1, max_words - overlap_words)))
    chunks = 0
    current: list[str] = []
    current_words = 0
    current_is_overlap_only = False

    def flush() -> None:
        nonlocal chunks, current, current_words, current_is_overlap_only
        if not current:
            return
        content = normalize_text("\n\n".join(current))
        if not content:
            current = []
            current_words = 0
            current_is_overlap_only = False
            return
        chunks += 1
        overlap = overlap_tail(content, overlap_words)
        current = [overlap] if overlap else []
        current_words = word_count(overlap)
        current_is_overlap_only = bool(overlap)

    for block in blocks:
        block_words = word_count(block)
        if current and current_words >= min_words and current_words + block_words > max_words:
            flush()
        current.append(block)
        current_words += block_words
        current_is_overlap_only = False
        if current_words >= max_words:
            flush()

    if current and not current_is_overlap_only:
        flush()

    if not chunks and estimated_total_chunks:
        raise ValueError("Documento sem texto para chunking.")
    return chunks


def chunk_ids_for_doc(
    doc_id: str,
    text: str,
    chunking_strategy: str,
    chunk_min_words: int,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[str]:
    if chunking_strategy == "recursive_text":
        count = recursive_text_chunk_count(text, chunk_min_words, chunk_size_words, chunk_overlap_words)
        return [chunk_id_for_doc(doc_id, ordinal) for ordinal in range(count)]

    from app.services.pdf_processor import chunk_text

    drafts = chunk_text(text, page_count=1, chunking_strategy=chunking_strategy)
    return [chunk_id_for_doc(doc_id, draft.ordinal) for draft in drafts]


def build_groundtruth(
    data_dir: Path,
    output_path: Path,
    collection_name: str,
    split: str,
    chunking_strategy: str,
    chunk_min_words: int,
    chunk_size_words: int,
    chunk_overlap_words: int,
    limit_queries: int | None,
    top_k: int | None,
) -> dict[str, Any]:
    processed_dir = data_dir / "processed"
    corpus = load_corpus(processed_dir / "corpus.jsonl")
    qrels = load_qrels(processed_dir / "qrels.jsonl", split)
    chunk_ids_by_doc: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for query in read_jsonl(processed_dir / "queries.jsonl"):
        if str(query.get("split")) != split:
            continue
        query_id = str(query["query_id"])
        relevant_doc_ids = sorted(qrels.get(query_id, set()))
        if not relevant_doc_ids:
            continue

        relevant_chunk_ids: list[str] = []
        for doc_id in relevant_doc_ids:
            if doc_id not in corpus:
                raise ValueError(f"Qrel aponta para doc_id ausente no corpus: {doc_id}")
            if doc_id not in chunk_ids_by_doc:
                chunk_ids_by_doc[doc_id] = chunk_ids_for_doc(
                    doc_id,
                    corpus[doc_id],
                    chunking_strategy,
                    chunk_min_words,
                    chunk_size_words,
                    chunk_overlap_words,
                )
            relevant_chunk_ids.extend(chunk_ids_by_doc[doc_id])

        case = {
            "id": query_id,
            "collection_name": collection_name,
            "query": str(query["question"]),
            "relevant_chunk_ids": relevant_chunk_ids,
            "relevant_doc_ids": relevant_doc_ids,
            "split": split,
            "correct_answer": query.get("correct_answer"),
            "options": query.get("options", []),
        }
        if top_k is not None:
            case["top_k"] = top_k
        rows.append(case)

        if limit_queries is not None and len(rows) >= limit_queries:
            break

    written = write_jsonl(output_path, rows)
    return {
        "output_path": str(output_path),
        "collection": collection_name,
        "split": split,
        "chunking_strategy": chunking_strategy,
        "chunk_min_words": chunk_min_words,
        "chunk_size_words": chunk_size_words,
        "chunk_overlap_words": chunk_overlap_words,
        "cases": written,
        "relevant_documents": len({doc_id for row in rows for doc_id in row["relevant_doc_ids"]}),
        "relevant_chunks": sum(len(row["relevant_chunk_ids"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta queries/qrels SciQ para o formato do benchmark ground truth do ingest."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--collection", default="sciq_baseline")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--chunking-strategy", choices=sorted(SUPPORTED_TEXT_CHUNKING_STRATEGIES), default="recursive_text")
    parser.add_argument("--chunk-min-words", type=int, default=DEFAULT_CHUNK_MIN_WORDS)
    parser.add_argument("--chunk-size-words", type=int, default=DEFAULT_CHUNK_SIZE_WORDS)
    parser.add_argument("--chunk-overlap-words", type=int, default=DEFAULT_CHUNK_OVERLAP_WORDS)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    summary = build_groundtruth(
        data_dir=args.data_dir,
        output_path=args.output,
        collection_name=args.collection,
        split=args.split,
        chunking_strategy=args.chunking_strategy,
        chunk_min_words=args.chunk_min_words,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        limit_queries=args.limit_queries,
        top_k=args.top_k,
    )
    print(
        "Ground truth exportado: "
        f"{summary['cases']} casos, {summary['relevant_chunks']} chunks relevantes em {summary['output_path']}."
    )


if __name__ == "__main__":
    main()
