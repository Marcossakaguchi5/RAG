from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import DEFAULT_DATA_DIR
from ingest_corpus import SUPPORTED_TEXT_CHUNKING_STRATEGIES
from benchmarks.sciq.export_groundtruth import build_groundtruth

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "groundtruth" / "qasper_ground_truth.jsonl"
DEFAULT_CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "180"))
DEFAULT_CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "700"))
DEFAULT_CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "100"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta QASPER para o formato ground truth do ingest.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--collection", default="qasper_text_baseline")
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
    print(f"Ground truth exportado: {summary['cases']} casos em {summary['output_path']}.")


if __name__ == "__main__":
    main()
