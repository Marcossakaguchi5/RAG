from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_LANGUAGE,
    configure_benchmark_environment,
)
from benchmarks.sciq.ingest_corpus import (  # The shared chunker/indexer keeps SciQ and QASPER comparable.
    SUPPORTED_TEXT_CHUNKING_STRATEGIES,
    chunk_id_for_doc,
    ingest as _ingest,
    point_id_for_chunk as _point_id_for_chunk,
)


def point_id_for_chunk(doc_id: str, ordinal: int) -> str:
    return _point_id_for_chunk(doc_id, ordinal, "qasper")


def ingest(
    corpus_path: Path,
    collection_name: str,
    batch_size: int,
    recreate: bool,
    chunking_strategy: str = "recursive_text",
) -> int:
    return _ingest(
        corpus_path=corpus_path,
        collection_name=collection_name,
        batch_size=batch_size,
        recreate=recreate,
        chunking_strategy=chunking_strategy,
        dataset_name="qasper",
        source_field="full_text.paragraphs",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa os parágrafos do QASPER no Qdrant.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--collection", default="qasper_text_baseline")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--sparse-language", default=DEFAULT_SPARSE_LANGUAGE)
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunking-strategy", choices=sorted(SUPPORTED_TEXT_CHUNKING_STRATEGIES), default="recursive_text")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    indexed = ingest(
        args.data_dir / "processed" / "corpus.jsonl",
        args.collection,
        args.batch_size,
        args.recreate,
        args.chunking_strategy,
    )
    print(f"Ingest QASPER finalizado: {indexed} chunks na coleção {args.collection}.")


if __name__ == "__main__":
    main()
