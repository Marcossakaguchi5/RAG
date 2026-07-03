from __future__ import annotations

import argparse
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_LANGUAGE,
    configure_benchmark_environment,
    qdrant_connection_hint,
    read_jsonl,
)


def point_id_for_doc(doc_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"sciq:{doc_id}"))


def recreate_collection(collection_name: str) -> None:
    from app.services.vector_store import ensure_collection, get_vector_client

    client = get_vector_client()
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name=collection_name)
        ensure_collection(collection_name)
    except Exception as error:
        from app.core.config import get_settings

        raise RuntimeError(qdrant_connection_hint(get_settings().qdrant_url)) from error


def iter_batches(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def ingest(corpus_path: Path, collection_name: str, batch_size: int, recreate: bool) -> int:
    from qdrant_client import models

    from app.services.embeddings import get_embedding_service
    from app.services.sparse_embeddings import get_sparse_embedding_service
    from app.services.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection, upsert_chunks

    if recreate:
        recreate_collection(collection_name)
    else:
        try:
            ensure_collection(collection_name)
        except Exception as error:
            from app.core.config import get_settings

            raise RuntimeError(qdrant_connection_hint(get_settings().qdrant_url)) from error

    rows = list(read_jsonl(corpus_path))
    embedding_service = get_embedding_service()
    sparse_service = get_sparse_embedding_service()

    indexed = 0
    for batch in iter_batches(rows, batch_size):
        texts = [str(row["text"]) for row in batch]
        dense_vectors = embedding_service.encode(texts)
        sparse_vectors = sparse_service.encode_documents(texts)
        points = []
        for row, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors, strict=True):
            doc_id = str(row["doc_id"])
            text = str(row["text"])
            points.append(
                models.PointStruct(
                    id=point_id_for_doc(doc_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload={
                        "chunk_id": doc_id,
                        "collection_name": collection_name,
                        "document_id": doc_id,
                        "document_name": doc_id,
                        "page_number": 1,
                        "ordinal": 0,
                        "content": text,
                        "dataset": "sciq",
                        "source_field": "support",
                    },
                )
            )
        upsert_chunks(points, collection_name)
        indexed += len(points)
        print(f"Indexados {indexed}/{len(rows)} supports...")

    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa o corpus SciQ no Qdrant.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--collection", default="sciq_baseline")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--sparse-language", default=DEFAULT_SPARSE_LANGUAGE)
    parser.add_argument("--fastembed-cache-dir", type=Path, default=DEFAULT_FASTEMBED_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    configure_benchmark_environment(args.qdrant_url, args.sparse_language, args.fastembed_cache_dir)
    corpus_path = args.data_dir / "processed" / "corpus.jsonl"
    indexed = ingest(corpus_path, args.collection, args.batch_size, args.recreate)
    print(f"Ingest finalizado: {indexed} documentos na coleção {args.collection}.")


if __name__ == "__main__":
    main()
