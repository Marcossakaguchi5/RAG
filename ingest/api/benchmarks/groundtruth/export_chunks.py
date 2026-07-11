from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "chunks.jsonl"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_hashes(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        name = path.name
        if name in hashes:
            raise ValueError(f"Mais de um PDF informado com o nome {name!r}.")
        hashes[name] = sha256_file(path)
    return hashes


def point_to_row(point: Any, hashes_by_name: dict[str, str]) -> dict[str, Any]:
    payload = point.payload or {}
    chunk_id = str(payload.get("chunk_id") or point.id).strip()
    document_name = str(
        payload.get("document_name") or payload.get("file_name") or ""
    ).strip()
    content = str(payload.get("content") or "").strip()
    if not chunk_id or not document_name or not content:
        raise ValueError(
            f"Ponto {point.id} sem chunk_id, document_name/file_name ou content."
        )

    page_number = int(payload.get("page_number") or payload.get("page_start") or 1)
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "document_id": str(payload.get("document_id") or ""),
        "document_name": document_name,
        "page_number": max(1, page_number),
        "ordinal": int(payload.get("ordinal") or payload.get("chunk_ordinal") or 0),
        "chunking_strategy": str(payload.get("chunking_strategy") or ""),
        "content": content,
    }
    if document_name in hashes_by_name:
        row["document_sha256"] = hashes_by_name[document_name]
    return row


def export_chunks(
    collection_name: str,
    batch_size: int,
    hashes_by_name: dict[str, str],
) -> list[dict[str, Any]]:
    from app.services.vector_store import get_vector_client

    client = get_vector_client()
    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection inexistente no Qdrant: {collection_name}")

    rows: list[dict[str, Any]] = []
    offset: Any | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        rows.extend(point_to_row(point, hashes_by_name) for point in points)
        if next_offset is None:
            break
        offset = next_offset

    rows.sort(
        key=lambda row: (
            row["document_name"],
            row["ordinal"],
            row["chunk_id"],
        )
    )
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta todos os chunks de uma collection para materializar o ground truth."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6335")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--pdf",
        action="append",
        type=Path,
        default=[],
        help="PDF original; pode ser repetido para adicionar document_sha256 aos chunks.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size deve ser positivo.")

    os.environ["QDRANT_URL"] = args.qdrant_url
    try:
        hashes_by_name = pdf_hashes(args.pdf)
        rows = export_chunks(args.collection, args.batch_size, hashes_by_name)
        write_jsonl_atomic(args.output, rows)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(
        json.dumps(
            {
                "collection": args.collection,
                "qdrant_url": args.qdrant_url,
                "output": str(args.output),
                "chunks": len(rows),
                "pdf_hashes": hashes_by_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
