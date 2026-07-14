from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_QDRANT_URL = "http://localhost:6335"
DEFAULT_SPARSE_LANGUAGE = "english"
DEFAULT_FASTEMBED_CACHE_DIR = DEFAULT_DATA_DIR / "model_cache" / "fastembed"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def configure_benchmark_environment(qdrant_url: str, sparse_language: str, fastembed_cache_dir: Path) -> None:
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["SPARSE_LANGUAGE"] = sparse_language
    os.environ["FASTEMBED_CACHE_DIR"] = str(fastembed_cache_dir)


def qdrant_connection_hint(qdrant_url: str) -> str:
    return (
        f"Não foi possível conectar ao Qdrant em {qdrant_url}. "
        "De fora do container, use `--qdrant-url http://localhost:6335`; "
        "dentro da API, use `--qdrant-url http://qdrant:6333`."
    )


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Linha JSONL inválida em {path}:{line_number}") from error


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value < 1 for value in values):
        raise ValueError("Informe valores de k positivos, por exemplo: 1,3,5,10")
    return values
