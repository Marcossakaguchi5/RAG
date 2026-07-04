from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_CASES_PATH = BENCHMARK_DIR / "ground_truth.example.jsonl"
def read_cases(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error_:
                raise ValueError(f"Linha {line_number} de {path} nao e JSON valido.") from error_
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(base_url: str, path: str, payload: dict[str, Any], token: str = "", timeout: float = 180) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as error_:
        detail = error_.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error_.code} em {path}: {detail}") from error_
    except error.URLError as error_:
        raise RuntimeError(f"Nao foi possivel conectar em {base_url}: {error_.reason}") from error_


def authenticate(base_url: str, password: str, timeout: float) -> str:
    body = post_json(base_url, "/api/auth/login", {"password": password}, timeout=timeout)
    return str(body["access_token"])


def default_run_dir(data_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return data_dir / "runs" / timestamp


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "nao", "não", "no", "off"}
    return bool(value)


def case_payload(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    query = str(case.get("query") or case.get("question") or "").strip()
    reference_answer = str(case.get("reference_answer") or case.get("expected_answer") or "").strip()
    if not query:
        raise ValueError("Caso sem query/question.")
    if not reference_answer:
        raise ValueError(f"Caso {case.get('id') or query[:40]} sem reference_answer/expected_answer.")
    return {
        "query": query,
        "collection_name": str(case.get("collection_name") or args.collection),
        "method": str(case.get("method") or args.method),
        "top_k": int(case.get("top_k") or args.top_k),
        "candidate_k": int(case.get("candidate_k") or args.candidate_k),
        "use_reranker": bool_value(case.get("use_reranker", args.use_reranker)),
        "evaluate_ragas": bool_value(case.get("evaluate_site_ragas", args.site_ragas)),
        "reference_answer": reference_answer,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "base_url": args.base_url,
        "collection_default": args.collection,
        "method_default": args.method,
        "top_k_default": args.top_k,
        "candidate_k_default": args.candidate_k,
        "use_reranker_default": args.use_reranker,
        "site_ragas_enabled": args.site_ragas,
        "total_cases": len(rows),
        "ok_cases": sum(1 for row in rows if row.get("status") == "ok"),
        "error_cases": sum(1 for row in rows if row.get("status") == "error"),
        "total_sources": sum(int(row.get("sources_count") or 0) for row in rows),
    }


def write_responses_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "status",
        "collection_name",
        "method",
        "top_k",
        "candidate_k",
        "used_reranker",
        "latency_ms",
        "sources_count",
        "query",
        "reference_answer",
        "answer",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "status": row.get("status"),
                    "collection_name": row.get("collection_name"),
                    "method": row.get("method"),
                    "top_k": row.get("top_k"),
                    "candidate_k": row.get("candidate_k"),
                    "used_reranker": row.get("used_reranker"),
                    "latency_ms": row.get("latency_ms"),
                    "sources_count": row.get("sources_count"),
                    "query": row.get("query"),
                    "reference_answer": row.get("reference_answer"),
                    "answer": row.get("answer"),
                    "error": row.get("error"),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    password = args.password or os.getenv("CHAT_APP_PASSWORD")
    if not password:
        raise SystemExit("Informe a senha com --password ou CHAT_APP_PASSWORD.")

    cases = read_cases(args.cases)
    run_dir = args.run_dir or default_run_dir(args.data_dir)
    token = authenticate(args.base_url, password, args.timeout)
    results = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or f"case-{index:04d}")
        try:
            payload = case_payload(case, args)
            response = post_json(args.base_url, "/api/rag", payload, token=token, timeout=args.timeout)
            sources = response.get("sources", []) or []
            results.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "query": payload["query"],
                    "reference_answer": payload["reference_answer"],
                    "collection_name": response.get("collection_name"),
                    "method": response.get("method"),
                    "top_k": response.get("top_k"),
                    "candidate_k": response.get("candidate_k"),
                    "used_reranker": response.get("used_reranker"),
                    "latency_ms": response.get("latency_ms"),
                    "answer": response.get("answer"),
                    "sources_count": len(sources),
                    "ragas": response.get("ragas", {}),
                    "sources": sources,
                }
            )
            print(f"[ok] {case_id}: {len(sources)} fonte(s)")
        except Exception as error_:
            results.append({"case_id": case_id, "status": "error", "error": str(error_)})
            print(f"[erro] {case_id}: {error_}")

    append_jsonl(run_dir / "results.jsonl", results)
    write_responses_csv(run_dir / "responses.csv", results)
    summary = summarize(results, args, run_dir)
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta respostas do Chat RAG para avaliacao oficial RAGAS.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8011")
    parser.add_argument("--password", default="")
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--site-ragas",
        action="store_true",
        help="Tambem salva o relatorio RAGAS oficial retornado pelo site. Para artigo, use evaluate_official.py em lote.",
    )
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    parser.set_defaults(use_reranker=True)
    summary = run(parser.parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
