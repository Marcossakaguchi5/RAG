from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import error, request


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_CASES_PATH = BENCHMARK_DIR / "ground_truth.example.jsonl"
METHODS = {"bm25", "dense", "hybrid"}
METRIC_FIELDS = ["precision_at_k", "recall_at_k", "map", "ndcg_at_k", "mrr"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(base_url: str, path: str, payload: dict[str, Any], token: str = "", timeout: float = 120) -> dict[str, Any]:
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


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(methods) - METHODS)
    if invalid:
        raise ValueError(f"Metodo(s) invalido(s): {', '.join(invalid)}")
    return methods or ["hybrid"]


def relevant_ids_for_case(case: dict[str, Any]) -> list[str]:
    raw = case.get("relevant_chunk_ids", case.get("relevant_ids", []))
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def case_payload(case: dict[str, Any], args: argparse.Namespace, method: str) -> dict[str, Any]:
    query = str(case.get("query") or case.get("question") or "").strip()
    relevant_ids = relevant_ids_for_case(case)
    if not query:
        raise ValueError("Caso sem query/question.")
    if not relevant_ids:
        raise ValueError(f"Caso {case.get('id') or query[:40]} sem relevant_chunk_ids/relevant_ids.")
    return {
        "query": query,
        "collection_name": str(case.get("collection_name") or args.collection),
        "method": method,
        "top_k": int(case.get("top_k") or args.top_k),
        "relevant_chunk_ids": relevant_ids,
    }


def methods_for_case(case: dict[str, Any], default_methods: list[str]) -> list[str]:
    raw = case.get("method")
    if not raw:
        return default_methods
    return parse_methods(str(raw))


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    method_summaries = {}
    for method in sorted({str(row.get("method")) for row in rows if row.get("method")}):
        method_rows = [row for row in rows if row.get("method") == method and row.get("status") == "ok"]
        averages = {}
        for field in METRIC_FIELDS:
            values = [
                row["metrics"].get(field)
                for row in method_rows
                if row.get("metrics", {}).get(field) is not None
            ]
            averages[field] = round(mean(values), 4) if values else None
        method_summaries[method] = {"ok_cases": len(method_rows), "metric_averages": averages}

    return {
        "run_dir": str(run_dir),
        "base_url": args.base_url,
        "collection_default": args.collection,
        "methods_default": parse_methods(args.methods),
        "top_k_default": args.top_k,
        "total_rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "error_rows": sum(1 for row in rows if row.get("status") == "error"),
        "methods": method_summaries,
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "status",
        "collection_name",
        "method",
        "top_k",
        *METRIC_FIELDS,
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "query",
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
                    **row.get("metrics", {}),
                    "retrieved_chunk_ids": ",".join(row.get("retrieved_chunk_ids", [])),
                    "relevant_chunk_ids": ",".join(row.get("relevant_chunk_ids", [])),
                    "query": row.get("query"),
                    "error": row.get("error"),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    password = args.password or os.getenv("INGEST_APP_PASSWORD")
    if not password:
        raise SystemExit("Informe a senha com --password ou INGEST_APP_PASSWORD.")

    cases = read_jsonl(args.cases)
    default_methods = parse_methods(args.methods)
    run_dir = args.run_dir or default_run_dir(args.data_dir)
    token = authenticate(args.base_url, password, args.timeout)
    results = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or f"case-{index:04d}")
        for method in methods_for_case(case, default_methods):
            try:
                payload = case_payload(case, args, method)
                response = post_json(args.base_url, "/api/search", payload, token=token, timeout=args.timeout)
                retrieved_ids = [str(item.get("chunk_id", "")) for item in response.get("results", [])]
                metrics = response.get("metrics", {})
                results.append(
                    {
                        "case_id": case_id,
                        "status": "ok",
                        "query": payload["query"],
                        "collection_name": payload["collection_name"],
                        "method": response.get("method", method),
                        "top_k": response.get("top_k", payload["top_k"]),
                        "relevant_chunk_ids": payload["relevant_chunk_ids"],
                        "retrieved_chunk_ids": retrieved_ids,
                        "metrics": {field: metrics.get(field) for field in METRIC_FIELDS},
                        "results": response.get("results", []),
                    }
                )
                print(f"[ok] {case_id}/{method}: {results[-1]['metrics']}")
            except Exception as error_:
                results.append({"case_id": case_id, "status": "error", "method": method, "error": str(error_), "metrics": {}})
                print(f"[erro] {case_id}/{method}: {error_}")

    write_jsonl(run_dir / "results.jsonl", results)
    write_metrics_csv(run_dir / "metrics.csv", results)
    summary = summarize(results, args, run_dir)
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa benchmark manual de recuperacao com ground truth.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8010")
    parser.add_argument("--password", default="")
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--methods", default="bm25,dense,hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    summary = run(parser.parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
