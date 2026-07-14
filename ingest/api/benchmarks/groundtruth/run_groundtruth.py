from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any
from urllib import error, request


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_CASES_PATH = BENCHMARK_DIR / "ground_truth.example.jsonl"
METHODS = {"bm25", "dense", "hybrid"}
METRIC_FIELDS = [
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "map",
    "ndcg_at_k",
    "mrr",
]


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


def relevance_grades_for_case(case: dict[str, Any]) -> dict[str, float]:
    raw = case.get("relevance_by_chunk")
    if raw is None:
        return {chunk_id: 1.0 for chunk_id in relevant_ids_for_case(case)}
    if not isinstance(raw, dict):
        raise ValueError("relevance_by_chunk deve ser um objeto chunk_id -> grau.")

    grades: dict[str, float] = {}
    for raw_chunk_id, raw_grade in raw.items():
        chunk_id = str(raw_chunk_id).strip()
        if not chunk_id or isinstance(raw_grade, bool):
            raise ValueError("relevance_by_chunk contem chunk ou grau invalido.")
        try:
            grade = float(raw_grade)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Grau invalido para {chunk_id!r}: {raw_grade!r}") from error
        if not math.isfinite(grade) or grade <= 0:
            raise ValueError(f"Grau de relevancia deve ser positivo para {chunk_id!r}.")
        grades[chunk_id] = grade

    relevant_ids = set(relevant_ids_for_case(case))
    if relevant_ids and relevant_ids != set(grades):
        raise ValueError(
            "relevant_chunk_ids e relevance_by_chunk devem conter os mesmos IDs."
        )
    if not grades:
        raise ValueError("relevance_by_chunk nao pode ser vazio.")
    return grades


def calculate_ranked_metrics(
    result_ids: list[str],
    relevance_by_chunk: dict[str, float],
    top_k: int,
) -> dict[str, float]:
    ranked = result_ids[:top_k]
    relevant_ids = set(relevance_by_chunk)
    hits = [chunk_id in relevant_ids for chunk_id in ranked]
    hit_count = sum(hits)

    accumulated_precision = 0.0
    reciprocal_rank = 0.0
    seen_relevant = 0
    dcg = 0.0
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant_ids:
            seen_relevant += 1
            accumulated_precision += seen_relevant / position
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1 / position
        grade = relevance_by_chunk.get(chunk_id, 0.0)
        dcg += (2**grade - 1) / math.log2(position + 1)

    ideal_grades = sorted(relevance_by_chunk.values(), reverse=True)[:top_k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(position + 1)
        for position, grade in enumerate(ideal_grades, start=1)
    )
    return {
        "hit_rate_at_k": round(1.0 if hit_count else 0.0, 6),
        "precision_at_k": round(hit_count / top_k, 6),
        "recall_at_k": round(hit_count / len(relevant_ids), 6),
        "map": round(accumulated_precision / len(relevant_ids), 6),
        "ndcg_at_k": round(dcg / ideal_dcg if ideal_dcg else 0.0, 6),
        "mrr": round(reciprocal_rank, 6),
    }


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
        latencies = sorted(
            float(row["latency_ms"])
            for row in method_rows
            if row.get("latency_ms") is not None
        )
        p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0
        method_summaries[method] = {
            "ok_cases": len(method_rows),
            "metric_averages": averages,
            "latency_ms": {
                "median": round(median(latencies), 3) if latencies else None,
                "p95": round(latencies[p95_index], 3) if latencies else None,
            },
        }

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
        "split",
        "category",
        "latency_ms",
        *METRIC_FIELDS,
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "relevance_by_chunk",
        "query",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "status": row.get("status"),
                    "collection_name": row.get("collection_name"),
                    "method": row.get("method"),
                    "top_k": row.get("top_k"),
                    "split": row.get("split"),
                    "category": row.get("category"),
                    "latency_ms": row.get("latency_ms"),
                    **row.get("metrics", {}),
                    "retrieved_chunk_ids": ",".join(row.get("retrieved_chunk_ids", [])),
                    "relevant_chunk_ids": ",".join(row.get("relevant_chunk_ids", [])),
                    "relevance_by_chunk": json.dumps(
                        row.get("relevance_by_chunk", {}), ensure_ascii=False
                    ),
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
                relevance_by_chunk = relevance_grades_for_case(case)
                request_started_at = perf_counter()
                response = post_json(args.base_url, "/api/search", payload, token=token, timeout=args.timeout)
                latency_ms = round((perf_counter() - request_started_at) * 1000, 3)
                retrieved_ids = [str(item.get("chunk_id", "")) for item in response.get("results", [])]
                metrics = calculate_ranked_metrics(
                    retrieved_ids,
                    relevance_by_chunk,
                    int(response.get("top_k", payload["top_k"])),
                )
                results.append(
                    {
                        "case_id": case_id,
                        "status": "ok",
                        "query": payload["query"],
                        "collection_name": payload["collection_name"],
                        "method": response.get("method", method),
                        "top_k": response.get("top_k", payload["top_k"]),
                        "split": case.get("split"),
                        "category": case.get("category"),
                        "latency_ms": latency_ms,
                        "relevant_chunk_ids": payload["relevant_chunk_ids"],
                        "relevance_by_chunk": relevance_by_chunk,
                        "retrieved_chunk_ids": retrieved_ids,
                        "metrics": metrics,
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Mantem codigo de saida zero mesmo com casos incompletos; nao recomendado para a rodada final.",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["error_rows"] and not args.continue_on_error:
        raise SystemExit(
            f"Rodada contem {summary['error_rows']} erro(s); artefatos foram salvos para diagnostico."
        )


if __name__ == "__main__":
    main()
