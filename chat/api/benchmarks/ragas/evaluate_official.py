from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "answer_relevancy",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Linha {line_number} de {path} nao e JSON valido.") from error
            if isinstance(data, dict):
                rows.append(data)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_results_path(data_dir: Path) -> Path:
    runs_dir = data_dir / "runs"
    candidates = sorted(runs_dir.glob("*/results.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"Nenhum results.jsonl encontrado em {runs_dir}. Rode run_groundtruth.py primeiro.")
    return candidates[0]


def default_eval_dir(results_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return results_path.parent / "ragas-official" / timestamp


def env_value(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback).strip()


def require_env(name: str, value: str) -> str:
    if not value:
        raise SystemExit(f"Configure {name} no ambiente ou passe o argumento correspondente.")
    return value


def source_contexts(row: dict[str, Any], max_context_chars: int) -> list[str]:
    contexts = []
    for source in row.get("sources", []) or []:
        content = str(source.get("content") or "").strip()
        if content:
            contexts.append(content[:max_context_chars])
    return contexts


def source_ids(row: dict[str, Any]) -> list[str]:
    ids = []
    for source in row.get("sources", []) or []:
        chunk_id = str(source.get("chunk_id") or "").strip()
        if chunk_id:
            ids.append(chunk_id)
    return ids


def prepare_sample(row: dict[str, Any], max_context_chars: int) -> dict[str, Any]:
    query = str(row.get("query") or "").strip()
    answer = str(row.get("answer") or "").strip()
    reference = str(row.get("reference_answer") or "").strip()
    contexts = source_contexts(row, max_context_chars)
    if not query:
        raise ValueError("Caso sem query.")
    if not answer:
        raise ValueError("Caso sem answer.")
    if not reference:
        raise ValueError("Caso sem reference_answer.")
    if not contexts:
        raise ValueError("Caso sem contexts recuperados.")
    return {
        "user_input": query,
        "response": answer,
        "reference": reference,
        "retrieved_contexts": contexts,
    }


def score_value(result: Any) -> tuple[float | None, str | None]:
    value = getattr(result, "value", result)
    reason = getattr(result, "reason", None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and math.isnan(numeric):
        numeric = None
    return numeric, str(reason) if reason else None


def build_scorers(args: argparse.Namespace) -> dict[str, Callable[[dict[str, Any]], Any]]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except ImportError as error:
        raise SystemExit(
            "Dependencias oficiais do RAGAS nao instaladas. Rode: "
            "pip install -r benchmarks/ragas/requirements-ragas.txt"
        ) from error

    api_key = require_env("RAGAS_LLM_API_KEY ou LLM_API_KEY", args.llm_api_key)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.llm_base_url,
        default_headers={
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "RAG Chat Benchmark",
        },
    )
    llm = llm_factory(args.llm_model, client=client)
    embeddings = None

    if "answer_relevancy" in args.metrics:
        embeddings = HuggingFaceEmbeddings(
            model=args.embedding_model,
            device=args.embedding_device or None,
            normalize_embeddings=True,
        )

    scorers = {}
    if "faithfulness" in args.metrics:
        metric = Faithfulness(llm=llm)
        scorers["faithfulness"] = lambda sample, scorer=metric: scorer.score(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    if "context_precision" in args.metrics:
        metric = ContextPrecision(llm=llm)
        scorers["context_precision"] = lambda sample, scorer=metric: scorer.score(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    if "context_recall" in args.metrics:
        metric = ContextRecall(llm=llm)
        scorers["context_recall"] = lambda sample, scorer=metric: scorer.score(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    if "factual_correctness" in args.metrics:
        metric = FactualCorrectness(llm=llm)
        scorers["factual_correctness"] = lambda sample, scorer=metric: scorer.score(
            response=sample["response"],
            reference=sample["reference"],
        )
    if "answer_relevancy" in args.metrics:
        metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
        scorers["answer_relevancy"] = lambda sample, scorer=metric: scorer.score(
            user_input=sample["user_input"],
            response=sample["response"],
        )
    return scorers


def write_metrics_csv(path: Path, rows: list[dict[str, Any]], metric_names: list[str]) -> None:
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
        *metric_names,
        "query",
        "answer",
        "reference_answer",
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
                    **row.get("metrics", {}),
                    "query": row.get("query"),
                    "answer": row.get("answer"),
                    "reference_answer": row.get("reference_answer"),
                    "error": row.get("error"),
                }
            )


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, results_path: Path, output_dir: Path) -> dict[str, Any]:
    averages = {}
    for name in args.metrics:
        values = [row["metrics"][name] for row in rows if row.get("status") == "ok" and row["metrics"].get(name) is not None]
        averages[name] = round(mean(values), 4) if values else None
    return {
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model if "answer_relevancy" in args.metrics else None,
        "metrics": args.metrics,
        "total_cases": len(rows),
        "ok_cases": sum(1 for row in rows if row.get("status") == "ok"),
        "error_cases": sum(1 for row in rows if row.get("status") == "error"),
        "metric_averages": averages,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    results_path = args.results or latest_results_path(args.data_dir)
    output_dir = args.output_dir or default_eval_dir(results_path)
    rows = read_jsonl(results_path)
    scorers = build_scorers(args)
    evaluated = []

    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or f"case-{index:04d}")
        try:
            if row.get("status") != "ok":
                raise ValueError(f"Caso nao esta ok no coletor: {row.get('error') or row.get('status')}")
            sample = prepare_sample(row, args.max_context_chars)
            metrics = {}
            reasons = {}
            for name, scorer in scorers.items():
                value, reason = score_value(scorer(sample))
                metrics[name] = value
                if reason:
                    reasons[name] = reason
            evaluated.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "query": row.get("query"),
                    "answer": row.get("answer"),
                    "reference_answer": row.get("reference_answer"),
                    "collection_name": row.get("collection_name"),
                    "method": row.get("method"),
                    "top_k": row.get("top_k"),
                    "candidate_k": row.get("candidate_k"),
                    "used_reranker": row.get("used_reranker"),
                    "latency_ms": row.get("latency_ms"),
                    "sources_count": row.get("sources_count"),
                    "source_ids": source_ids(row),
                    "metrics": metrics,
                    "reasons": reasons,
                }
            )
            print(f"[ok] {case_id}: {metrics}")
        except Exception as error:
            if not args.continue_on_error:
                raise
            evaluated.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "query": row.get("query"),
                    "answer": row.get("answer"),
                    "reference_answer": row.get("reference_answer"),
                    "collection_name": row.get("collection_name"),
                    "method": row.get("method"),
                    "top_k": row.get("top_k"),
                    "candidate_k": row.get("candidate_k"),
                    "used_reranker": row.get("used_reranker"),
                    "latency_ms": row.get("latency_ms"),
                    "sources_count": row.get("sources_count"),
                    "metrics": {},
                    "error": str(error),
                }
            )
            print(f"[erro] {case_id}: {error}")

    write_jsonl(output_dir / "official_ragas_results.jsonl", evaluated)
    write_metrics_csv(output_dir / "official_ragas_metrics.csv", evaluated, args.metrics)
    summary = summarize(evaluated, args, results_path, output_dir)
    write_json(output_dir / "official_ragas_summary.json", summary)
    return summary


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Metricas desconhecidas: {', '.join(unknown)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula metricas oficiais RAGAS para resultados do Chat RAG.")
    parser.add_argument("--results", type=Path, default=None, help="Caminho para results.jsonl gerado por run_groundtruth.py.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metrics", type=parse_metrics, default=DEFAULT_METRICS)
    parser.add_argument("--llm-base-url", default=env_value("RAGAS_LLM_BASE_URL", env_value("LLM_BASE_URL", "https://openrouter.ai/api/v1")))
    parser.add_argument("--llm-api-key", default=env_value("RAGAS_LLM_API_KEY", env_value("LLM_API_KEY")))
    parser.add_argument("--llm-model", default=env_value("RAGAS_LLM_MODEL", env_value("RAGAS_MODEL", env_value("LLM_MODEL", "deepseek/deepseek-v4-flash"))))
    parser.add_argument("--embedding-model", default=env_value("RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--embedding-device", default=env_value("RAGAS_EMBEDDING_DEVICE"))
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--continue-on-error", action="store_true")
    summary = run(parser.parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
