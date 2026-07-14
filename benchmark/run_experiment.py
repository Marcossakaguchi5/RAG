#!/usr/bin/env python3
"""Single entry point for the academic SciQ, PDF-IR, and optional RAG runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence
from urllib import error, parse, request


ROOT = Path(__file__).resolve().parents[1]
INGEST_API = ROOT / "apps" / "ingest" / "api"
CHAT_API = ROOT / "apps" / "chat" / "api"
DEFAULT_RUNS = ROOT / "benchmark" / "runs" / "novas"
CHUNKING_STRATEGIES = {
    "fixed_token",
    "recursive_text",
    "docling_hierarchical",
    "docling_hybrid",
    "docling_hybrid_parent_child",
    "docling_hybrid_contextual",
}
RETRIEVAL_METHODS = {"bm25", "dense", "hybrid"}
APPROVED_REVIEW_STATUSES = {"approved", "adjudicated"}
DEFAULT_SCIQ_DATASET_REVISION = "2c94ad3e1aafab77146f384e23536f97a4849815"
PACKAGE_NAMES = [
    "datasets",
    "docling",
    "fastembed",
    "numpy",
    "qdrant-client",
    "ragas",
    "sentence-transformers",
    "torch",
]


class ExperimentError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def parse_csv(raw: str, allowed: set[str], label: str) -> list[str]:
    values = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ExperimentError(f"{label} invalido(s): {', '.join(unknown)}")
    if not values:
        raise ExperimentError(f"Informe ao menos um {label}.")
    return values


def safe_collection_name(*parts: str) -> str:
    raw = "_".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_-") or "experiment"
    if len(normalized) <= 64:
        return normalized
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:55]}_{suffix}"


def ensure_new_run_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ExperimentError(f"Diretorio de rodada nao esta vazio: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_review_summary(path: Path) -> dict[str, int]:
    statuses: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"JSON invalido em {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ExperimentError(f"Caso em {path}:{line_number} nao e um objeto JSON.")
            provenance = row.get("provenance")
            status = (
                str(provenance.get("review_status") or "missing").strip().lower()
                if isinstance(provenance, dict)
                else "missing"
            )
            statuses[status or "missing"] += 1
    if not statuses:
        raise ExperimentError(f"Arquivo de casos vazio: {path}")
    return dict(sorted(statuses.items()))


def require_reviewed_cases(path: Path, *, allow_draft: bool) -> dict[str, int]:
    summary = case_review_summary(path)
    unapproved = {
        status: count
        for status, count in summary.items()
        if status not in APPROVED_REVIEW_STATUSES
    }
    if unapproved and not allow_draft:
        detail = ", ".join(f"{status}={count}" for status, count in unapproved.items())
        raise ExperimentError(
            f"Casos ainda nao aprovados ({detail}). Revise-os ou use "
            "--allow-draft-cases somente para uma rodada piloto."
        )
    return summary


def hash_files(base: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    hashes: dict[str, str] = {}
    if not base.exists():
        return hashes
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded or "__pycache__" in path.parts:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def git_metadata() -> dict[str, Any]:
    def capture(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip() if completed.returncode == 0 else ""

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD") or None,
        "branch": capture("branch", "--show-current") or None,
        "dirty": bool(status),
        "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
    }


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def base_manifest(study: str, run_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": study,
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "parameters": parameters,
        "git": git_metadata(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"[exec] cwd={cwd} {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def _json_body(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Any | None = None,
    token: str = "",
    timeout: float = 180,
) -> Any:
    headers: dict[str, str] = {}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ExperimentError(f"HTTP {exc.code} em {path}: {detail}") from exc
    except error.URLError as exc:
        raise ExperimentError(f"Falha ao acessar {base_url}: {exc.reason}") from exc


def authenticate(base_url: str, password: str, timeout: float) -> str:
    response = _json_body(
        base_url,
        "/api/auth/login",
        method="POST",
        payload={"password": password},
        timeout=timeout,
    )
    return str(response["access_token"])


def _multipart_upload(
    base_url: str,
    token: str,
    collection: str,
    chunking_strategy: str,
    pdf_paths: list[Path],
    timeout: float,
) -> dict[str, Any]:
    boundary = f"----rag-experiment-{uuid.uuid4().hex}"
    body = bytearray()

    def add(value: bytes) -> None:
        body.extend(value)
        body.extend(b"\r\n")

    for name, value in (
        ("collection_name", collection),
        ("chunking_strategy", chunking_strategy),
    ):
        add(f"--{boundary}".encode())
        add(f'Content-Disposition: form-data; name="{name}"'.encode())
        add(b"")
        add(value.encode("utf-8"))

    for pdf_path in pdf_paths:
        filename = pdf_path.name.replace('"', "")
        add(f"--{boundary}".encode())
        add(
            (
                'Content-Disposition: form-data; name="files"; '
                f'filename="{filename}"'
            ).encode("utf-8")
        )
        add(b"Content-Type: application/pdf")
        add(b"")
        add(pdf_path.read_bytes())
    body.extend(f"--{boundary}--\r\n".encode())

    req = request.Request(
        f"{base_url.rstrip('/')}/api/documents/upload",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ExperimentError(f"HTTP {exc.code} ao ingerir PDFs: {detail}") from exc
    except error.URLError as exc:
        raise ExperimentError(f"Falha ao acessar {base_url}: {exc.reason}") from exc


def prepare_pdf_collection(
    *,
    base_url: str,
    token: str,
    collection: str,
    strategy: str,
    pdf_paths: list[Path],
    reuse_collection: bool,
    timeout: float,
) -> dict[str, Any]:
    _json_body(
        base_url,
        "/api/collections",
        method="POST",
        payload={"name": collection},
        token=token,
        timeout=timeout,
    )
    documents = _json_body(
        base_url,
        "/api/documents?" + parse.urlencode({"collection_name": collection}),
        token=token,
        timeout=timeout,
    )
    if documents:
        if not reuse_collection:
            raise ExperimentError(
                f"Collection {collection} ja contem documentos. Use uma collection nova "
                "ou --reuse-collection apos validar o corpus."
            )
        expected_names = sorted(path.name for path in pdf_paths)
        actual_names = sorted(str(item.get("original_name")) for item in documents)
        strategies = {str(item.get("chunking_strategy")) for item in documents}
        if actual_names != expected_names or strategies != {strategy}:
            raise ExperimentError(
                f"Collection {collection} nao corresponde aos PDFs/chunking solicitados."
            )
        return {"action": "reused", "documents": documents}

    response = _multipart_upload(
        base_url,
        token,
        collection,
        strategy,
        pdf_paths,
        timeout,
    )
    if response.get("errors"):
        raise ExperimentError(f"Falhas de ingestao: {response['errors']}")
    return {"action": "uploaded", **response}


def finish_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    *,
    status: str,
    error_message: str | None = None,
) -> None:
    manifest["status"] = status
    manifest["finished_at"] = utc_now()
    if error_message:
        manifest["error"] = error_message
    manifest["artifacts_sha256"] = hash_files(
        run_dir,
        exclude={manifest_path.relative_to(run_dir).as_posix()},
    )
    write_json_atomic(manifest_path, manifest)


def run_sciq(args: argparse.Namespace) -> int:
    methods = parse_csv(args.methods, RETRIEVAL_METHODS, "metodo")
    run_id = args.run_id or default_run_id("sciq")
    run_dir = ensure_new_run_dir(args.run_dir or DEFAULT_RUNS / "sciq" / run_id)
    collection = args.collection or safe_collection_name(run_id)
    parameters = {
        "collection": collection,
        "qdrant_url": args.qdrant_url,
        "dataset_revision": args.dataset_revision,
        "embedding_model": args.embedding_model,
        "embedding_model_revision": args.embedding_model_revision or None,
        "sparse_model": args.sparse_model,
        "sparse_language": args.sparse_language,
        "methods": methods,
        "split": args.split,
        "top_k": args.top_k,
        "k_values": args.k,
        "chunking_strategy": args.chunking_strategy,
        "chunking_parameters": {
            "chunk_min_words": args.chunk_min_words,
            "chunk_size_words": args.chunk_size_words,
            "chunk_overlap_words": args.chunk_overlap_words,
            "chunk_size_tokens": args.chunk_size_tokens,
            "chunk_overlap_tokens": args.chunk_overlap_tokens,
        },
        "limit_queries": args.limit_queries,
        "seed": args.seed,
        "statistics": args.statistics,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "recreate": not args.collection or args.recreate,
    }
    manifest = base_manifest("sciq", run_id, parameters)
    manifest_path = run_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    command = [
        sys.executable,
        "benchmarks/sciq/run_all.py",
        "--run-dir",
        str(run_dir),
        "--collection",
        collection,
        "--qdrant-url",
        args.qdrant_url,
        "--sparse-language",
        args.sparse_language,
        "--dataset-revision",
        args.dataset_revision,
        "--methods",
        ",".join(methods),
        "--split",
        args.split,
        "--top-k",
        str(args.top_k),
        "--k",
        args.k,
        "--chunking-strategy",
        args.chunking_strategy,
        "--seed",
        str(args.seed),
        "--chunk-min-words",
        str(args.chunk_min_words),
        "--chunk-size-words",
        str(args.chunk_size_words),
        "--chunk-overlap-words",
        str(args.chunk_overlap_words),
    ]
    if not args.collection or args.recreate:
        command.append("--recreate")
    if args.limit_queries is not None:
        command.extend(["--limit-queries", str(args.limit_queries)])
    if args.skip_prepare:
        command.append("--skip-prepare")
    if args.plot_results:
        command.append("--plot-results")

    environment = os.environ.copy()
    environment["EMBEDDING_MODEL"] = args.embedding_model
    environment["SPARSE_MODEL"] = args.sparse_model
    environment["SPARSE_LANGUAGE"] = args.sparse_language
    environment["CHUNK_MIN_WORDS"] = str(args.chunk_min_words)
    environment["CHUNK_SIZE_WORDS"] = str(args.chunk_size_words)
    environment["CHUNK_OVERLAP_WORDS"] = str(args.chunk_overlap_words)
    environment["CHUNK_SIZE_TOKENS"] = str(args.chunk_size_tokens)
    environment["CHUNK_OVERLAP_TOKENS"] = str(args.chunk_overlap_tokens)
    if args.embedding_model_revision:
        environment["EMBEDDING_MODEL_REVISION"] = args.embedding_model_revision
    try:
        run_command(command, cwd=INGEST_API, env=environment, dry_run=args.dry_run)
        if args.statistics:
            statistics_command = [
                sys.executable,
                str(ROOT / "benchmark" / "analyze_sciq.py"),
                "--run-dir",
                str(run_dir),
                "--data-dir",
                str(INGEST_API / "benchmarks" / "sciq" / "data"),
                "--split",
                args.split,
                "--k",
                args.k,
                "--seed",
                str(args.seed),
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
            if args.limit_queries is not None:
                statistics_command.extend(["--limit-queries", str(args.limit_queries)])
            run_command(
                statistics_command,
                cwd=ROOT,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            processed = INGEST_API / "benchmarks" / "sciq" / "data" / "processed"
            manifest["processed_inputs_sha256"] = hash_files(processed)
            summary_path = run_dir / f"summary_{args.split}.json"
            if not summary_path.exists():
                raise ExperimentError(f"Resumo SciQ nao encontrado: {summary_path}")
            manifest["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        finish_manifest(manifest, manifest_path, run_dir, status="dry_run" if args.dry_run else "complete")
    except Exception as exc:
        finish_manifest(manifest, manifest_path, run_dir, status="failed", error_message=str(exc))
        raise
    print(f"Rodada SciQ: {run_dir}")
    return 0


def run_pdf_ir(args: argparse.Namespace) -> int:
    strategies = parse_csv(args.chunking_strategies, CHUNKING_STRATEGIES, "chunking")
    methods = parse_csv(args.methods, RETRIEVAL_METHODS, "metodo")
    if args.collection and len(strategies) != 1:
        raise ExperimentError("--collection so pode ser usado com um unico chunking.")
    password = args.password or os.getenv("INGEST_APP_PASSWORD", "")
    if not password and not args.dry_run:
        raise ExperimentError("Configure INGEST_APP_PASSWORD ou passe --password.")
    pdf_paths = [path.resolve() for path in args.pdf]
    for path in [args.cases.resolve(), *pdf_paths]:
        if not path.is_file():
            raise ExperimentError(f"Arquivo inexistente: {path}")
    review_summary = require_reviewed_cases(
        args.cases.resolve(),
        allow_draft=args.allow_draft_cases,
    )

    run_id = args.run_id or default_run_id("pdf-ir")
    run_dir = ensure_new_run_dir(args.run_dir or DEFAULT_RUNS / "pdf-ir" / run_id)
    parameters = {
        "cases": str(args.cases.resolve()),
        "cases_sha256": sha256_file(args.cases.resolve()),
        "case_review_statuses": review_summary,
        "allow_draft_cases": args.allow_draft_cases,
        "pdfs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in pdf_paths
        ],
        "chunking_strategies": strategies,
        "methods": methods,
        "top_k": args.top_k,
        "qdrant_url": args.qdrant_url,
        "ingest_url": args.ingest_url,
        "reuse_collection": args.reuse_collection,
        "min_coverage": args.min_coverage,
        "plot_results": args.plot_results,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "source_audit_extractor": args.source_audit_extractor,
    }
    manifest = base_manifest("pdf-ir", run_id, parameters)
    manifest["conditions"] = []
    manifest_path = run_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    try:
        source_audit_path = run_dir / "source-audit.json"
        source_audit_command = [
            sys.executable,
            str(ROOT / "benchmark" / "groundtruth" / "audit_pdf.py"),
            "--cases",
            str(args.cases.resolve()),
            "--report",
            str(source_audit_path),
            "--extractor",
            args.source_audit_extractor,
        ]
        for pdf_path in pdf_paths:
            source_audit_command.extend(["--pdf", str(pdf_path)])
        run_command(source_audit_command, cwd=ROOT, dry_run=args.dry_run)
        if not args.dry_run:
            manifest["source_audit"] = json.loads(
                source_audit_path.read_text(encoding="utf-8")
            )["summary"]

        token = "" if args.dry_run else authenticate(args.ingest_url, password, args.timeout)
        if not args.dry_run:
            manifest["ingest_configuration"] = _json_body(
                args.ingest_url,
                "/api/experiment-config",
                token=token,
                timeout=args.timeout,
            )
        password_environment = os.environ.copy()
        password_environment["INGEST_APP_PASSWORD"] = password
        for strategy in strategies:
            collection = args.collection or safe_collection_name(
                args.collection_prefix or pdf_paths[0].stem,
                strategy,
                run_id.removeprefix("pdf-ir-"),
            )
            condition_dir = run_dir / strategy
            condition_dir.mkdir(parents=True, exist_ok=True)
            condition: dict[str, Any] = {
                "strategy": strategy,
                "collection": collection,
                "directory": str(condition_dir),
            }
            manifest["conditions"].append(condition)

            if args.dry_run:
                condition["ingestion"] = {"action": "dry_run"}
            else:
                condition["ingestion"] = prepare_pdf_collection(
                    base_url=args.ingest_url,
                    token=token,
                    collection=collection,
                    strategy=strategy,
                    pdf_paths=pdf_paths,
                    reuse_collection=args.reuse_collection,
                    timeout=args.timeout,
                )

            chunks_path = condition_dir / "chunks.jsonl"
            ingest_cases = condition_dir / "ingest-groundtruth.jsonl"
            ragas_cases = condition_dir / "ragas-groundtruth.jsonl"
            matching_report = condition_dir / "matching-report.json"
            retrieval_dir = condition_dir / "retrieval"

            export_command = [
                sys.executable,
                str(INGEST_API / "benchmarks" / "groundtruth" / "export_chunks.py"),
                "--collection",
                collection,
                "--qdrant-url",
                args.qdrant_url,
                "--output",
                str(chunks_path),
            ]
            for pdf_path in pdf_paths:
                export_command.extend(["--pdf", str(pdf_path)])
            run_command(export_command, cwd=ROOT, dry_run=args.dry_run)

            materialize_command = [
                sys.executable,
                str(ROOT / "benchmark" / "groundtruth" / "materialize.py"),
                "materialize",
                "--cases",
                str(args.cases.resolve()),
                "--chunks",
                str(chunks_path),
                "--ingest-out",
                str(ingest_cases),
                "--ragas-out",
                str(ragas_cases),
                "--report-out",
                str(matching_report),
                "--min-coverage",
                str(args.min_coverage),
            ]
            run_command(materialize_command, cwd=ROOT, dry_run=args.dry_run)

            retrieval_command = [
                sys.executable,
                "benchmarks/groundtruth/run_groundtruth.py",
                "--cases",
                str(ingest_cases),
                "--run-dir",
                str(retrieval_dir),
                "--base-url",
                args.ingest_url,
                "--collection",
                collection,
                "--methods",
                ",".join(methods),
                "--top-k",
                str(args.top_k),
                "--timeout",
                str(args.timeout),
            ]
            run_command(
                retrieval_command,
                cwd=INGEST_API,
                env=password_environment,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                condition["matching_report"] = json.loads(
                    matching_report.read_text(encoding="utf-8")
                )["summary"]
                condition["retrieval_summary"] = json.loads(
                    (retrieval_dir / "summary.json").read_text(encoding="utf-8")
                )

        if args.plot_results:
            plot_command = [
                sys.executable,
                str(ROOT / "benchmark" / "plot_pdf_ir.py"),
                "--run-dir",
                str(run_dir),
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
                "--seed",
                str(args.seed),
            ]
            run_command(plot_command, cwd=ROOT, dry_run=args.dry_run)

        finish_manifest(manifest, manifest_path, run_dir, status="dry_run" if args.dry_run else "complete")
    except Exception as exc:
        finish_manifest(manifest, manifest_path, run_dir, status="failed", error_message=str(exc))
        raise
    print(f"Rodada PDF/RI: {run_dir}")
    return 0


def run_rag(args: argparse.Namespace) -> int:
    methods = parse_csv(args.methods, RETRIEVAL_METHODS, "metodo")
    password = args.password or os.getenv("CHAT_APP_PASSWORD", "")
    if not password and not args.dry_run:
        raise ExperimentError("Configure CHAT_APP_PASSWORD ou passe --password.")
    cases_path = args.cases.resolve()
    if not cases_path.is_file():
        raise ExperimentError(f"Arquivo inexistente: {cases_path}")
    review_summary = require_reviewed_cases(
        cases_path,
        allow_draft=args.allow_draft_cases,
    )

    run_id = args.run_id or default_run_id("rag")
    run_dir = ensure_new_run_dir(args.run_dir or DEFAULT_RUNS / "rag" / run_id)
    parameters = {
        "cases": str(cases_path),
        "cases_sha256": sha256_file(cases_path),
        "case_review_statuses": review_summary,
        "allow_draft_cases": args.allow_draft_cases,
        "collection": args.collection,
        "methods": methods,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k if args.use_reranker else args.top_k,
        "use_reranker": args.use_reranker,
        "chat_url": args.chat_url,
        "evaluate_ragas": args.evaluate_ragas,
        "ragas_llm_base_url": args.ragas_llm_base_url,
        "ragas_llm_model": args.ragas_llm_model,
        "ragas_embedding_model": args.ragas_embedding_model,
    }
    manifest = base_manifest("rag", run_id, parameters)
    manifest["conditions"] = []
    manifest_path = run_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CHAT_APP_PASSWORD"] = password

    try:
        if not args.dry_run:
            chat_token = authenticate(args.chat_url, password, args.timeout)
            manifest["chat_configuration"] = _json_body(
                args.chat_url,
                "/api/experiment-config",
                token=chat_token,
                timeout=args.timeout,
            )
        for method in methods:
            condition_dir = run_dir / method
            condition_dir.mkdir(parents=True, exist_ok=True)
            condition = {"method": method, "directory": str(condition_dir)}
            manifest["conditions"].append(condition)
            collect_command = [
                sys.executable,
                "benchmarks/ragas/run_groundtruth.py",
                "--cases",
                str(cases_path),
                "--run-dir",
                str(condition_dir),
                "--base-url",
                args.chat_url,
                "--collection",
                args.collection,
                "--method",
                method,
                "--top-k",
                str(args.top_k),
                "--candidate-k",
                str(args.candidate_k if args.use_reranker else args.top_k),
                "--timeout",
                str(args.timeout),
            ]
            if not args.use_reranker:
                collect_command.append("--no-reranker")
            run_command(
                collect_command,
                cwd=CHAT_API,
                env=environment,
                dry_run=args.dry_run,
            )
            if args.evaluate_ragas:
                evaluate_command = [
                    sys.executable,
                    "benchmarks/ragas/evaluate_official.py",
                    "--results",
                    str(condition_dir / "results.jsonl"),
                    "--output-dir",
                    str(condition_dir / "ragas-official"),
                    "--llm-base-url",
                    args.ragas_llm_base_url,
                    "--llm-model",
                    args.ragas_llm_model,
                    "--embedding-model",
                    args.ragas_embedding_model,
                ]
                run_command(
                    evaluate_command,
                    cwd=CHAT_API,
                    env=environment,
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                condition["collection_summary"] = json.loads(
                    (condition_dir / "summary.json").read_text(encoding="utf-8")
                )
                if args.evaluate_ragas:
                    condition["ragas_summary"] = json.loads(
                        (condition_dir / "ragas-official" / "official_ragas_summary.json").read_text(
                            encoding="utf-8"
                        )
                    )

        finish_manifest(manifest, manifest_path, run_dir, status="dry_run" if args.dry_run else "complete")
    except Exception as exc:
        finish_manifest(manifest, manifest_path, run_dir, status="failed", error_message=str(exc))
        raise
    print(f"Rodada RAG: {run_dir}")
    return 0


def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orquestra as rodadas academicas sem depender das interfaces web."
    )
    subparsers = parser.add_subparsers(dest="study", required=True)

    sciq = subparsers.add_parser("sciq", help="baseline controlada SciQ")
    add_common_run_arguments(sciq)
    sciq.add_argument("--collection", default="")
    sciq.add_argument("--recreate", action="store_true")
    sciq.add_argument("--qdrant-url", default="http://localhost:6335")
    sciq.add_argument(
        "--dataset-revision",
        default=os.getenv("SCIQ_DATASET_REVISION", DEFAULT_SCIQ_DATASET_REVISION),
    )
    sciq.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    sciq.add_argument("--embedding-model-revision", default="")
    sciq.add_argument(
        "--sparse-model",
        default=os.getenv("SPARSE_MODEL", "Qdrant/bm25"),
    )
    sciq.add_argument("--sparse-language", default="english")
    sciq.add_argument("--methods", default="bm25,dense,hybrid")
    sciq.add_argument("--split", choices=["train", "validation", "test"], default="test")
    sciq.add_argument("--top-k", type=int, default=10)
    sciq.add_argument("--k", default="1,3,5,10")
    sciq.add_argument(
        "--chunking-strategy",
        choices=["fixed_token", "recursive_text"],
        default="recursive_text",
    )
    sciq.add_argument("--limit-queries", type=int, default=None)
    sciq.add_argument("--seed", type=int, default=42)
    sciq.add_argument("--bootstrap-repetitions", type=int, default=2000)
    sciq.add_argument("--chunk-min-words", type=int, default=180)
    sciq.add_argument("--chunk-size-words", type=int, default=700)
    sciq.add_argument("--chunk-overlap-words", type=int, default=100)
    sciq.add_argument("--chunk-size-tokens", type=int, default=512)
    sciq.add_argument("--chunk-overlap-tokens", type=int, default=64)
    sciq.add_argument("--skip-prepare", action="store_true")
    sciq.add_argument(
        "--no-plot-results",
        dest="plot_results",
        action="store_false",
    )
    sciq.add_argument(
        "--no-statistics",
        dest="statistics",
        action="store_false",
    )
    sciq.set_defaults(plot_results=True, statistics=True, handler=run_sciq)

    pdf_ir = subparsers.add_parser("pdf-ir", help="ingestao e avaliacao de RI em PDFs")
    add_common_run_arguments(pdf_ir)
    pdf_ir.add_argument("--pdf", type=Path, action="append", required=True)
    pdf_ir.add_argument("--cases", type=Path, required=True)
    pdf_ir.add_argument("--allow-draft-cases", action="store_true")
    pdf_ir.add_argument("--collection", default="")
    pdf_ir.add_argument("--collection-prefix", default="")
    pdf_ir.add_argument("--reuse-collection", action="store_true")
    pdf_ir.add_argument("--chunking-strategies", default="recursive_text")
    pdf_ir.add_argument("--methods", default="bm25,dense,hybrid")
    pdf_ir.add_argument("--top-k", type=int, default=10)
    pdf_ir.add_argument("--min-coverage", type=float, default=1.0)
    pdf_ir.add_argument("--bootstrap-repetitions", type=int, default=2000)
    pdf_ir.add_argument("--seed", type=int, default=42)
    pdf_ir.add_argument(
        "--source-audit-extractor",
        choices=["auto", "pypdf", "pdftotext"],
        default="auto",
    )
    pdf_ir.add_argument("--qdrant-url", default="http://localhost:6335")
    pdf_ir.add_argument("--ingest-url", default="http://localhost:8010")
    pdf_ir.add_argument("--password", default="", help=argparse.SUPPRESS)
    pdf_ir.add_argument("--timeout", type=float, default=900)
    pdf_ir.add_argument(
        "--no-plot-results",
        dest="plot_results",
        action="store_false",
    )
    pdf_ir.set_defaults(plot_results=True, handler=run_pdf_ir)

    rag = subparsers.add_parser("rag", help="extensao RAG/RAGAS sobre casos aprovados")
    add_common_run_arguments(rag)
    rag.add_argument("--cases", type=Path, required=True)
    rag.add_argument("--allow-draft-cases", action="store_true")
    rag.add_argument("--collection", required=True)
    rag.add_argument("--methods", default="bm25,dense,hybrid")
    rag.add_argument("--top-k", type=int, default=5)
    rag.add_argument("--candidate-k", type=int, default=20)
    rag.add_argument("--use-reranker", action="store_true")
    rag.add_argument("--chat-url", default="http://localhost:8011")
    rag.add_argument("--password", default="", help=argparse.SUPPRESS)
    rag.add_argument("--timeout", type=float, default=180)
    rag.add_argument("--evaluate-ragas", action="store_true")
    rag.add_argument("--ragas-llm-base-url", default="https://openrouter.ai/api/v1")
    rag.add_argument("--ragas-llm-model", default="deepseek/deepseek-v4-flash")
    rag.add_argument("--ragas-embedding-model", default="BAAI/bge-m3")
    rag.set_defaults(handler=run_rag)
    return parser


def validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in (
        "top_k",
        "candidate_k",
        "limit_queries",
        "timeout",
        "bootstrap_repetitions",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ExperimentError(f"--{name.replace('_', '-')} deve ser positivo.")
    if hasattr(args, "bootstrap_repetitions") and args.bootstrap_repetitions < 100:
        raise ExperimentError("--bootstrap-repetitions deve ser >= 100.")
    if hasattr(args, "min_coverage") and not 0 < args.min_coverage <= 1:
        raise ExperimentError("--min-coverage deve estar no intervalo (0, 1].")
    if hasattr(args, "chunk_size_words"):
        if args.chunk_min_words <= 0 or args.chunk_size_words <= 0:
            raise ExperimentError("Tamanhos de chunk em palavras devem ser positivos.")
        if not 0 <= args.chunk_overlap_words < args.chunk_size_words:
            raise ExperimentError("Overlap em palavras deve ser >= 0 e menor que o chunk.")
        if args.chunk_size_tokens < 32:
            raise ExperimentError("--chunk-size-tokens deve ser >= 32.")
        if not 0 <= args.chunk_overlap_tokens < args.chunk_size_tokens:
            raise ExperimentError("Overlap em tokens deve ser >= 0 e menor que o chunk.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_positive_arguments(args)
        return int(args.handler(args))
    except (ExperimentError, OSError, subprocess.CalledProcessError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
