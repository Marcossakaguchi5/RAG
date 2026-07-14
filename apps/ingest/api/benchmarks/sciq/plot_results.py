from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from common import DEFAULT_DATA_DIR, read_jsonl
from evaluate_retrieval import load_qrels, load_run
from metrics import calculate_query_metrics


METRICS = ["hit_rate", "precision", "recall", "map", "ndcg", "mrr"]
METHOD_COLORS = {
    "bm25": "#2f6f9f",
    "dense": "#7a4f9f",
    "hybrid": "#1f8a70",
}
TEXT_COLOR = "#24313f"
GRID_COLOR = "#d9e1e8"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def discover_source(data_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir

    timestamped_runs = [
        path
        for path in (data_dir / "runs").glob("*")
        if path.is_dir() and any(path.glob("summary_*.json"))
    ]
    if timestamped_runs:
        return max(timestamped_runs, key=lambda path: path.stat().st_mtime)
    return data_dir


def results_dir_for(source_dir: Path) -> Path:
    if (source_dir / "results").is_dir():
        return source_dir / "results"
    return source_dir / "results"


def retrieval_dir_for(source_dir: Path, data_dir: Path) -> Path:
    if (source_dir / "retrieval").is_dir():
        return source_dir / "retrieval"
    if source_dir == data_dir:
        return data_dir / "runs"
    return source_dir / "retrieval"


def default_output_dir(source_dir: Path, data_dir: Path) -> Path:
    if source_dir == data_dir:
        return data_dir / "plots"
    return source_dir / "plots"


def ensure_output_dir(output_dir: Path, source_dir: Path, data_dir: Path, explicit: bool) -> Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except PermissionError:
        if explicit:
            raise
        fallback = data_dir / "plots" / source_dir.name
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Sem permissao para escrever em {output_dir}; usando {fallback}.")
        return fallback


def infer_method(path: Path, payload: dict[str, Any] | None = None) -> str:
    run_path = str((payload or {}).get("run_path") or "")
    for candidate in [path.name, run_path]:
        match = re.search(r"(bm25|dense|hybrid)", candidate)
        if match:
            return match.group(1)
    return path.stem.split("_", 1)[0]


def load_metric_payloads(source_dir: Path) -> dict[str, dict[str, Any]]:
    result_dir = results_dir_for(source_dir)
    payloads = {}
    for path in sorted(result_dir.glob("*_metrics.json")):
        payload = read_json(path)
        payloads[infer_method(path, payload)] = payload
    if not payloads:
        raise SystemExit(f"Nenhum *_metrics.json encontrado em {result_dir}.")
    return payloads


def metric_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method, payload in sorted(payloads.items()):
        for k_label, values in sorted(
            payload.get("metrics", {}).items(),
            key=lambda item: int(str(item[0]).removeprefix("@")),
        ):
            row = {
                "method": method,
                "split": payload.get("split", ""),
                "k": int(str(k_label).removeprefix("@")),
                "qrels_queries": payload.get("qrels_queries"),
                "run_queries": payload.get("run_queries"),
                "missing_queries": payload.get("missing_queries"),
            }
            row.update({metric: values.get(metric, 0.0) for metric in METRICS})
            rows.append(row)
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["method", "split", "k", "qrels_queries", "run_queries", "missing_queries", *METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#24313f} .title{font-size:20px;font-weight:700}",
        ".axis{stroke:#24313f;stroke-width:1.2} .grid{stroke:#d9e1e8;stroke-width:1} .label{font-size:12px}",
        ".tick{font-size:11px;fill:#5d6b78} .legend{font-size:12px}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def svg_footer() -> str:
    return "</svg>\n"


def line(x1: float, y1: float, x2: float, y2: float, class_name: str = "axis") -> str:
    return f'<line class="{class_name}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'


def text(x: float, y: float, value: str, class_name: str = "label", anchor: str = "middle") -> str:
    return f'<text class="{class_name}" x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}">{html.escape(value)}</text>'


def value_to_y(value: float, top: float, height: float, max_value: float) -> float:
    if max_value <= 0:
        return top + height
    return top + height - (value / max_value) * height


def render_legend(methods: list[str], x: float, y: float) -> list[str]:
    parts = []
    for index, method in enumerate(methods):
        offset = index * 92
        color = METHOD_COLORS.get(method, "#65717c")
        parts.append(f'<rect x="{x + offset:.2f}" y="{y - 10:.2f}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(text(x + offset + 20, y + 1, method, "legend", "start"))
    return parts


def bar_chart_at_k(rows: list[dict[str, Any]], output_path: Path, selected_k: int | None = None) -> None:
    if selected_k is None:
        selected_k = max(int(row["k"]) for row in rows)
    chart_rows = [row for row in rows if int(row["k"]) == selected_k]
    methods = sorted({row["method"] for row in chart_rows})
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 80, 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = 1.0
    group_width = plot_width / len(METRICS)
    bar_width = min(34, group_width / max(1, len(methods) + 1))

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, f"Metricas de recuperacao @ {selected_k}", "title"))
    parts.extend(render_legend(methods, left, 62))

    for tick in range(0, 6):
        value = tick / 5
        y = value_to_y(value, top, plot_height, max_value)
        parts.append(line(left, y, width - right, y, "grid"))
        parts.append(text(left - 12, y + 4, f"{value:.1f}", "tick", "end"))
    parts.append(line(left, top, left, top + plot_height))
    parts.append(line(left, top + plot_height, width - right, top + plot_height))

    values_by_method = {row["method"]: row for row in chart_rows}
    for metric_index, metric in enumerate(METRICS):
        center = left + group_width * metric_index + group_width / 2
        start_x = center - (len(methods) * bar_width + (len(methods) - 1) * 8) / 2
        for method_index, method in enumerate(methods):
            value = float(values_by_method.get(method, {}).get(metric, 0.0))
            bar_x = start_x + method_index * (bar_width + 8)
            bar_y = value_to_y(value, top, plot_height, max_value)
            color = METHOD_COLORS.get(method, "#65717c")
            parts.append(
                f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" '
                f'height="{top + plot_height - bar_y:.2f}" fill="{color}" rx="3"/>'
            )
            parts.append(text(bar_x + bar_width / 2, bar_y - 5, f"{value:.2f}", "tick"))
        parts.append(text(center, top + plot_height + 32, metric, "label"))

    parts.append(svg_footer())
    write_text(output_path, "\n".join(parts))


def metric_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    methods = sorted({row["method"] for row in rows})
    k_values = sorted({int(row["k"]) for row in rows})
    cells = [metric for metric in METRICS if metric != "precision"]
    width, height = 1120, 760
    margin_x, margin_y = 70, 90
    gap_x, gap_y = 50, 70
    cols, rows_count = 2, math.ceil(len(cells) / 2)
    cell_w = (width - margin_x * 2 - gap_x) / cols
    cell_h = (height - margin_y - 60 - gap_y * (rows_count - 1)) / rows_count
    values = {(row["method"], int(row["k"])): row for row in rows}

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "Curvas por k", "title"))
    parts.extend(render_legend(methods, margin_x, 62))

    for metric_index, metric in enumerate(cells):
        col = metric_index % cols
        row_index = metric_index // cols
        x0 = margin_x + col * (cell_w + gap_x)
        y0 = margin_y + row_index * (cell_h + gap_y)
        max_value = 1.0
        parts.append(text(x0 + cell_w / 2, y0 - 16, metric, "label"))
        for tick in range(0, 6):
            value = tick / 5
            y = value_to_y(value, y0, cell_h, max_value)
            parts.append(line(x0, y, x0 + cell_w, y, "grid"))
            parts.append(text(x0 - 8, y + 4, f"{value:.1f}", "tick", "end"))
        parts.append(line(x0, y0, x0, y0 + cell_h))
        parts.append(line(x0, y0 + cell_h, x0 + cell_w, y0 + cell_h))
        for k in k_values:
            x = x0 + (k_values.index(k) / max(1, len(k_values) - 1)) * cell_w
            parts.append(text(x, y0 + cell_h + 22, str(k), "tick"))
        parts.append(text(x0 + cell_w / 2, y0 + cell_h + 42, "k", "tick"))

        for method in methods:
            points = []
            for k in k_values:
                x = x0 + (k_values.index(k) / max(1, len(k_values) - 1)) * cell_w
                value = float(values.get((method, k), {}).get(metric, 0.0))
                y = value_to_y(value, y0, cell_h, max_value)
                points.append((x, y, value))
            color = METHOD_COLORS.get(method, "#65717c")
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y, _ in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    parts.append(svg_footer())
    write_text(output_path, "\n".join(parts))


def load_retrieval_rows(retrieval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs = {}
    for path in sorted(retrieval_dir.glob("*.jsonl")):
        match = re.search(r"(bm25|dense|hybrid)", path.name)
        if not match:
            continue
        runs[match.group(1)] = list(read_jsonl(path))
    return runs


def latency_chart(retrieval_rows: dict[str, list[dict[str, Any]]], output_path: Path) -> bool:
    latencies = {}
    for method, rows in retrieval_rows.items():
        by_query = {}
        for row in rows:
            if "latency_ms" in row:
                by_query[str(row["query_id"])] = float(row["latency_ms"])
        if by_query:
            latencies[method] = mean(by_query.values())
    if not latencies:
        return False

    methods = sorted(latencies)
    width, height = 820, 480
    left, right, top, bottom = 95, 35, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(latencies.values()) * 1.15
    bar_width = min(80, plot_width / max(1, len(methods) * 2))

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "Latencia media por query", "title"))
    for tick in range(0, 6):
        value = max_value * tick / 5
        y = value_to_y(value, top, plot_height, max_value)
        parts.append(line(left, y, width - right, y, "grid"))
        parts.append(text(left - 12, y + 4, f"{value:.0f} ms", "tick", "end"))
    parts.append(line(left, top, left, top + plot_height))
    parts.append(line(left, top + plot_height, width - right, top + plot_height))

    for index, method in enumerate(methods):
        center = left + (index + 0.5) * plot_width / len(methods)
        value = latencies[method]
        y = value_to_y(value, top, plot_height, max_value)
        color = METHOD_COLORS.get(method, "#65717c")
        parts.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{top + plot_height - y:.2f}" fill="{color}" rx="4"/>'
        )
        parts.append(text(center, y - 7, f"{value:.0f} ms", "tick"))
        parts.append(text(center, top + plot_height + 28, method, "label"))

    parts.append(svg_footer())
    write_text(output_path, "\n".join(parts))
    return True


def first_relevant_ranks(retrieval_dir: Path, data_dir: Path, split: str, k_values: list[int]) -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = {}
    qrels = load_qrels(data_dir / "processed" / "qrels.jsonl", split)
    for run_path in sorted(retrieval_dir.glob("*.jsonl")):
        method = infer_method(run_path)
        run = load_run(run_path, split)
        max_k = max(k_values)
        counter: Counter[str] = Counter()
        for query_id, relevant_ids in qrels.items():
            ranked = run.get(query_id, [])[:max_k]
            rank = next((index for index, doc_id in enumerate(ranked, start=1) if doc_id in relevant_ids), None)
            counter[str(rank) if rank is not None else "miss"] += 1
        counters[method] = counter
    return counters


def hit_rank_chart(counters: dict[str, Counter[str]], output_path: Path) -> bool:
    if not counters:
        return False
    methods = sorted(counters)
    rank_labels = sorted(
        {label for counter in counters.values() for label in counter if label != "miss"},
        key=lambda value: int(value),
    )
    labels = rank_labels + ["miss"]
    width, height = 1050, 560
    left, right, top, bottom = 85, 35, 70, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(max(counter.values()) for counter in counters.values()) or 1
    group_width = plot_width / len(labels)
    bar_width = min(28, group_width / max(1, len(methods) + 1))

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "Posicao do primeiro documento relevante", "title"))
    parts.extend(render_legend(methods, left, 62))
    for tick in range(0, 6):
        value = max_value * tick / 5
        y = value_to_y(value, top, plot_height, max_value)
        parts.append(line(left, y, width - right, y, "grid"))
        parts.append(text(left - 10, y + 4, f"{value:.0f}", "tick", "end"))
    parts.append(line(left, top, left, top + plot_height))
    parts.append(line(left, top + plot_height, width - right, top + plot_height))

    for label_index, label in enumerate(labels):
        center = left + group_width * label_index + group_width / 2
        start_x = center - (len(methods) * bar_width + (len(methods) - 1) * 8) / 2
        for method_index, method in enumerate(methods):
            value = counters[method].get(label, 0)
            x = start_x + method_index * (bar_width + 8)
            y = value_to_y(value, top, plot_height, max_value)
            color = METHOD_COLORS.get(method, "#65717c")
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{top + plot_height - y:.2f}" fill="{color}" rx="3"/>'
            )
            if value:
                parts.append(text(x + bar_width / 2, y - 5, str(value), "tick"))
        parts.append(text(center, top + plot_height + 30, label if label == "miss" else f"rank {label}", "label"))

    parts.append(svg_footer())
    write_text(output_path, "\n".join(parts))
    return True


def write_report(output_dir: Path, rows: list[dict[str, Any]], images: list[str], source_dir: Path) -> None:
    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in ["method", "k", *METRICS]
        )
        + "</tr>"
        for row in rows
    )
    image_blocks = "\n".join(
        f'<section><h2>{html.escape(Path(image).stem.replace("_", " ").title())}</h2>'
        f'<img src="{html.escape(image)}" alt="{html.escape(image)}"/></section>'
        for image in images
    )
    content = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <title>SciQ Retrieval Benchmark</title>
  <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#24313f;background:#f7f9fb}}
    h1{{margin-bottom:4px}} code{{background:#e9eef3;padding:2px 5px;border-radius:4px}}
    img{{max-width:100%;background:#fff;border:1px solid #d9e1e8;margin:8px 0 28px}}
    table{{border-collapse:collapse;background:#fff;margin-top:16px}} th,td{{border:1px solid #d9e1e8;padding:6px 9px;text-align:right}}
    th:first-child,td:first-child{{text-align:left}} th{{background:#eef3f7}}
  </style>
</head>
<body>
  <h1>SciQ Retrieval Benchmark</h1>
  <p>Fonte: <code>{html.escape(str(source_dir))}</code></p>
  {image_blocks}
  <h2>Tabela consolidada</h2>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(column)}</th>" for column in ["method", "k", *METRICS])}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    write_text(output_dir / "report.html", content)


def plot(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = discover_source(args.data_dir, args.run_dir)
    output_dir = args.output_dir or default_output_dir(source_dir, args.data_dir)
    payloads = load_metric_payloads(source_dir)
    rows = metric_rows(payloads)
    selected_k = args.k or max(int(row["k"]) for row in rows)
    retrieval_dir = retrieval_dir_for(source_dir, args.data_dir)

    output_dir = ensure_output_dir(output_dir, source_dir, args.data_dir, explicit=args.output_dir is not None)
    write_metrics_csv(output_dir / "metrics_consolidated.csv", rows)

    images = []
    bar_path = output_dir / f"metrics_at_{selected_k}.svg"
    bar_chart_at_k(rows, bar_path, selected_k)
    images.append(bar_path.name)

    curves_path = output_dir / "metric_curves.svg"
    metric_curves(rows, curves_path)
    images.append(curves_path.name)

    retrieval_rows = load_retrieval_rows(retrieval_dir)
    latency_path = output_dir / "latency_by_method.svg"
    if latency_chart(retrieval_rows, latency_path):
        images.append(latency_path.name)

    split = str(next(iter(payloads.values())).get("split", "test"))
    k_values = sorted({int(row["k"]) for row in rows})
    rank_path = output_dir / "first_relevant_rank.svg"
    if hit_rank_chart(first_relevant_ranks(retrieval_dir, args.data_dir, split, k_values), rank_path):
        images.append(rank_path.name)

    write_report(output_dir, rows, images, source_dir)
    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "images": images,
        "csv": str(output_dir / "metrics_consolidated.csv"),
        "report": str(output_dir / "report.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera graficos SVG para os resultados do benchmark SciQ.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=None, help="Valor de k para o grafico de barras. Padrao: maior k.")
    args = parser.parse_args()

    summary = plot(args)
    print("Graficos gerados:")
    print(f"  origem: {summary['source_dir']}")
    print(f"  saida: {summary['output_dir']}")
    for image in summary["images"]:
        print(f"  - {image}")
    print(f"  - {Path(summary['csv']).name}")
    print(f"  - {Path(summary['report']).name}")


if __name__ == "__main__":
    main()
