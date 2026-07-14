#!/usr/bin/env python3
"""Generate three slide-ready SVG charts from the two main PDF-IR runs.

The script intentionally uses only the Python standard library.  SVG is kept as
the output format because it stays sharp when resized in PowerPoint, LibreOffice
or Google Slides.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path


METHODS = ("bm25", "dense", "hybrid")
METRICS = (
    ("mrr", "MRR@10"),
    ("map", "MAP@10"),
    ("ndcg_at_k", "nDCG@10"),
    ("recall_at_k", "Recall@10"),
)
METHOD_LABELS = {"bm25": "BM25", "dense": "Denso", "hybrid": "Híbrido (RRF)"}
HEATMAP_METHOD_LABELS = {"bm25": "BM25", "dense": "Denso", "hybrid": "Híbrido"}
METHOD_COLORS = {"bm25": "#356A9A", "dense": "#8B5DA8", "hybrid": "#23846B"}


@dataclass(frozen=True)
class MetricResult:
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class Corpus:
    name: str
    pages: int
    run_dir: Path


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_metrics(corpus: Corpus) -> dict[tuple[str, str], MetricResult]:
    results: dict[tuple[str, str], MetricResult] = {}
    summary_path = corpus.run_dir / "plots" / "metrics_with_ci.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Métricas não encontradas: {summary_path}")
    with summary_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["chunking_strategy"] != "recursive_text":
                continue
            method, metric = row["method"], row["metric"]
            if method in METHODS and metric != "map" and metric in {item[0] for item in METRICS}:
                results[(method, metric)] = MetricResult(
                    mean=float(row["mean"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                )

    # MAP is recalculated at each cutoff by the pipeline, so select its top-10 value.
    by_k_path = corpus.run_dir / "plots" / "metrics_by_k.csv"
    if not by_k_path.is_file():
        raise FileNotFoundError(f"Métricas por cutoff não encontradas: {by_k_path}")
    with by_k_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row["chunking_strategy"] == "recursive_text"
                and row["method"] in METHODS
                and row["metric"] == "map"
                and int(row["k"]) == 10
            ):
                results[(row["method"], "map")] = MetricResult(
                    mean=float(row["mean"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                )

    expected = {(method, metric) for method in METHODS for metric, _ in METRICS}
    missing = expected - set(results)
    if missing:
        formatted = ", ".join(f"{method}/{metric}" for method, metric in sorted(missing))
        raise ValueError(f"A rodada {corpus.name} não possui as métricas esperadas: {formatted}")
    return results


def svg_start(title: str, subtitle: str, width: int = 1600, height: int = 900) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#172033}'
        '.title{font-size:36px;font-weight:700}.subtitle{font-size:20px;fill:#526175}'
        '.caption{font-size:17px;fill:#526175}.tick{font-size:18px;fill:#42526A}'
        '.label{font-size:20px;font-weight:700}.value{font-size:19px;font-weight:700}'
        '.grid{stroke:#DCE3EC;stroke-width:1.5}.axis{stroke:#718096;stroke-width:1.5}'
        '.ci{stroke:#172033;stroke-width:2.5}'
        '</style>',
        f'<text class="title" x="100" y="72">{escape(title)}</text>',
        f'<text class="subtitle" x="100" y="106">{escape(subtitle)}</text>',
    ]


def render_legend(parts: list[str], *, x: int, y: int) -> None:
    for index, method in enumerate(METHODS):
        item_x = x + index * 235
        parts.append(f'<rect x="{item_x}" y="{y - 15}" width="18" height="18" rx="3" fill="{METHOD_COLORS[method]}"/>')
        parts.append(f'<text class="tick" x="{item_x + 28}" y="{y}">{escape(METHOD_LABELS[method])}</text>')


def render_y_axis(parts: list[str], *, left: float, right: float, top: float, bottom: float) -> None:
    height = bottom - top
    for tick in range(6):
        value = tick / 5
        y = bottom - value * height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 16}" y="{y + 6:.1f}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')


def render_metric_table(
    parts: list[str],
    *,
    top: float,
    value_for: callable,
    first_header: str = "Recuperador",
) -> None:
    """Render a compact, non-overlapping table of the plotted values."""
    left, right = 245, 1515
    method_width = 270
    row_height, header_height = 38, 40
    metric_width = (right - left - method_width) / len(METRICS)
    bottom = top + header_height + row_height * len(METHODS)
    parts.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" rx="5" fill="white" stroke="#C7D0DC" stroke-width="1.5"/>')
    parts.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{header_height}" rx="5" fill="#EEF3F8"/>')
    parts.append(f'<text class="label" x="{left + 16}" y="{top + 27}">{escape(first_header)}</text>')
    for metric_index, (_metric, label) in enumerate(METRICS):
        center = left + method_width + (metric_index + 0.5) * metric_width
        parts.append(f'<text class="label" x="{center:.1f}" y="{top + 27}" text-anchor="middle">{label}</text>')
        if metric_index:
            x = left + method_width + metric_index * metric_width
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#D9E1EA" stroke-width="1"/>')
    parts.append(f'<line x1="{left + method_width}" y1="{top}" x2="{left + method_width}" y2="{bottom}" stroke="#C7D0DC" stroke-width="1.5"/>')
    for method_index, method in enumerate(METHODS):
        row_top = top + header_height + method_index * row_height
        if method_index % 2 == 1:
            parts.append(f'<rect x="{left + 1}" y="{row_top}" width="{right - left - 2}" height="{row_height}" fill="#F8FAFC"/>')
        if method_index:
            parts.append(f'<line x1="{left}" y1="{row_top}" x2="{right}" y2="{row_top}" stroke="#D9E1EA" stroke-width="1"/>')
        parts.append(f'<circle cx="{left + 20}" cy="{row_top + row_height / 2:.1f}" r="7" fill="{METHOD_COLORS[method]}"/>')
        parts.append(f'<text class="tick" x="{left + 36}" y="{row_top + 25}">{escape(METHOD_LABELS[method])}</text>')
        for metric_index, (metric, _label) in enumerate(METRICS):
            center = left + method_width + (metric_index + 0.5) * metric_width
            parts.append(f'<text class="value" x="{center:.1f}" y="{row_top + 25}" text-anchor="middle">{escape(value_for(method, metric))}</text>')


def render_cutoff_table(
    parts: list[str],
    *,
    top: float,
    values: dict[tuple[str, int], MetricResult],
    cutoffs: tuple[int, ...],
) -> None:
    left, right = 245, 1515
    method_width = 270
    row_height, header_height = 38, 40
    cutoff_width = (right - left - method_width) / len(cutoffs)
    bottom = top + header_height + row_height * len(METHODS)
    parts.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" rx="5" fill="white" stroke="#C7D0DC" stroke-width="1.5"/>')
    parts.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{header_height}" rx="5" fill="#EEF3F8"/>')
    parts.append(f'<text class="label" x="{left + 16}" y="{top + 27}">Recuperador</text>')
    for index, cutoff in enumerate(cutoffs):
        center = left + method_width + (index + 0.5) * cutoff_width
        parts.append(f'<text class="label" x="{center:.1f}" y="{top + 27}" text-anchor="middle">Recall@{cutoff}</text>')
        if index:
            x = left + method_width + index * cutoff_width
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#D9E1EA" stroke-width="1"/>')
    parts.append(f'<line x1="{left + method_width}" y1="{top}" x2="{left + method_width}" y2="{bottom}" stroke="#C7D0DC" stroke-width="1.5"/>')
    for method_index, method in enumerate(METHODS):
        row_top = top + header_height + method_index * row_height
        if method_index % 2 == 1:
            parts.append(f'<rect x="{left + 1}" y="{row_top}" width="{right - left - 2}" height="{row_height}" fill="#F8FAFC"/>')
        if method_index:
            parts.append(f'<line x1="{left}" y1="{row_top}" x2="{right}" y2="{row_top}" stroke="#D9E1EA" stroke-width="1"/>')
        parts.append(f'<circle cx="{left + 20}" cy="{row_top + row_height / 2:.1f}" r="7" fill="{METHOD_COLORS[method]}"/>')
        parts.append(f'<text class="tick" x="{left + 36}" y="{row_top + 25}">{escape(METHOD_LABELS[method])}</text>')
        for index, cutoff in enumerate(cutoffs):
            center = left + method_width + (index + 0.5) * cutoff_width
            parts.append(f'<text class="value" x="{center:.1f}" y="{row_top + 25}" text-anchor="middle">{values[(method, cutoff)].mean:.3f}</text>')


def corpus_chart(corpus: Corpus, results: dict[tuple[str, str], MetricResult]) -> str:
    title = f"Recuperação no corpus “{corpus.name}” ({corpus.pages} páginas)"
    subtitle = "50 consultas • segmentação recursiva • top-10 • pontos: média; barras de erro: IC95% bootstrap"
    parts = svg_start(title, subtitle)
    left, right, top, bottom = 125, 1515, 210, 575
    render_legend(parts, x=550, y=157)
    render_y_axis(parts, left=left, right=right, top=top, bottom=bottom)

    chart_width = right - left
    chart_height = bottom - top
    x_values = [left + index * chart_width / (len(METRICS) - 1) for index in range(len(METRICS))]
    for metric_index, (_metric, metric_label) in enumerate(METRICS):
        x = x_values[metric_index]
        parts.append(f'<text class="label" x="{x:.1f}" y="{bottom + 43}" text-anchor="middle">{metric_label}</text>')
    for method in METHODS:
        points: list[tuple[float, float, MetricResult]] = []
        for metric_index, (metric, _metric_label) in enumerate(METRICS):
            value = results[(method, metric)]
            x = x_values[metric_index]
            y = bottom - value.mean * chart_height
            points.append((x, y, value))
        parts.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _value in points)}" '
            f'fill="none" stroke="{METHOD_COLORS[method]}" stroke-width="5" stroke-linejoin="round"/>'
        )
        for x, y, value in points:
            high_y = bottom - value.high * chart_height
            low_y = bottom - value.low * chart_height
            parts.append(f'<line class="ci" x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{high_y:.1f}" x2="{x + 8:.1f}" y2="{high_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{low_y:.1f}" x2="{x + 8:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{METHOD_COLORS[method]}" stroke="white" stroke-width="3"/>')
    render_metric_table(parts, top=655, value_for=lambda method, metric: f"{results[(method, metric)].mean:.3f}")
    parts.append('<text class="caption" x="100" y="855">Leitura: valores maiores são melhores. MRR, MAP e nDCG também refletem a posição da evidência relevante no ranking.</text>')
    parts.append('</svg>')
    return "".join(parts)


def read_sciq_metrics(run_dir: Path) -> dict[tuple[str, str], MetricResult]:
    path = run_dir / "statistics" / "metrics_with_ci.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Métricas SciQ não encontradas: {path}")
    expected_metrics = {metric for metric, _label in METRICS}
    results: dict[tuple[str, str], MetricResult] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            method, metric = row["method"], row["metric"]
            if method in METHODS and metric in expected_metrics and int(row["k"]) == 10:
                results[(method, metric)] = MetricResult(
                    mean=float(row["mean"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                )
    expected = {(method, metric) for method in METHODS for metric in expected_metrics}
    missing = expected - set(results)
    if missing:
        formatted = ", ".join(f"{method}/{metric}" for method, metric in sorted(missing))
        raise ValueError(f"A rodada SciQ não possui as métricas esperadas: {formatted}")
    return results


def sciq_chart(results: dict[tuple[str, str], MetricResult]) -> str:
    parts = svg_start(
        "Recuperação no SciQ",
        "878 consultas de teste • 12.135 supports indexados • segmentação recursiva • top-10 • barras de erro: IC95% bootstrap",
    )
    left, right, top, bottom = 125, 1515, 210, 575
    render_legend(parts, x=550, y=157)
    render_y_axis(parts, left=left, right=right, top=top, bottom=bottom)
    chart_width, chart_height = right - left, bottom - top
    x_values = [left + index * chart_width / (len(METRICS) - 1) for index in range(len(METRICS))]
    for metric_index, (_metric, metric_label) in enumerate(METRICS):
        x = x_values[metric_index]
        parts.append(f'<text class="label" x="{x:.1f}" y="{bottom + 43}" text-anchor="middle">{metric_label}</text>')
    for method in METHODS:
        points: list[tuple[float, float, MetricResult]] = []
        for metric_index, (metric, _metric_label) in enumerate(METRICS):
            value = results[(method, metric)]
            x = x_values[metric_index]
            y = bottom - value.mean * chart_height
            points.append((x, y, value))
        parts.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _value in points)}" '
            f'fill="none" stroke="{METHOD_COLORS[method]}" stroke-width="5" stroke-linejoin="round"/>'
        )
        for x, y, value in points:
            high_y = bottom - value.high * chart_height
            low_y = bottom - value.low * chart_height
            parts.append(f'<line class="ci" x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{high_y:.1f}" x2="{x + 8:.1f}" y2="{high_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{low_y:.1f}" x2="{x + 8:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{METHOD_COLORS[method]}" stroke="white" stroke-width="3"/>')
    render_metric_table(parts, top=655, value_for=lambda method, metric: f"{results[(method, metric)].mean:.3f}")
    parts.append('<text class="caption" x="100" y="855">Há um único support relevante por consulta no SciQ; por isso MAP@10 e MRR@10 coincidem nesta avaliação.</text>')
    parts.append('</svg>')
    return "".join(parts)


def read_sciq_recall_by_k(run_dir: Path) -> dict[tuple[str, int], MetricResult]:
    path = run_dir / "statistics" / "metrics_with_ci.csv"
    cutoffs = (1, 3, 5, 10)
    results: dict[tuple[str, int], MetricResult] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            method = row["method"]
            cutoff = int(row["k"])
            if method in METHODS and cutoff in cutoffs and row["metric"] == "recall_at_k":
                results[(method, cutoff)] = MetricResult(
                    mean=float(row["mean"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                )
    expected = {(method, cutoff) for method in METHODS for cutoff in cutoffs}
    if expected - set(results):
        raise ValueError("A rodada SciQ não possui Recall@1, @3, @5 e @10 completo.")
    return results


def sciq_recall_curve_chart(values: dict[tuple[str, int], MetricResult]) -> str:
    cutoffs = (1, 3, 5, 10)
    parts = svg_start(
        "SciQ: evidência encontrada ao ampliar o top-k",
        "878 consultas de teste • Recall@k (equivale a Hit@k neste conjunto) • barras de erro: IC95% bootstrap",
    )
    left, right, top, bottom = 125, 1515, 210, 575
    render_legend(parts, x=550, y=157)
    render_y_axis(parts, left=left, right=right, top=top, bottom=bottom)
    chart_width, chart_height = right - left, bottom - top
    x_values = [left + index * chart_width / (len(cutoffs) - 1) for index in range(len(cutoffs))]
    for index, cutoff in enumerate(cutoffs):
        parts.append(f'<text class="label" x="{x_values[index]:.1f}" y="{bottom + 43}" text-anchor="middle">top-{cutoff}</text>')
    for method in METHODS:
        points: list[tuple[float, float, MetricResult]] = []
        for index, cutoff in enumerate(cutoffs):
            value = values[(method, cutoff)]
            points.append((x_values[index], bottom - value.mean * chart_height, value))
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _value in points)}" fill="none" stroke="{METHOD_COLORS[method]}" stroke-width="5" stroke-linejoin="round"/>')
        for x, y, value in points:
            high_y, low_y = bottom - value.high * chart_height, bottom - value.low * chart_height
            parts.append(f'<line class="ci" x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{high_y:.1f}" x2="{x + 8:.1f}" y2="{high_y:.1f}"/>')
            parts.append(f'<line class="ci" x1="{x - 8:.1f}" y1="{low_y:.1f}" x2="{x + 8:.1f}" y2="{low_y:.1f}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{METHOD_COLORS[method]}" stroke="white" stroke-width="3"/>')
    render_cutoff_table(parts, top=655, values=values, cutoffs=cutoffs)
    parts.append('<text class="caption" x="100" y="855">O ganho do top-1 para o top-10 mostra a parcela de consultas que exige mais de um trecho recuperado para conter a evidência.</text>')
    parts.append('</svg>')
    return "".join(parts)


def first_relevant_rank_counts(run_dir: Path) -> dict[str, list[int]]:
    counts = {method: [0] * 11 for method in METHODS}
    for method in METHODS:
        path = run_dir / "recursive_text" / "retrieval" / "results.jsonl"
        # A single file contains all methods in the PDF-IR runner.
        if not path.is_file():
            raise FileNotFoundError(f"Resultados de PDF-IR não encontrados: {path}")
        break
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            method = row.get("method")
            if method not in counts:
                continue
            relevant = {str(chunk_id) for chunk_id in row.get("relevant_chunk_ids", [])}
            ranking = [str(chunk_id) for chunk_id in row.get("retrieved_chunk_ids", [])]
            rank = next((index for index, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant), None)
            counts[method][rank - 1 if rank is not None and rank <= 10 else 10] += 1
    if not all(sum(values) for values in counts.values()):
        raise ValueError("Não foi possível obter a distribuição de posições para todos os métodos.")
    return counts


def first_relevant_rank_chart(counts: dict[str, list[int]]) -> str:
    maximum = max(max(values) for values in counts.values())
    axis_max = max(5, math.ceil(maximum / 5) * 5)
    parts = svg_start(
        "Livro de RI: posição da primeira evidência relevante",
        "50 consultas • segmentação recursiva • top-10 • a última posição reúne consultas sem evidência relevante recuperada",
    )
    left, right, top, bottom = 110, 1515, 210, 690
    render_legend(parts, x=550, y=157)
    plot_height = bottom - top
    for tick in range(6):
        value = axis_max * tick / 5
        y = bottom - value / axis_max * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 16}" y="{y + 6:.1f}" text-anchor="end">{value:.0f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    labels = [str(rank) for rank in range(1, 11)] + ["não\nencontrada"]
    x_values = [left + index * (right - left) / (len(labels) - 1) for index in range(len(labels))]
    for index, label in enumerate(labels):
        x = x_values[index]
        if "\n" in label:
            first, second = label.split("\n")
            parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 32}" text-anchor="middle"><tspan x="{x:.1f}" dy="0">{first}</tspan><tspan x="{x:.1f}" dy="18">{second}</tspan></text>')
        else:
            parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 32}" text-anchor="middle">{label}</text>')
    for method in METHODS:
        points = [(x_values[index], bottom - value / axis_max * plot_height) for index, value in enumerate(counts[method])]
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{METHOD_COLORS[method]}" stroke-width="4" stroke-linejoin="round"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{METHOD_COLORS[method]}" stroke="white" stroke-width="2"/>')
    parts.append('<text class="label" x="110" y="188">Quantidade de consultas</text>')
    parts.append('<text class="caption" x="100" y="845">Quanto mais à esquerda estiver a evidência, menor é a quantidade de contexto necessária para a resposta do tutor.</text>')
    parts.append('</svg>')
    return "".join(parts)


def heat_color(value: float) -> str:
    """Map a score in [0, 1] to a readable sequential teal scale."""
    start = (246, 248, 250)
    middle = (133, 202, 180)
    end = (10, 112, 103)
    if value <= 0.5:
        fraction = max(0.0, value) / 0.5
        colors = tuple(round(start[index] + (middle[index] - start[index]) * fraction) for index in range(3))
    else:
        fraction = min(1.0, (value - 0.5) / 0.5)
        colors = tuple(round(middle[index] + (end[index] - middle[index]) * fraction) for index in range(3))
    return "#" + "".join(f"{component:02X}" for component in colors)


def general_heatmap_chart(
    sciq: dict[tuple[str, str], MetricResult],
    ato_de_ler: dict[tuple[str, str], MetricResult],
    livro_ri: dict[tuple[str, str], MetricResult],
) -> str:
    parts = svg_start(
        "Visão geral: desempenho de recuperação nos três conjuntos",
        "Mesma escala de 0 a 1 • célula mais escura = melhor desempenho • valores são médias no top-10",
    )
    panels = (
        ("SciQ", "878 consultas", sciq),
        ("Ato de Ler", "49 páginas • 50 consultas", ato_de_ler),
        ("Livro de RI", "581 páginas • 50 consultas", livro_ri),
    )
    panel_lefts = (90, 560, 1030)
    panel_width, cell_width, cell_height = 430, 84, 86
    label_width, top = 94, 260
    for panel_index, (title, subtitle, values) in enumerate(panels):
        left = panel_lefts[panel_index]
        grid_left = left + label_width
        parts.append(f'<text class="label" x="{left}" y="{top - 62}">{escape(title)}</text>')
        parts.append(f'<text class="subtitle" x="{left}" y="{top - 38}">{escape(subtitle)}</text>')
        for metric_index, (_metric, label) in enumerate(METRICS):
            x = grid_left + metric_index * cell_width + cell_width / 2
            parts.append(f'<text class="tick" x="{x:.1f}" y="{top - 12}" text-anchor="middle">{label.replace("@10", "")}</text>')
        for method_index, method in enumerate(METHODS):
            y = top + method_index * cell_height
            parts.append(f'<rect x="{left}" y="{y}" width="{label_width - 10}" height="{cell_height - 4}" fill="{METHOD_COLORS[method]}" rx="4"/>')
            parts.append(f'<text x="{left + (label_width - 10) / 2:.1f}" y="{y + 51}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="17" font-weight="700" fill="white">{escape(HEATMAP_METHOD_LABELS[method])}</text>')
            for metric_index, (metric, _label) in enumerate(METRICS):
                value = values[(method, metric)].mean
                x = grid_left + metric_index * cell_width
                text_color = "#FFFFFF" if value >= 0.76 else "#172033"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_width - 4}" height="{cell_height - 4}" fill="{heat_color(value)}" rx="4"/>')
                parts.append(f'<text x="{x + (cell_width - 4) / 2:.1f}" y="{y + 51}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="19" font-weight="700" fill="{text_color}">{value:.3f}</text>')
        parts.append(f'<rect x="{left}" y="{top - 88}" width="{panel_width}" height="{cell_height * len(METHODS) + 102}" fill="none" stroke="#D7DEE8" stroke-width="1.5" rx="8"/>')

    legend_left, legend_top, legend_width = 575, 650, 450
    for step in range(100):
        value = step / 99
        x = legend_left + step * legend_width / 100
        parts.append(f'<rect x="{x:.1f}" y="{legend_top}" width="{legend_width / 100 + 1:.1f}" height="20" fill="{heat_color(value)}"/>')
    parts.append(f'<text class="tick" x="{legend_left - 12}" y="{legend_top + 16}" text-anchor="end">0,0</text>')
    parts.append(f'<text class="tick" x="{legend_left + legend_width + 12}" y="{legend_top + 16}">1,0</text>')
    parts.append(f'<text class="caption" x="100" y="740">Leitura: o híbrido lidera no SciQ e no corpus de 49 páginas; no livro de RI, denso e híbrido têm resultados muito próximos e superam o BM25.</text>')
    parts.append(f'<text class="caption" x="100" y="775">No SciQ há uma única evidência relevante por pergunta, por isso MRR@10 e MAP@10 têm o mesmo valor.</text>')
    parts.append('</svg>')
    return "".join(parts)


def comparison_chart(
    short: Corpus,
    long: Corpus,
    short_results: dict[tuple[str, str], MetricResult],
    long_results: dict[tuple[str, str], MetricResult],
) -> str:
    parts = svg_start(
        "Efeito do tamanho do corpus na recuperação",
        f"Mesmo protocolo: 50 consultas, segmentação recursiva e top-10 • {short.pages} vs. {long.pages} páginas",
    )
    left, right, top, bottom = 175, 1470, 210, 555
    render_y_axis(parts, left=left, right=right, top=top, bottom=bottom)
    chart_width, chart_height = right - left, bottom - top
    panel_width = chart_width / len(METRICS)
    x_padding = 44

    for panel_index, (metric, metric_label) in enumerate(METRICS):
        panel_left = left + panel_index * panel_width
        short_x, long_x = panel_left + x_padding, panel_left + panel_width - x_padding
        if panel_index:
            parts.append(f'<line x1="{panel_left}" y1="{top}" x2="{panel_left}" y2="{bottom + 52}" stroke="#C7D0DC" stroke-width="1.5"/>')
        parts.append(f'<text class="label" x="{panel_left + panel_width / 2:.1f}" y="{top - 26}" text-anchor="middle">{metric_label}</text>')
        for method in METHODS:
            short_value = short_results[(method, metric)].mean
            long_value = long_results[(method, metric)].mean
            short_y = bottom - short_value * chart_height
            long_y = bottom - long_value * chart_height
            color = METHOD_COLORS[method]
            parts.append(f'<line x1="{short_x:.1f}" y1="{short_y:.1f}" x2="{long_x:.1f}" y2="{long_y:.1f}" stroke="{color}" stroke-width="5" opacity="0.74"/>')
            parts.append(f'<circle cx="{short_x:.1f}" cy="{short_y:.1f}" r="10" fill="{color}"/>')
            parts.append(f'<circle cx="{long_x:.1f}" cy="{long_y:.1f}" r="10" fill="{color}"/>')
        parts.append(f'<text class="tick" x="{short_x:.1f}" y="{bottom + 35}" text-anchor="middle">{short.pages} páginas</text>')
        parts.append(f'<text class="tick" x="{long_x:.1f}" y="{bottom + 35}" text-anchor="middle">{long.pages} páginas</text>')

    render_metric_table(
        parts,
        top=640,
        first_header=f"Método ({short.pages}→{long.pages} p.)",
        value_for=lambda method, metric: f"{short_results[(method, metric)].mean:.3f} → {long_results[(method, metric)].mean:.3f}",
    )
    render_legend(parts, x=570, y=810)
    parts.append('<text class="caption" x="100" y="865">Cada linha conecta o mesmo recuperador entre os dois corpora; inclinação descendente indica queda de desempenho ao ampliar o espaço de busca.</text>')
    parts.append('</svg>')
    return "".join(parts)


def write_chart(path: Path, content: str) -> None:
    path.write_text(content + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Gera três gráficos SVG para os slides de PDF-IR.")
    parser.add_argument("--output-dir", type=Path, default=root / "research" / "figures" / "slides_pdf_ir")
    parser.add_argument(
        "--ato-de-ler-run",
        type=Path,
        default=root / "benchmark" / "runs" / "novas" / "pdf-ir" / "pdf-principal-02",
    )
    parser.add_argument(
        "--livro-ri-run",
        type=Path,
        default=root / "benchmark" / "runs" / "novas" / "pdf-ir" / "iir-principal-03",
    )
    parser.add_argument(
        "--sciq-run",
        type=Path,
        default=root / "benchmark" / "runs" / "novas" / "sciq" / "sciq-atualizada-01",
    )
    args = parser.parse_args()

    ato_de_ler = Corpus("Ato de Ler", 49, args.ato_de_ler_run.resolve())
    livro_ri = Corpus("Livro de RI", 581, args.livro_ri_run.resolve())
    ato_metrics = read_metrics(ato_de_ler)
    livro_metrics = read_metrics(livro_ri)
    sciq_metrics = read_sciq_metrics(args.sciq_run.resolve())
    sciq_recall = read_sciq_recall_by_k(args.sciq_run.resolve())
    first_rank_counts = first_relevant_rank_counts(livro_ri.run_dir)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_chart(output_dir / "01_ato_de_ler_49_paginas.svg", corpus_chart(ato_de_ler, ato_metrics))
    write_chart(output_dir / "02_livro_ri_581_paginas.svg", corpus_chart(livro_ri, livro_metrics))
    write_chart(
        output_dir / "03_comparacao_49_vs_581_paginas.svg",
        comparison_chart(ato_de_ler, livro_ri, ato_metrics, livro_metrics),
    )
    write_chart(output_dir / "04_sciq_878_consultas.svg", sciq_chart(sciq_metrics))
    write_chart(output_dir / "05_sciq_recall_por_k.svg", sciq_recall_curve_chart(sciq_recall))
    write_chart(output_dir / "06_livro_ri_primeira_evidencia.svg", first_relevant_rank_chart(first_rank_counts))
    write_chart(
        output_dir / "07_heatmap_geral_tres_datasets.svg",
        general_heatmap_chart(sciq_metrics, ato_metrics, livro_metrics),
    )


if __name__ == "__main__":
    main()
