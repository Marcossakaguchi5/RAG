#!/usr/bin/env python3
"""Generate dependency-free tables and SVG plots for a PDF-IR experiment run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any


METRICS = [
    ("mrr", "MRR"),
    ("ndcg_at_k", "nDCG@k"),
    ("recall_at_k", "Recall@k"),
]
ALL_METRICS = [
    ("hit_rate_at_k", "Hit rate"),
    ("precision_at_k", "Precision"),
    ("recall_at_k", "Recall"),
    ("map", "MAP"),
    ("ndcg_at_k", "nDCG"),
    ("mrr", "MRR"),
]
CURVE_METRICS = [
    ("hit_rate_at_k", "Hit rate"),
    ("recall_at_k", "Recall"),
    ("ndcg_at_k", "nDCG"),
    ("mrr", "MRR"),
]
PAIRED_METRICS = ["hit_rate_at_k", "mrr", "ndcg_at_k", "recall_at_k"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} nao contem um objeto JSON.")
            rows.append(value)
    return rows


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    result_paths = sorted(run_dir.glob("*/retrieval/results.jsonl"))
    if not result_paths:
        raise ValueError(f"Nenhum */retrieval/results.jsonl encontrado em {run_dir}.")
    for path in result_paths:
        strategy = path.parents[1].name
        for row in read_jsonl(path):
            combined.append({"chunking_strategy": strategy, **row})
    errors = [row for row in combined if row.get("status") != "ok"]
    if errors:
        raise ValueError(
            f"A rodada contem {len(errors)} linha(s) com erro; corrija antes de gerar graficos."
        )
    return combined


def bootstrap_mean_ci(
    values: list[float],
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Nao ha valores para calcular intervalo de confianca.")
    point = mean(values)
    if len(values) == 1:
        return point, point, point
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(repetitions)
    )
    low = estimates[max(0, int(0.025 * repetitions))]
    high = estimates[min(repetitions - 1, int(0.975 * repetitions))]
    return point, low, high


def summarize(
    rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["chunking_strategy"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_index, ((strategy, method), group_rows) in enumerate(sorted(groups.items())):
        for metric_index, (metric, _label) in enumerate(METRICS):
            values = [
                float(row["metrics"][metric])
                for row in group_rows
                if row.get("metrics", {}).get(metric) is not None
            ]
            point, low, high = bootstrap_mean_ci(
                values,
                repetitions=repetitions,
                seed=seed + group_index * 100 + metric_index,
            )
            summaries.append(
                {
                    "chunking_strategy": strategy,
                    "method": method,
                    "metric": metric,
                    "queries": len(values),
                    "mean": round(point, 6),
                    "ci95_low": round(low, 6),
                    "ci95_high": round(high, 6),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": seed,
                }
            )
    return summaries


def ranked_metrics_at_k(row: dict[str, Any], k: int) -> dict[str, float]:
    """Recalculate the standard retrieval metrics from a persisted ranking."""
    ranking = [str(item) for item in row.get("retrieved_chunk_ids", [])[:k]]
    grades = {
        str(chunk_id): float(grade)
        for chunk_id, grade in dict(row.get("relevance_by_chunk", {})).items()
    }
    if not grades:
        grades = {str(chunk_id): 1.0 for chunk_id in row.get("relevant_chunk_ids", [])}
    relevant = set(grades)
    hits = [chunk_id in relevant for chunk_id in ranking]
    hit_count = sum(hits)
    accumulated_precision = 0.0
    reciprocal_rank = 0.0
    seen_relevant = 0
    dcg = 0.0
    for position, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            seen_relevant += 1
            accumulated_precision += seen_relevant / position
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1 / position
        grade = grades.get(chunk_id, 0.0)
        dcg += (2**grade - 1) / math.log2(position + 1)
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(position + 1)
        for position, grade in enumerate(sorted(grades.values(), reverse=True)[:k], start=1)
    )
    return {
        "hit_rate_at_k": 1.0 if hit_count else 0.0,
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant) if relevant else 0.0,
        "map": accumulated_precision / len(relevant) if relevant else 0.0,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "mrr": reciprocal_rank,
    }


def k_values_for_rows(rows: list[dict[str, Any]]) -> list[int]:
    maximum = max(len(row.get("retrieved_chunk_ids", [])) for row in rows)
    values = [k for k in (1, 3, 5, 10) if k <= maximum]
    if maximum and maximum not in values:
        values.append(maximum)
    return values


def summarize_by_k(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    by_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(
            (str(row["chunking_strategy"]), str(row["method"])), []
        ).append(row)
    for condition_index, ((strategy, method), condition_rows) in enumerate(sorted(by_condition.items())):
        for k_index, k in enumerate(k_values_for_rows(condition_rows)):
            recalculated = [ranked_metrics_at_k(row, k) for row in condition_rows]
            for metric_index, (metric, _label) in enumerate(ALL_METRICS):
                point, low, high = bootstrap_mean_ci(
                    [item[metric] for item in recalculated],
                    repetitions=repetitions,
                    seed=seed + 30000 + condition_index * 1000 + k_index * 100 + metric_index,
                )
                summaries.append(
                    {
                        "chunking_strategy": strategy,
                        "method": method,
                        "k": k,
                        "metric": metric,
                        "queries": len(condition_rows),
                        "mean": round(point, 6),
                        "ci95_low": round(low, 6),
                        "ci95_high": round(high, 6),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": seed,
                    }
                )
    return summaries


def condition_label(strategy: str, method: str) -> str:
    first, second = series_label_lines(strategy, method)
    return f"{first} / {second}"


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def add_holm_adjustment(
    rows: list[dict[str, Any]],
    *,
    group_field: str = "chunking_strategy",
) -> None:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("mcnemar_exact_p") is not None:
            by_strategy.setdefault(str(row[group_field]), []).append(row)
    for strategy_rows in by_strategy.values():
        ordered = sorted(strategy_rows, key=lambda row: float(row["mcnemar_exact_p"]))
        previous = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * float(row["mcnemar_exact_p"]))
            adjusted = max(previous, adjusted)
            row["mcnemar_holm_p"] = round(adjusted, 8)
            previous = adjusted
    for row in rows:
        row.setdefault("mcnemar_holm_p", None)


def paired_differences(
    rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_condition: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["chunking_strategy"]), str(row["method"]))
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("Resultado sem case_id; comparacao pareada impossivel.")
        by_condition.setdefault(key, {})[case_id] = row

    output: list[dict[str, Any]] = []
    strategies = sorted({strategy for strategy, _method in by_condition})
    comparison_index = 0
    for strategy in strategies:
        methods = sorted(method for item_strategy, method in by_condition if item_strategy == strategy)
        for left_method, right_method in combinations(methods, 2):
            left = by_condition[(strategy, left_method)]
            right = by_condition[(strategy, right_method)]
            if set(left) != set(right):
                raise ValueError(
                    f"Condicoes nao pareadas em {strategy}: {left_method} e {right_method}."
                )
            case_ids = sorted(left)
            for metric_index, metric in enumerate(PAIRED_METRICS):
                differences = [
                    float(left[case_id]["metrics"][metric])
                    - float(right[case_id]["metrics"][metric])
                    for case_id in case_ids
                ]
                point, low, high = bootstrap_mean_ci(
                    differences,
                    repetitions=repetitions,
                    seed=seed + 10000 + comparison_index * 100 + metric_index,
                )
                wins = sum(value > 0 for value in differences)
                losses = sum(value < 0 for value in differences)
                ties = len(differences) - wins - losses
                row = {
                    "chunking_strategy": strategy,
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "queries": len(differences),
                    "mean_difference_left_minus_right": round(point, 6),
                    "ci95_low": round(low, 6),
                    "ci95_high": round(high, 6),
                    "left_wins": wins,
                    "ties": ties,
                    "right_wins": losses,
                    "mcnemar_exact_p": (
                        round(exact_mcnemar_p(wins, losses), 8)
                        if metric == "hit_rate_at_k"
                        else None
                    ),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": seed,
                }
                output.append(row)
            comparison_index += 1
    add_holm_adjustment(output)
    return output


def paired_chunking_differences(
    rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare chunking strategies while keeping the retriever fixed."""

    by_condition: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["chunking_strategy"]))
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("Resultado sem case_id; comparacao pareada impossivel.")
        by_condition.setdefault(key, {})[case_id] = row

    output: list[dict[str, Any]] = []
    methods = sorted({method for method, _strategy in by_condition})
    comparison_index = 0
    for method in methods:
        strategies = sorted(
            strategy
            for item_method, strategy in by_condition
            if item_method == method
        )
        for left_strategy, right_strategy in combinations(strategies, 2):
            left = by_condition[(method, left_strategy)]
            right = by_condition[(method, right_strategy)]
            if set(left) != set(right):
                raise ValueError(
                    f"Chunkings nao pareados para {method}: "
                    f"{left_strategy} e {right_strategy}."
                )
            case_ids = sorted(left)
            for metric_index, metric in enumerate(PAIRED_METRICS):
                differences = [
                    float(left[case_id]["metrics"][metric])
                    - float(right[case_id]["metrics"][metric])
                    for case_id in case_ids
                ]
                point, low, high = bootstrap_mean_ci(
                    differences,
                    repetitions=repetitions,
                    seed=seed + 20000 + comparison_index * 100 + metric_index,
                )
                wins = sum(value > 0 for value in differences)
                losses = sum(value < 0 for value in differences)
                output.append(
                    {
                        "method": method,
                        "left_chunking_strategy": left_strategy,
                        "right_chunking_strategy": right_strategy,
                        "metric": metric,
                        "queries": len(differences),
                        "mean_difference_left_minus_right": round(point, 6),
                        "ci95_low": round(low, 6),
                        "ci95_high": round(high, 6),
                        "left_wins": wins,
                        "ties": len(differences) - wins - losses,
                        "right_wins": losses,
                        "mcnemar_exact_p": (
                            round(exact_mcnemar_p(wins, losses), 8)
                            if metric == "hit_rate_at_k"
                            else None
                        ),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": seed,
                    }
                )
            comparison_index += 1
    add_holm_adjustment(output, group_field="method")
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def series_label_lines(strategy: str, method: str) -> tuple[str, str]:
    """Return compact, readable labels for a strategy/method condition."""
    strategy_label = strategy.replace("_", " ")
    if strategy_label.startswith("docling "):
        strategy_label = "Docling " + strategy_label.removeprefix("docling ")
    else:
        strategy_label = strategy_label.title()
    return strategy_label, method.upper()


def svg_chart(summaries: list[dict[str, Any]], metric: str, label: str) -> str:
    series = sorted(
        {(row["chunking_strategy"], row["method"]) for row in summaries}
    )
    colors = ["#31688e", "#7e4fa3", "#24907b", "#d17c28", "#b84a62", "#5f6b78"]
    width = max(960, 230 * len(series) + 120)
    height = 430
    left, right = 85, 35
    plot_width = width - left - right
    bar_slot = plot_width / max(1, len(series))
    bar_width = min(110, bar_slot * 0.54)
    by_key = {
        (row["metric"], row["chunking_strategy"], row["method"]): row
        for row in summaries
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#253247}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px}.label{font-size:15px;font-weight:700}.tick{font-size:13px}.grid{stroke:#d7dee8;stroke-width:1}.axis{stroke:#253247;stroke-width:1.5}.ci{stroke:#1f2937;stroke-width:2}</style>',
        f'<text class="title" x="{left}" y="42">PDF-IR: {html.escape(label)}</text>',
        f'<text class="subtitle" x="{left}" y="65">Média por consulta e intervalo de confiança de 95% por bootstrap</text>',
    ]
    plot_top, plot_bottom = 88, 305
    plot_height = plot_bottom - plot_top
    for tick in range(6):
        value = tick / 5
        y = plot_bottom - value * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{plot_bottom}" x2="{width-right}" y2="{plot_bottom}"/>')
    for index, (strategy, method) in enumerate(series):
        row = by_key[(metric, strategy, method)]
        x = left + index * bar_slot + (bar_slot - bar_width) / 2
        mean_value = float(row["mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        bar_height = max(0.0, min(1.0, mean_value)) * plot_height
        y = plot_bottom - bar_height
        center = x + bar_width / 2
        high_y = plot_bottom - max(0.0, min(1.0, high)) * plot_height
        low_y = plot_bottom - max(0.0, min(1.0, low)) * plot_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<line class="ci" x1="{center:.2f}" y1="{high_y:.2f}" x2="{center:.2f}" y2="{low_y:.2f}"/>')
        parts.append(f'<line class="ci" x1="{center-7:.2f}" y1="{high_y:.2f}" x2="{center+7:.2f}" y2="{high_y:.2f}"/>')
        parts.append(f'<line class="ci" x1="{center-7:.2f}" y1="{low_y:.2f}" x2="{center+7:.2f}" y2="{low_y:.2f}"/>')
        label_y = max(plot_top + 16, high_y - 10)
        parts.append(f'<text class="label" x="{center:.2f}" y="{label_y:.2f}" text-anchor="middle">{mean_value:.3f}</text>')
        strategy_line, method_line = series_label_lines(strategy, method)
        parts.append(f'<text class="tick" x="{center:.2f}" y="{plot_bottom + 30}" text-anchor="middle"><tspan x="{center:.2f}" dy="0">{html.escape(strategy_line)}</tspan><tspan x="{center:.2f}" dy="18">{html.escape(method_line)}</tspan></text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_shell(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#253247}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px}.label{font-size:13px}.tick{font-size:12px}.legend{font-size:12px}.grid{stroke:#d7dee8;stroke-width:1}.axis{stroke:#253247;stroke-width:1.5}</style>',
        f'<text class="title" x="70" y="40">{html.escape(title)}</text>',
    ]
    if subtitle:
        parts.append(f'<text class="subtitle" x="70" y="62">{html.escape(subtitle)}</text>')
    return parts


def _render_condition_legend(
    parts: list[str], conditions: list[tuple[str, str]], colors: list[str], x: float, y: float
) -> None:
    for index, (strategy, method) in enumerate(conditions):
        column = index % 3
        row = index // 3
        item_x = x + column * 285
        item_y = y + row * 20
        parts.append(f'<rect x="{item_x}" y="{item_y - 11}" width="13" height="13" fill="{colors[index % len(colors)]}" rx="2"/>')
        parts.append(f'<text class="legend" x="{item_x + 19}" y="{item_y}">{html.escape(condition_label(strategy, method))}</text>')


def metrics_at_k_chart(by_k: list[dict[str, Any]], k: int) -> str:
    rows = [row for row in by_k if int(row["k"]) == k]
    conditions = sorted({(str(row["chunking_strategy"]), str(row["method"])) for row in rows})
    width, height = 1360, 650
    left, right, top, bottom = 80, 35, 125, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    group_width = plot_width / len(ALL_METRICS)
    bar_width = min(34, group_width / max(1, len(conditions) + 1))
    colors = ["#31688e", "#7e4fa3", "#24907b", "#d17c28", "#b84a62", "#5f6b78"]
    values = {(row["metric"], row["chunking_strategy"], row["method"]): row for row in rows}
    parts = _svg_shell(width, height, f"Métricas de recuperação @ {k}", "Média por consulta; barras de erro mostram IC95% bootstrap")
    _render_condition_legend(parts, conditions, colors, left, 91)
    for tick in range(6):
        value = tick / 5
        y = top + plot_height - value * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width-right}" y2="{top + plot_height}"/>')
    for metric_index, (metric, label) in enumerate(ALL_METRICS):
        center = left + group_width * metric_index + group_width / 2
        start = center - (len(conditions) * bar_width + (len(conditions) - 1) * 8) / 2
        for condition_index, (strategy, method) in enumerate(conditions):
            row = values[(metric, strategy, method)]
            point, low, high = float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"])
            x = start + condition_index * (bar_width + 8)
            y = top + plot_height - point * plot_height
            center_x = x + bar_width / 2
            high_y = top + plot_height - high * plot_height
            low_y = top + plot_height - low * plot_height
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{top + plot_height - y:.2f}" fill="{colors[condition_index % len(colors)]}"/>')
            parts.append(f'<line x1="{center_x:.2f}" y1="{high_y:.2f}" x2="{center_x:.2f}" y2="{low_y:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{center_x-4:.2f}" y1="{high_y:.2f}" x2="{center_x+4:.2f}" y2="{high_y:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{center_x-4:.2f}" y1="{low_y:.2f}" x2="{center_x+4:.2f}" y2="{low_y:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
        parts.append(f'<text class="label" x="{center:.2f}" y="{top + plot_height + 28}" text-anchor="middle">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def metric_curves_chart(by_k: list[dict[str, Any]]) -> str:
    conditions = sorted({(str(row["chunking_strategy"]), str(row["method"])) for row in by_k})
    k_values = sorted({int(row["k"]) for row in by_k})
    values = {(row["metric"], row["chunking_strategy"], row["method"], int(row["k"])): row for row in by_k}
    width, height = 1280, 850
    left, top, cell_width, cell_height, gap_x, gap_y = 85, 145, 505, 245, 105, 105
    colors = ["#31688e", "#7e4fa3", "#24907b", "#d17c28", "#b84a62", "#5f6b78"]
    parts = _svg_shell(width, height, "Curvas de métricas por k", "Os pontos são calculados a partir dos rankings top-10 persistidos")
    _render_condition_legend(parts, conditions, colors, left, 93)
    for metric_index, (metric, label) in enumerate(CURVE_METRICS):
        column, row_index = metric_index % 2, metric_index // 2
        x0 = left + column * (cell_width + gap_x)
        y0 = top + row_index * (cell_height + gap_y)
        parts.append(f'<text class="label" x="{x0 + cell_width / 2:.2f}" y="{y0 - 16}" text-anchor="middle">{html.escape(label)}</text>')
        for tick in range(6):
            value = tick / 5
            y = y0 + cell_height - value * cell_height
            parts.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + cell_width}" y2="{y:.2f}"/>')
            parts.append(f'<text class="tick" x="{x0-8}" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + cell_height}"/>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0 + cell_height}" x2="{x0 + cell_width}" y2="{y0 + cell_height}"/>')
        for k_index, k in enumerate(k_values):
            x = x0 + k_index * cell_width / max(1, len(k_values) - 1)
            parts.append(f'<text class="tick" x="{x:.2f}" y="{y0 + cell_height + 22}" text-anchor="middle">{k}</text>')
        for condition_index, (strategy, method) in enumerate(conditions):
            points = []
            for k_index, k in enumerate(k_values):
                row = values[(metric, strategy, method, k)]
                x = x0 + k_index * cell_width / max(1, len(k_values) - 1)
                y = y0 + cell_height - float(row["mean"]) * cell_height
                points.append((x, y))
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            color = colors[condition_index % len(colors)]
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def first_relevant_rank_chart(rows: list[dict[str, Any]]) -> str:
    conditions = sorted({(str(row["chunking_strategy"]), str(row["method"])) for row in rows})
    maximum = max(len(row.get("retrieved_chunk_ids", [])) for row in rows)
    labels = [str(rank) for rank in range(1, maximum + 1)] + ["miss"]
    counts = {condition: {label: 0 for label in labels} for condition in conditions}
    for row in rows:
        condition = (str(row["chunking_strategy"]), str(row["method"]))
        relevant = {str(chunk_id) for chunk_id in row.get("relevant_chunk_ids", [])}
        rank = next(
            (index for index, chunk_id in enumerate(row.get("retrieved_chunk_ids", []), start=1) if str(chunk_id) in relevant),
            None,
        )
        counts[condition][str(rank) if rank is not None else "miss"] += 1
    width, height = 1280, 640
    left, right, top, bottom = 80, 35, 125, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    group_width = plot_width / len(labels)
    bar_width = min(26, group_width / max(1, len(conditions) + 1))
    colors = ["#31688e", "#7e4fa3", "#24907b", "#d17c28", "#b84a62", "#5f6b78"]
    maximum_count = max(value for condition in conditions for value in counts[condition].values()) or 1
    parts = _svg_shell(width, height, "Posição da primeira evidência relevante", "Quantidade de consultas cuja primeira evidência relevante aparece em cada posição")
    _render_condition_legend(parts, conditions, colors, left, 91)
    for tick in range(6):
        value = maximum_count * tick / 5
        y = top + plot_height - value / maximum_count * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.0f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width-right}" y2="{top + plot_height}"/>')
    for label_index, label in enumerate(labels):
        center = left + label_index * group_width + group_width / 2
        start = center - (len(conditions) * bar_width + (len(conditions) - 1) * 7) / 2
        for condition_index, condition in enumerate(conditions):
            value = counts[condition][label]
            x = start + condition_index * (bar_width + 7)
            y = top + plot_height - value / maximum_count * plot_height
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{top + plot_height - y:.2f}" fill="{colors[condition_index % len(colors)]}"/>')
            if value:
                parts.append(f'<text class="tick" x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle">{value}</text>')
        display = "não encontrado" if label == "miss" else f"posição {label}"
        parts.append(f'<text class="tick" x="{center:.2f}" y="{top + plot_height + 27}" text-anchor="middle">{display}</text>')
    parts.append("</svg>")
    return "".join(parts)


def latency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("latency_ms") is not None:
            groups.setdefault((str(row["chunking_strategy"]), str(row["method"])), []).append(float(row["latency_ms"]))
    summary = []
    for (strategy, method), values in sorted(groups.items()):
        sorted_values = sorted(values)
        summary.append(
            {
                "chunking_strategy": strategy,
                "method": method,
                "queries": len(values),
                "mean_ms": round(mean(values), 3),
                "median_ms": round(median(values), 3),
                "p95_ms": round(sorted_values[max(0, math.ceil(0.95 * len(values)) - 1)], 3),
            }
        )
    return summary


def latency_chart(summary: list[dict[str, Any]]) -> str:
    conditions = [(str(row["chunking_strategy"]), str(row["method"])) for row in summary]
    width, height = max(900, 230 * len(conditions) + 120), 570
    left, right, top, bottom = 85, 35, 90, 120
    plot_width, plot_height = width - left - right, height - top - bottom
    max_value = max(float(row["p95_ms"]) for row in summary) * 1.15
    group_width = plot_width / max(1, len(conditions))
    bar_width = min(55, group_width * 0.24)
    colors = ["#31688e", "#7e4fa3"]
    parts = _svg_shell(width, height, "Latência de recuperação", "Mediana e percentil 95 por consulta; menor é melhor")
    parts.extend([
        '<rect x="85" y="69" width="13" height="13" fill="#31688e" rx="2"/>',
        '<text class="legend" x="104" y="80">Mediana</text>',
        '<rect x="185" y="69" width="13" height="13" fill="#7e4fa3" rx="2"/>',
        '<text class="legend" x="204" y="80">P95</text>',
    ])
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_height - value / max_value * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.0f} ms</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width-right}" y2="{top + plot_height}"/>')
    for index, row in enumerate(summary):
        center = left + (index + 0.5) * group_width
        for metric_index, field in enumerate(("median_ms", "p95_ms")):
            value = float(row[field])
            x = center + (metric_index - 0.5) * (bar_width + 10)
            y = top + plot_height - value / max_value * plot_height
            parts.append(f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{top + plot_height - y:.2f}" fill="{colors[metric_index]}"/>')
            parts.append(f'<text class="tick" x="{x:.2f}" y="{y-6:.2f}" text-anchor="middle">{value:.0f}</text>')
        line1, line2 = series_label_lines(str(row["chunking_strategy"]), str(row["method"]))
        parts.append(f'<text class="tick" x="{center:.2f}" y="{top + plot_height + 28}" text-anchor="middle"><tspan x="{center:.2f}" dy="0">{html.escape(line1)}</tspan><tspan x="{center:.2f}" dy="18">{html.escape(line2)}</tspan></text>')
    parts.append("</svg>")
    return "".join(parts)


def html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Sem comparacoes.</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def html_report(
    svg_files: list[tuple[str, str]],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    chunking_comparisons: list[dict[str, Any]],
) -> str:
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>PDF-IR report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#253247}}.charts{{display:grid;gap:2rem}}img{{max-width:100%;height:auto;border:1px solid #e2e8f0}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd4df;padding:.4rem;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}</style>
</head><body><h1>Relatório PDF-IR</h1><p>Intervalos obtidos por bootstrap de consultas dentro de cada condição.</p>
<section class="charts">{''.join(f'<figure><figcaption><h2>{html.escape(label)}</h2></figcaption><img src="{html.escape(name)}" alt="{html.escape(label)}"></figure>' for name, label in svg_files)}</section><h2>Agregados</h2>{html_table(summaries)}
<h2>Diferenças pareadas entre recuperadores</h2>{html_table(comparisons)}
<h2>Diferenças pareadas entre chunkings</h2>{html_table(chunking_comparisons)}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera tabela e graficos de uma rodada PDF-IR.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise SystemExit("Use pelo menos 100 repeticoes bootstrap.")

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "plots").resolve()
    rows = load_run(run_dir)
    summaries = summarize(
        rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    by_k = summarize_by_k(
        rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    latency_rows = latency_summary(rows)
    comparisons = paired_differences(
        rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    chunking_comparisons = paired_chunking_differences(
        rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_with_ci.csv", summaries)
    (output_dir / "metrics_with_ci.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "metrics_by_k.csv", by_k)
    (output_dir / "metrics_by_k.json").write_text(
        json.dumps(by_k, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "latency_summary.csv", latency_rows)
    (output_dir / "latency_summary.json").write_text(
        json.dumps(latency_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if comparisons:
        write_csv(output_dir / "paired_differences.csv", comparisons)
    (output_dir / "paired_differences.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if chunking_comparisons:
        write_csv(
            output_dir / "paired_chunking_differences.csv",
            chunking_comparisons,
        )
    (output_dir / "paired_chunking_differences.json").write_text(
        json.dumps(chunking_comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_svg = output_dir / "metrics_with_ci.svg"
    if legacy_svg.exists():
        legacy_svg.unlink()
    svg_files = []
    for metric, label in METRICS:
        svg_name = f"{metric}_with_ci.svg"
        (output_dir / svg_name).write_text(
            svg_chart(summaries, metric, label), encoding="utf-8"
        )
        svg_files.append((svg_name, label))
    max_k = max(int(row["k"]) for row in by_k)
    additional_charts = [
        ("metrics_at_k.svg", f"Métricas de recuperação @ {max_k}", metrics_at_k_chart(by_k, max_k)),
        ("metric_curves.svg", "Curvas de métricas por k", metric_curves_chart(by_k)),
        ("first_relevant_rank.svg", "Posição da primeira evidência relevante", first_relevant_rank_chart(rows)),
        ("latency_by_condition.svg", "Latência por condição", latency_chart(latency_rows)),
    ]
    for svg_name, label, svg in additional_charts:
        (output_dir / svg_name).write_text(svg, encoding="utf-8")
        svg_files.append((svg_name, label))
    (output_dir / "report.html").write_text(
        html_report(svg_files, summaries, comparisons, chunking_comparisons),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
