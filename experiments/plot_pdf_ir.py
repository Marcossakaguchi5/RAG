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
from statistics import mean
from typing import Any


METRICS = [
    ("mrr", "MRR"),
    ("ndcg_at_k", "nDCG@k"),
    ("recall_at_k", "Recall@k"),
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
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_chart(summaries: list[dict[str, Any]]) -> str:
    series = sorted(
        {(row["chunking_strategy"], row["method"]) for row in summaries}
    )
    colors = ["#31688e", "#7e4fa3", "#24907b", "#d17c28", "#b84a62", "#5f6b78"]
    panel_height = 350
    width = 1280
    height = 80 + panel_height * len(METRICS)
    left, right = 90, 35
    plot_width = width - left - right
    bar_slot = plot_width / max(1, len(series))
    bar_width = min(70, bar_slot * 0.62)
    by_key = {
        (row["metric"], row["chunking_strategy"], row["method"]): row
        for row in summaries
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#253247}.title{font-size:24px;font-weight:700}.label{font-size:13px}.tick{font-size:12px}.grid{stroke:#d7dee8;stroke-width:1}.axis{stroke:#253247;stroke-width:1.5}.ci{stroke:#1f2937;stroke-width:2}</style>',
        '<text class="title" x="40" y="42">PDF-IR: médias e IC95% bootstrap por consulta</text>',
    ]
    for panel_index, (metric, label) in enumerate(METRICS):
        top = 75 + panel_index * panel_height
        plot_top = top + 35
        plot_bottom = top + 265
        plot_height = plot_bottom - plot_top
        parts.append(f'<text class="title" x="{left}" y="{top + 22}">{html.escape(label)}</text>')
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
            parts.append(f'<text class="label" x="{center:.2f}" y="{max(plot_top+12, y-8):.2f}" text-anchor="middle">{mean_value:.3f}</text>')
            series_label = f"{strategy} / {method}"
            parts.append(f'<text class="tick" x="{center:.2f}" y="{plot_bottom+22}" text-anchor="middle" transform="rotate(28 {center:.2f} {plot_bottom+22})">{html.escape(series_label)}</text>')
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
    svg_name: str,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    chunking_comparisons: list[dict[str, Any]],
) -> str:
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>PDF-IR report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#253247}}img{{max-width:100%}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd4df;padding:.4rem;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}</style>
</head><body><h1>Relatório PDF-IR</h1><p>Intervalos obtidos por bootstrap de consultas dentro de cada condição.</p>
<img src="{html.escape(svg_name)}" alt="Métricas PDF-IR"><h2>Agregados</h2>{html_table(summaries)}
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
    svg_name = "metrics_with_ci.svg"
    (output_dir / svg_name).write_text(svg_chart(summaries), encoding="utf-8")
    (output_dir / "report.html").write_text(
        html_report(svg_name, summaries, comparisons, chunking_comparisons),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
