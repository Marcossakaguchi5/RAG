from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryMetrics:
    hit_rate: float
    precision: float
    recall: float
    average_precision: float
    ndcg: float
    mrr: float


def calculate_query_metrics(result_ids: list[str], relevant_ids: set[str], top_k: int) -> QueryMetrics:
    if not relevant_ids:
        return QueryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ranked = result_ids[:top_k]
    relevant_hits = [doc_id in relevant_ids for doc_id in ranked]
    hits = sum(relevant_hits)

    accumulated_precision = 0.0
    reciprocal_rank = 0.0
    dcg = 0.0
    seen_relevant = 0

    for position, is_relevant in enumerate(relevant_hits, start=1):
        if not is_relevant:
            continue
        seen_relevant += 1
        accumulated_precision += seen_relevant / position
        if reciprocal_rank == 0.0:
            reciprocal_rank = 1 / position
        dcg += 1 / math.log2(position + 1)

    ideal_count = min(len(relevant_ids), top_k)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_count + 1))

    return QueryMetrics(
        hit_rate=1.0 if hits else 0.0,
        precision=hits / top_k,
        recall=hits / len(relevant_ids),
        average_precision=accumulated_precision / len(relevant_ids),
        ndcg=dcg / ideal_dcg if ideal_dcg else 0.0,
        mrr=reciprocal_rank,
    )


def average_metrics(items: list[QueryMetrics]) -> dict[str, float]:
    if not items:
        return {
            "hit_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
        }

    total = len(items)
    return {
        "hit_rate": round(sum(item.hit_rate for item in items) / total, 6),
        "precision": round(sum(item.precision for item in items) / total, 6),
        "recall": round(sum(item.recall for item in items) / total, 6),
        "map": round(sum(item.average_precision for item in items) / total, 6),
        "ndcg": round(sum(item.ndcg for item in items) / total, 6),
        "mrr": round(sum(item.mrr for item in items) / total, 6),
    }
