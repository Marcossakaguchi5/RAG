import math

from app.schemas import EvaluationMetrics


def calculate_metrics(result_ids: list[str], relevant_ids: list[str], top_k: int) -> EvaluationMetrics:
    relevant = set(relevant_ids)
    if not relevant:
        return EvaluationMetrics(
            evaluated=False,
            message="Informe IDs de chunks relevantes para calcular métricas supervisionadas.",
        )

    ranked = result_ids[:top_k]
    relevant_hits = [chunk_id in relevant for chunk_id in ranked]
    hits = sum(relevant_hits)
    precision = hits / top_k
    recall = hits / len(relevant)

    accumulated_precision = 0.0
    reciprocal_rank = 0.0
    dcg = 0.0
    for position, is_relevant in enumerate(relevant_hits, start=1):
        if is_relevant:
            accumulated_precision += sum(relevant_hits[:position]) / position
            if reciprocal_rank == 0:
                reciprocal_rank = 1 / position
            dcg += 1 / math.log2(position + 1)

    average_precision = accumulated_precision / len(relevant)
    ideal_count = min(len(relevant), top_k)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return EvaluationMetrics(
        evaluated=True,
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        map=round(average_precision, 4),
        ndcg_at_k=round(ndcg, 4),
        mrr=round(reciprocal_rank, 4),
    )
