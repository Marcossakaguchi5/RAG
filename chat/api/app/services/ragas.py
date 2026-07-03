from app.core.config import get_settings
from app.schemas import RagasMetric, RagasReport, RagSource
from app.services.llm import get_llm_client


def _context_block(sources: list[RagSource]) -> str:
    return "\n\n".join(
        f"[{source.rank}] {source.document_name} p.{source.page_number}\n{source.content}"
        for source in sources
    )


def _metric(name: str, raw: object, reason: object = None) -> RagasMetric:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is not None:
        value = max(0.0, min(1.0, value))
    return RagasMetric(name=name, value=value, reason=str(reason) if reason else None)


def evaluate_ragas(query: str, answer: str, sources: list[RagSource], reference_answer: str = "") -> RagasReport:
    if not sources:
        return RagasReport(evaluated=False, message="Sem contextos recuperados para avaliar.")

    settings = get_settings()
    llm = get_llm_client()
    prompt = f"""
Avalie uma resposta RAG em portugues usando criterios compatíveis com RAGAS.
Retorne apenas JSON valido com chaves numericas de 0 a 1 e justificativas curtas:
faithfulness, faithfulness_reason, answer_relevancy, answer_relevancy_reason,
context_precision, context_precision_reason, context_recall, context_recall_reason,
answer_correctness, answer_correctness_reason.

Pergunta:
{query}

Resposta gerada:
{answer}

Contextos recuperados:
{_context_block(sources)}

Resposta de referencia opcional:
{reference_answer or "NAO INFORMADA"}

Regras:
- faithfulness mede se a resposta e sustentada pelos contextos.
- answer_relevancy mede se a resposta responde diretamente a pergunta.
- context_precision mede se os contextos usados sao relevantes para a pergunta/resposta.
- context_recall so deve receber nota alta quando a referencia estiver informada e os contextos cobrirem seus fatos.
- answer_correctness so deve receber nota alta quando a referencia estiver informada e a resposta concordar com ela.
- Se nao houver referencia, use null em context_recall e answer_correctness.
"""
    data = llm.json_chat(
        [
            {"role": "system", "content": "Voce e um avaliador objetivo de sistemas RAG. Responda somente JSON."},
            {"role": "user", "content": prompt},
        ],
        model=settings.resolved_ragas_model,
    )
    metrics = [
        _metric("Faithfulness", data.get("faithfulness"), data.get("faithfulness_reason")),
        _metric("Answer relevancy", data.get("answer_relevancy"), data.get("answer_relevancy_reason")),
        _metric("Context precision", data.get("context_precision"), data.get("context_precision_reason")),
        _metric("Context recall", data.get("context_recall"), data.get("context_recall_reason")),
        _metric("Answer correctness", data.get("answer_correctness"), data.get("answer_correctness_reason")),
    ]
    if not any(metric.value is not None for metric in metrics):
        return RagasReport(evaluated=False, message="O avaliador RAGAS nao retornou JSON valido.")
    return RagasReport(evaluated=True, metrics=metrics)
