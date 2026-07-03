from app.core.config import get_settings
from app.schemas import RagSource
from app.services.llm import get_llm_client


def build_context(sources: list[RagSource]) -> str:
    settings = get_settings()
    chunks: list[str] = []
    used = 0
    for source in sources:
        text = (
            f"[{source.rank}] chunk_id={source.chunk_id}\n"
            f"Documento: {source.document_name} | pagina: {source.page_number}\n"
            f"{source.content.strip()}"
        )
        if used + len(text) > settings.max_context_characters:
            break
        chunks.append(text)
        used += len(text)
    return "\n\n---\n\n".join(chunks)


def generate_answer(query: str, sources: list[RagSource]) -> str:
    if not sources:
        return "Nao encontrei trechos relevantes nessa collection para responder com seguranca."

    context = build_context(sources)
    llm = get_llm_client()
    return llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Voce e um assistente RAG. Responda em portugues, use apenas os contextos fornecidos, "
                    "cite as fontes entre colchetes como [1] e diga claramente quando os contextos nao sustentam uma parte da resposta."
                ),
            },
            {
                "role": "user",
                "content": f"Pergunta:\n{query}\n\nContextos:\n{context}\n\nResponda de forma direta e fundamentada.",
            },
        ]
    )
