import hashlib

from app.core.config import get_settings
from app.schemas import RagSource
from app.services.llm import get_llm_client


SYSTEM_PROMPT = (
    "Voce e um assistente RAG. Responda em portugues, use apenas os contextos fornecidos, "
    "cite as fontes entre colchetes como [1] e diga claramente quando os contextos nao sustentam uma parte da resposta."
)
USER_PROMPT_TEMPLATE = "Pergunta:\n{query}\n\nContextos:\n{context}\n\nResponda de forma direta e fundamentada."


def prompt_sha256() -> str:
    template = f"system:\n{SYSTEM_PROMPT}\n\nuser:\n{USER_PROMPT_TEMPLATE}"
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _format_source(source: RagSource) -> str:
    return (
        f"[{source.rank}] chunk_id={source.chunk_id}\n"
        f"Documento: {source.document_name} | pagina: {source.page_number}\n"
        f"{source.content.strip()}"
    )


def select_generation_sources(sources: list[RagSource]) -> list[RagSource]:
    settings = get_settings()
    selected: list[RagSource] = []
    used = 0
    for source in sources:
        if not source.content.strip():
            continue
        text = _format_source(source)
        separator_size = len("\n\n---\n\n") if selected else 0
        if used + separator_size + len(text) > settings.max_context_characters:
            break
        selected.append(source)
        used += separator_size + len(text)
    return selected


def build_context(sources: list[RagSource]) -> str:
    return "\n\n---\n\n".join(_format_source(source) for source in select_generation_sources(sources))


def generate_answer(query: str, sources: list[RagSource]) -> str:
    if not sources:
        return "Nao encontrei trechos relevantes nessa collection para responder com seguranca."

    context = build_context(sources)
    llm = get_llm_client()
    return llm.chat(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(query=query, context=context),
            },
        ]
    )
