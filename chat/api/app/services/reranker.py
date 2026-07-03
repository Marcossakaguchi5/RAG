import math
import re
import unicodedata

from app.schemas import RagSource, SearchHit


STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "que", "qual", "quais", "como", "sobre", "sao", "ser", "tem", "ha", "ou", "ao", "aos",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]{2,}", _normalize(text)) if token not in STOPWORDS]


def _lexical_score(query_tokens: list[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = _tokens(content)
    if not content_tokens:
        return 0.0
    frequencies: dict[str, int] = {}
    for token in content_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    unique_query = set(query_tokens)
    overlap = sum(1 for token in unique_query if token in frequencies) / len(unique_query)
    tf = sum(math.log1p(frequencies.get(token, 0)) for token in unique_query) / len(unique_query)
    phrase_bonus = 0.12 if " ".join(query_tokens[: min(5, len(query_tokens))]) in _normalize(content) else 0.0
    return min(1.0, (0.72 * overlap) + (0.28 * min(tf, 1.0)) + phrase_bonus)


def rerank(query: str, hits: list[SearchHit], top_k: int, enabled: bool) -> list[RagSource]:
    if not hits:
        return []

    max_retrieval_score = max(abs(hit.score) for hit in hits) or 1.0
    query_tokens = _tokens(query)
    scored: list[tuple[float, int, SearchHit]] = []
    for index, hit in enumerate(hits):
        retrieval_score = max(0.0, hit.score / max_retrieval_score)
        lexical_score = _lexical_score(query_tokens, hit.content)
        final_score = (0.62 * retrieval_score) + (0.38 * lexical_score) if enabled else retrieval_score
        scored.append((final_score, index, hit))

    if enabled:
        scored.sort(key=lambda item: (-item[0], item[1]))

    sources: list[RagSource] = []
    for rank, (score, original_index, hit) in enumerate(scored[:top_k], start=1):
        sources.append(
            RagSource(
                **hit.model_dump(),
                rank=rank,
                retrieval_rank=original_index + 1,
                rerank_score=round(score, 6) if enabled else None,
            )
        )
    return sources
