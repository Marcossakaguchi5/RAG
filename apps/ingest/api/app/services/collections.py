import re

from app.core.config import get_settings

COLLECTION_NAME_MAX_LENGTH = 64
COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_collection_name(value: str | None) -> str:
    name = (value or get_settings().qdrant_collection).strip()
    if not name:
        raise ValueError("Informe o nome da collection.")
    if len(name) > COLLECTION_NAME_MAX_LENGTH:
        raise ValueError(f"O nome da collection deve ter até {COLLECTION_NAME_MAX_LENGTH} caracteres.")
    if not COLLECTION_NAME_PATTERN.fullmatch(name):
        raise ValueError("Use apenas letras, números, hífen ou sublinhado no nome da collection.")
    return name
