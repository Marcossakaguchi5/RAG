import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings


security = HTTPBearer(auto_error=False)


def password_is_valid(password: str) -> bool:
    expected = get_settings().chat_app_password
    return hmac.compare_digest(password.encode(), expected.encode())


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _signature(payload: str) -> str:
    secret = get_settings().chat_app_password.encode()
    return _b64_encode(hmac.new(secret, payload.encode(), hashlib.sha256).digest())


def create_access_token() -> tuple[str, int]:
    expires_at = int(time.time()) + get_settings().chat_auth_token_ttl_seconds
    payload = _b64_encode(json.dumps({"exp": expires_at}, separators=(",", ":")).encode())
    return f"{payload}.{_signature(payload)}", expires_at


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload, signature = token.split(".", 1)
        if not hmac.compare_digest(signature, _signature(payload)):
            raise ValueError("assinatura invalida")
        data = json.loads(_b64_decode(payload))
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessao invalida.",
        ) from error
    if int(data.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sessao expirada.")
    return data


def require_authenticated(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Autenticacao obrigatoria.")
    return decode_access_token(credentials.credentials)
