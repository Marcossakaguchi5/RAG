import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Acesso não autorizado.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def password_is_valid(password: str) -> bool:
    return hmac.compare_digest(password, get_settings().ingest_app_password)


def create_access_token() -> tuple[str, int]:
    settings = get_settings()
    expires_at = int(time.time()) + settings.ingest_auth_token_ttl_seconds
    payload = _encode(json.dumps({"exp": expires_at}, separators=(",", ":")).encode("utf-8"))
    signature = _encode(
        hmac.new(settings.ingest_app_password.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{payload}.{signature}", expires_at


def verify_access_token(token: str) -> None:
    try:
        payload, supplied_signature = token.split(".", maxsplit=1)
        expected_signature = _encode(
            hmac.new(
                get_settings().ingest_app_password.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("invalid signature")
        data = json.loads(_decode(payload))
        if not isinstance(data.get("exp"), int) or data["exp"] <= time.time():
            raise ValueError("expired token")
    except (ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        raise _unauthorized() from None


def require_authenticated(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    verify_access_token(credentials.credentials)
