"""JWT security helpers — token creation, validation, and password hashing."""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ── Password hashing ──────────────────────────────────────────────────────────
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    """Return bcrypt hash of *plain* password."""
    return _pwd_context.hash(plain)


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _create_token(subject: str, expires_delta: timedelta, extra: dict[str, Any]) -> str:
    """Internal helper — encode a JWT with expiry and extra claims."""
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_delta,
        **extra,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, email: str) -> str:
    """Create a short-lived access JWT."""
    return _create_token(
        subject=user_id,
        expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        extra={"email": email, "type": "access"},
    )


def create_refresh_token(user_id: str, email: str) -> str:
    """Create a long-lived refresh JWT."""
    return _create_token(
        subject=user_id,
        expires_delta=timedelta(days=settings.jwt_refresh_token_expire_days),
        extra={"email": email, "type": "refresh"},
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate JWT. Raises ValueError on failure."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc
