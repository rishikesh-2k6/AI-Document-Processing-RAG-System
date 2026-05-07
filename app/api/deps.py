"""Shared FastAPI dependencies: auth, DB session, rate limiter."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import decode_token
from app.db.models import User
from app.db.session import get_db

log = get_logger(__name__)

# ── Auth dependency ───────────────────────────────────────────────────────────


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate JWT from Authorization header, return User.

    Raises:
        HTTPException 401: Missing or invalid token.
        HTTPException 403: User is inactive.
        HTTPException 404: User no longer exists.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise credentials_exc

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
    except ValueError:
        raise credentials_exc

    token_type: str = payload.get("type", "")
    if token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use access token",
        )

    user_id_str: str | None = payload.get("sub")
    if not user_id_str:
        raise credentials_exc

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exc

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return user


# ── Convenience type aliases ──────────────────────────────────────────────────

CurrentUser = Annotated[User, Depends(get_current_user)]
DBSession = Annotated[AsyncSession, Depends(get_db)]
