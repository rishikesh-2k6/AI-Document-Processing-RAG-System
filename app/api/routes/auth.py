"""Auth routes: login and token refresh."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    """New user registration payload."""

    email: EmailStr
    password: str
    full_name: str | None = None


class LoginRequest(BaseModel):
    """Login credentials."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Refresh token payload."""

    refresh_token: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(payload: RegisterRequest, db: DBSession) -> TokenResponse:
    """Create a new user account and return JWT tokens.

    - **email**: Unique email address.
    - **password**: Plain-text password (will be hashed).
    - **full_name**: Optional display name.
    """
    # Check for existing user
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    await db.flush()  # get the ID without committing

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id), user.email),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive JWT tokens",
)
async def login(payload: LoginRequest, db: DBSession) -> TokenResponse:
    """Exchange email + password for an access token and refresh token.

    Tokens use HS256 signing. Access token expires in 30 minutes by default.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user: User | None = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id), user.email),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an expired access token",
)
async def refresh_token(payload: RefreshRequest, db: DBSession) -> TokenResponse:
    """Use a valid refresh token to obtain a new access + refresh token pair."""
    try:
        claims = decode_token(payload.refresh_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    if claims.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is not a refresh token"
        )

    user_id = claims.get("sub", "")
    email = claims.get("email", "")

    return TokenResponse(
        access_token=create_access_token(user_id, email),
        refresh_token=create_refresh_token(user_id, email),
    )
