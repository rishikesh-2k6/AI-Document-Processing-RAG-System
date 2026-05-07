"""Unit tests for JWT security helpers."""

from __future__ import annotations

import time

import pytest
from jose import jwt

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """Tests for bcrypt password hashing."""

    def test_hash_is_not_plaintext(self) -> None:
        hashed = hash_password("mysecretpassword")
        assert hashed != "mysecretpassword"

    def test_verify_correct_password(self) -> None:
        hashed = hash_password("correct_horse_battery_staple")
        assert verify_password("correct_horse_battery_staple", hashed) is True

    def test_verify_wrong_password(self) -> None:
        hashed = hash_password("correct_horse_battery_staple")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_different_each_time(self) -> None:
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2  # bcrypt uses random salt


class TestJWTTokens:
    """Tests for JWT token creation and decoding."""

    USER_ID = "550e8400-e29b-41d4-a716-446655440000"
    EMAIL = "test@example.com"

    def test_access_token_has_correct_type(self) -> None:
        token = create_access_token(self.USER_ID, self.EMAIL)
        payload = decode_token(token)
        assert payload["type"] == "access"

    def test_refresh_token_has_correct_type(self) -> None:
        token = create_refresh_token(self.USER_ID, self.EMAIL)
        payload = decode_token(token)
        assert payload["type"] == "refresh"

    def test_token_contains_subject(self) -> None:
        token = create_access_token(self.USER_ID, self.EMAIL)
        payload = decode_token(token)
        assert payload["sub"] == self.USER_ID

    def test_token_contains_email(self) -> None:
        token = create_access_token(self.USER_ID, self.EMAIL)
        payload = decode_token(token)
        assert payload["email"] == self.EMAIL

    def test_decode_invalid_token_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid token"):
            decode_token("not.a.valid.jwt")

    def test_decode_tampered_token_raises(self) -> None:
        token = create_access_token(self.USER_ID, self.EMAIL)
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(ValueError):
            decode_token(tampered)

    def test_access_token_has_expiry(self) -> None:
        token = create_access_token(self.USER_ID, self.EMAIL)
        payload = decode_token(token)
        assert "exp" in payload
        # Should expire in the future
        assert payload["exp"] > time.time()
