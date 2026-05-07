"""Pytest configuration and shared fixtures."""

import pytest


def pytest_configure(config):  # type: ignore
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped in fast mode)"
    )
