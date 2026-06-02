"""Shared test fixtures for Palimp."""


import pytest

from palimp.storage import SQLiteStore


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary SQLite database."""
    return str(tmp_path / "test_palimp.db")


@pytest.fixture
def store(tmp_db):
    """Return an initialized SQLiteStore backed by a temp file."""
    return SQLiteStore(tmp_db)


@pytest.fixture
def memory_store():
    """Return an in-memory SQLiteStore (faster for unit tests)."""
    return SQLiteStore(":memory:")
