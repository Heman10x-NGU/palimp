"""Tests for GraphCtx CLI commands."""

from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from graphctx.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help smoke tests
# ---------------------------------------------------------------------------


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "port" in result.stdout.lower()


def test_memory_add_help():
    result = runner.invoke(app, ["memory", "add", "--help"])
    assert result.exit_code == 0
    assert "namespace" in result.stdout.lower()


def test_knowledge_add_help():
    result = runner.invoke(app, ["knowledge", "add", "--help"])
    assert result.exit_code == 0
    assert "title" in result.stdout.lower()


def test_recall_help():
    result = runner.invoke(app, ["recall", "--help"])
    assert result.exit_code == 0
    assert "namespace" in result.stdout.lower()


def test_context_help():
    result = runner.invoke(app, ["context", "--help"])
    assert result.exit_code == 0
    assert "entity" in result.stdout.lower()


def test_stats_help():
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0
    assert "namespace" in result.stdout.lower()


def test_doctor_help():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "db" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_memory_add(tmp_path):
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(
        app,
        [
            "memory", "add",
            "Alice prefers concise answers.",
            "--namespace", "test",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "memory_id:" in result.stdout
    assert "episode_id:" in result.stdout


def test_knowledge_add_inline(tmp_path):
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(
        app,
        [
            "knowledge", "add",
            "--namespace", "test",
            "--title", "Architecture",
            "--content", "GraphCtx uses SQLite and FTS5.",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "knowledge_id:" in result.stdout
    assert "episode_id:" in result.stdout


def test_knowledge_add_file(tmp_path):
    content_file = tmp_path / "doc.md"
    content_file.write_text("GraphCtx stores memories and knowledge.")
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(
        app,
        [
            "knowledge", "add",
            "--namespace", "test",
            "--title", "Doc",
            "--file", str(content_file),
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "knowledge_id:" in result.stdout


def test_recall_after_add(tmp_path):
    db_path = str(tmp_path / "test.db")
    # Add a memory first
    runner.invoke(
        app,
        [
            "memory", "add",
            "Alice prefers concise technical answers.",
            "--namespace", "ns1",
            "--db", db_path,
        ],
    )
    # Recall — use a term that FTS5 can match
    result = runner.invoke(
        app,
        [
            "recall",
            "Alice prefers",
            "--namespace", "ns1",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "kind:" in result.stdout


def test_stats_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(
        app,
        ["stats", "--namespace", "empty", "--db", db_path],
    )
    assert result.exit_code == 0, result.stdout
    assert "memories:" in result.stdout
    assert "0" in result.stdout


def test_doctor_ok(tmp_path):
    db_path = str(tmp_path / "test.db")
    # Create the DB by adding a memory
    runner.invoke(
        app,
        [
            "memory", "add", "test content",
            "--namespace", "doc", "--db", db_path,
        ],
    )
    result = runner.invoke(app, ["doctor", "--db", db_path])
    assert result.exit_code == 0, result.stdout
    assert "integrity_check: ok" in result.stdout
    assert "all checks passed" in result.stdout


def test_doctor_missing_db():
    result = runner.invoke(app, ["doctor", "--db", "/nonexistent/path/db.sqlite"])
    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()
