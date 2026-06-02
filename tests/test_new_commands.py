"""Tests for init-agent and import CLI commands."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from palimp.cli import app

runner = CliRunner()


@pytest.fixture
def db_path(tmp_path):
    """Return a path to a temporary database."""
    return str(tmp_path / "test_palimp.db")


@pytest.fixture
def docs_dir(tmp_path):
    """Create a docs directory with sample files."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "readme.md").write_text("# Project\nThis is the readme.")
    (docs / "notes.txt").write_text("Some notes about the project.")
    (docs / "guide.rst").write_text("Guide\n=====\nA guide to the project.")

    sub = docs / "subdir"
    sub.mkdir()
    (sub / "deep.md").write_text("# Deep\nNested document content.")

    return str(docs)


@pytest.fixture
def large_file(tmp_path):
    """Create a file that exceeds the default max_bytes."""
    fpath = tmp_path / "large.md"
    fpath.write_text("x" * 2_000_000)
    return str(fpath)


# ---------------------------------------------------------------------------
# init-agent tests
# ---------------------------------------------------------------------------


class TestInitAgent:
    def test_init_agent_creates_db(self, db_path):
        """DB file is created by init-agent."""
        assert not os.path.exists(db_path)
        result = runner.invoke(
            app,
            ["init-agent", "--namespace", "test", "--db", db_path, "--client", "claude"],
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(db_path)

    def test_init_agent_inserts_starter_data(self, db_path):
        """Memory and knowledge exist after init."""
        result = runner.invoke(
            app,
            ["init-agent", "--namespace", "test", "--db", db_path, "--client", "claude"],
        )
        assert result.exit_code == 0, result.output
        assert "Starter memory inserted:" in result.output
        assert "Starter knowledge inserted:" in result.output

        # Verify data is actually in the DB
        from palimp.storage import SQLiteStore

        store = SQLiteStore(db_path)
        stats = store.get_stats("test")
        assert stats["memories"] >= 1
        assert stats["knowledge_items"] >= 1

    def test_init_agent_prints_mcp_config(self, db_path):
        """Output contains MCP JSON config."""
        result = runner.invoke(
            app,
            ["init-agent", "--namespace", "test", "--db", db_path, "--client", "claude"],
        )
        assert result.exit_code == 0, result.output
        assert "mcpServers" in result.output
        assert "palimp" in result.output
        assert "serve" in result.output

    def test_init_agent_client_variants(self, db_path):
        """All 4 client types produce valid config."""
        for client in ("claude", "cursor", "codex", "generic"):
            db_p = db_path + f".{client}"
            result = runner.invoke(
                app,
                ["init-agent", "--namespace", "test", "--db", db_p, "--client", client],
            )
            assert result.exit_code == 0, f"client={client}: {result.output}"
            assert "mcpServers" in result.output, f"client={client} missing mcpServers"


# ---------------------------------------------------------------------------
# import tests
# ---------------------------------------------------------------------------


class TestImportCommand:
    def test_import_single_file(self, tmp_path):
        """Single .md file imported as knowledge."""
        md_file = tmp_path / "single.md"
        md_file.write_text("# Hello\nThis is a test document.")
        db_path = str(tmp_path / "test.db")

        result = runner.invoke(
            app,
            ["import", "--namespace", "test", "--path", str(md_file), "--db", db_path],
        )
        assert result.exit_code == 0, result.output
        assert "success: 1" in result.output
        assert "total:" in result.output
        assert "skipped:" in result.output

        # Verify in DB
        from palimp.storage import SQLiteStore

        store = SQLiteStore(db_path)
        stats = store.get_stats("test")
        assert stats["knowledge_items"] >= 1

    def test_import_recursive_directory(self, tmp_path, docs_dir):
        """Nested docs imported with --recursive."""
        db_path = str(tmp_path / "test.db")

        result = runner.invoke(
            app,
            [
                "import",
                "--namespace", "test",
                "--path", docs_dir,
                "--db", db_path,
                "--recursive",
            ],
        )
        assert result.exit_code == 0, result.output
        # 3 top-level (.md, .txt, .rst) + 1 in subdir/.md = 4
        assert "success: 4" in result.output

    def test_import_skips_large_files(self, tmp_path, large_file):
        """Files over max_bytes are skipped."""
        db_path = str(tmp_path / "test.db")

        result = runner.invoke(
            app,
            [
                "import",
                "--namespace", "test",
                "--path", large_file,
                "--db", db_path,
                "--max-bytes", "1000",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "skipped: 1" in result.output
        assert "success: 0" in result.output

    def test_import_supported_extensions(self, tmp_path):
        """,.md, .txt, .rst all work."""
        for ext, content in [
            (".md", "# Markdown doc"),
            (".txt", "Plain text doc"),
            (".rst", "RST doc\n====="),
        ]:
            fpath = tmp_path / f"file{ext}"
            fpath.write_text(content)

        db_path = str(tmp_path / "test.db")
        result = runner.invoke(
            app,
            [
                "import",
                "--namespace", "test",
                "--path", str(tmp_path),
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "success: 3" in result.output

    def test_import_directory_no_recursive(self, tmp_path, docs_dir):
        """Without --recursive, only top-level files are imported."""
        db_path = str(tmp_path / "test.db")

        result = runner.invoke(
            app,
            [
                "import",
                "--namespace", "test",
                "--path", docs_dir,
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.output
        # 3 top-level only, subdir/deep.md excluded
        assert "success: 3" in result.output
