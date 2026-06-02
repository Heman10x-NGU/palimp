"""Tests for GraphCtx eval v3 (15 categories) and benchmark v3 metrics."""

from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from graphctx.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Eval v3 tests
# ---------------------------------------------------------------------------


def test_eval_has_15_categories(tmp_path):
    """Eval output has exactly 15 categories."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        ["eval", "mini", "--namespace", "eval_15cats", "--json", "--db", db_path],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexit_code: {result.exit_code}"
    data = json.loads(result.stdout)
    assert len(data["categories"]) == 15, (
        f"Expected 15 categories, got {len(data['categories'])}: "
        f"{[c['category'] for c in data['categories']]}"
    )


def test_eval_all_pass(tmp_path):
    """All 15 categories pass."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        ["eval", "mini", "--namespace", "eval_pass", "--json", "--db", db_path],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexit_code: {result.exit_code}"
    data = json.loads(result.stdout)

    failed = [c for c in data["categories"] if not c["pass"]]
    assert data["summary"]["all_passed"], (
        f"Failed categories: {[c['category'] for c in failed]}. "
        f"Notes: {[(c['category'], c['notes']) for c in failed]}"
    )
    assert data["summary"]["total"] == 15


def test_eval_json_output(tmp_path):
    """--json produces valid JSON with expected structure."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        ["eval", "mini", "--namespace", "eval_json", "--json", "--db", db_path],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert "namespace" in data
    assert "categories" in data
    assert "summary" in data
    assert "passed" in data["summary"]
    assert "total" in data["summary"]
    assert "all_passed" in data["summary"]


# ---------------------------------------------------------------------------
# Benchmark v3 tests
# ---------------------------------------------------------------------------


def test_benchmark_v3_metrics(tmp_path):
    """Benchmark output includes graph_hops, temporal, alias, reranker info."""
    db_path = str(tmp_path / "bench.db")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--namespace", "bench_v3",
            "--items", "50",
            "--queries", "5",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexit_code: {result.exit_code}"
    data = json.loads(result.stdout)

    assert "v3_config" in data, "Missing v3_config in benchmark output"
    v3 = data["v3_config"]
    assert "graph_max_hops" in v3, "Missing graph_max_hops"
    assert "temporal_filter_enabled" in v3, "Missing temporal_filter_enabled"
    assert "alias_dedup_enabled" in v3, "Missing alias_dedup_enabled"
    assert "reranker_enabled" in v3, "Missing reranker_enabled"
    assert "category_distribution" in v3, "Missing category_distribution"

    assert isinstance(v3["graph_max_hops"], int)
    assert isinstance(v3["temporal_filter_enabled"], bool)
    assert isinstance(v3["alias_dedup_enabled"], bool)
    assert isinstance(v3["reranker_enabled"], bool)
    assert isinstance(v3["category_distribution"], dict)


# ---------------------------------------------------------------------------
# Individual category tests
# ---------------------------------------------------------------------------


def test_multi_hop_2hop_eval(tmp_path):
    """Multi-hop 2-hop bridge category passes."""
    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.eval import run_eval_mini
    from graphctx.storage import SQLiteStore

    db_path = str(tmp_path / "eval_2hop.db")
    results = run_eval_mini(ns="eval_2hop", db_path=db_path)

    cat_results = {c["category"]: c for c in results["categories"]}
    assert "multi_hop_2hop" in cat_results
    assert cat_results["multi_hop_2hop"]["pass"], (
        f"multi_hop_2hop failed: {cat_results['multi_hop_2hop']['notes']}"
    )


def test_alias_dedup_eval(tmp_path):
    """Alias dedup category passes."""
    from graphctx.eval import run_eval_mini

    db_path = str(tmp_path / "eval_alias.db")
    results = run_eval_mini(ns="eval_alias", db_path=db_path)

    cat_results = {c["category"]: c for c in results["categories"]}
    assert "alias_dedup" in cat_results
    assert cat_results["alias_dedup"]["pass"], (
        f"alias_dedup failed: {cat_results['alias_dedup']['notes']}"
    )
