"""Tests for graphctx benchmark and eval mini commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from graphctx.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# benchmark tests
# ---------------------------------------------------------------------------


def test_benchmark_runs(tmp_path):
    """Benchmark command completes without error."""
    db_path = str(tmp_path / "bench.db")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--namespace", "bench_test",
            "--items", "50",
            "--queries", "5",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexit_code: {result.exit_code}"


def test_benchmark_json_output(tmp_path):
    """--json produces valid JSON."""
    db_path = str(tmp_path / "bench.db")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--namespace", "bench_json",
            "--items", "50",
            "--queries", "5",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert "namespace" in data
    assert "ingest" in data
    assert "recall" in data


def test_benchmark_metrics(tmp_path):
    """p50, p95, max all present in recall metrics."""
    db_path = str(tmp_path / "bench.db")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--namespace", "bench_metrics",
            "--items", "50",
            "--queries", "5",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)

    for mode in ("fast", "hybrid", "thinking"):
        recall = data["recall"][mode]
        assert "p50_ms" in recall, f"Missing p50_ms in {mode}"
        assert "p95_ms" in recall, f"Missing p95_ms in {mode}"
        assert "max_ms" in recall, f"Missing max_ms in {mode}"
        assert "avg_result_count" in recall, f"Missing avg_result_count in {mode}"


# ---------------------------------------------------------------------------
# eval mini tests
# ---------------------------------------------------------------------------


def test_eval_mini_runs(tmp_path):
    """Eval mini command completes without error."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        [
            "eval", "mini",
            "--namespace", "eval_test",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexit_code: {result.exit_code}"


def test_eval_mini_all_categories(tmp_path):
    """All 15 categories are present in results."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        [
            "eval", "mini",
            "--namespace", "eval_cats",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)

    expected_categories = {
        "single_hop_preference",
        "static_knowledge",
        "temporal_current",
        "temporal_historical",
        "contradiction_warning",
        "namespace_isolation",
        "prompt_injection_safety",
        "deletion_exclusion",
        "multi_hop_2hop",
        "multi_hop_3hop",
        "alias_dedup",
        "category_priority_under_budget",
        "trigger_keyword_recall",
        "runbook_gotcha_pack",
        "no_answer_abstention",
    }
    actual_categories = {cat["category"] for cat in data["categories"]}
    assert actual_categories == expected_categories, (
        f"Missing: {expected_categories - actual_categories}, "
        f"Extra: {actual_categories - expected_categories}"
    )


def test_eval_mini_json_output(tmp_path):
    """--json produces valid JSON with expected structure."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        [
            "eval", "mini",
            "--namespace", "eval_json",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert "namespace" in data
    assert "categories" in data
    assert "summary" in data
    assert "passed" in data["summary"]
    assert "total" in data["summary"]
    assert "all_passed" in data["summary"]


def test_eval_mini_passes(tmp_path):
    """All categories pass (multi_hop_bridge is a placeholder that always passes)."""
    db_path = str(tmp_path / "eval.db")
    result = runner.invoke(
        app,
        [
            "eval", "mini",
            "--namespace", "eval_pass",
            "--json",
            "--db", db_path,
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)

    for cat in data["categories"]:
        assert cat["pass"], (
            f"Category '{cat['category']}' failed: {cat['notes']}"
        )

    assert data["summary"]["all_passed"], (
        f"Not all categories passed: {data['summary']['passed']}/{data['summary']['total']}"
    )
