"""Tests for temporal validity filtering."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta


from palimp.temporal import (
    classify_temporal_status,
    detect_temporal_cues,
    should_include_in_mode,
    temporal_score_boost,
)


# ---------------------------------------------------------------------------
# classify_temporal_status
# ---------------------------------------------------------------------------


class TestClassifyTemporalStatus:
    def test_classify_current_valid_from_past_no_until(self) -> None:
        """valid_from=past, valid_until=None -> current."""
        past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        status, reason = classify_temporal_status(valid_from=past, valid_until=None)
        assert status == "current"
        assert "valid since" in reason

    def test_classify_historical_valid_until_past(self) -> None:
        """valid_until in the past -> historical."""
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        status, reason = classify_temporal_status(valid_from=None, valid_until=past)
        assert status == "historical"
        assert "expired" in reason or "before reference" in reason

    def test_classify_future_valid_from_future(self) -> None:
        """valid_from in the future -> future."""
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        status, reason = classify_temporal_status(valid_from=future, valid_until=None)
        assert status == "future"
        assert "after reference" in reason

    def test_classify_current_both_bounds(self) -> None:
        """valid_from=past, valid_until=future -> current."""
        past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        status, reason = classify_temporal_status(valid_from=past, valid_until=future)
        assert status == "current"
        assert "valid between" in reason

    def test_classify_unknown_no_bounds(self) -> None:
        """No temporal bounds -> unknown."""
        status, reason = classify_temporal_status(valid_from=None, valid_until=None)
        assert status == "unknown"
        assert "no temporal bounds" in reason

    def test_as_of_changes_status(self) -> None:
        """Same fact, different as_of -> different status."""
        vf = "2023-01-01T00:00:00Z"
        vu = "2023-12-31T23:59:59Z"

        # Before the valid window -> future
        s1, _ = classify_temporal_status(vf, vu, as_of="2022-06-01T00:00:00Z")
        assert s1 == "future"

        # During the valid window -> current
        s2, _ = classify_temporal_status(vf, vu, as_of="2023-06-01T00:00:00Z")
        assert s2 == "current"

        # After the valid window -> historical
        s3, _ = classify_temporal_status(vf, vu, as_of="2024-06-01T00:00:00Z")
        assert s3 == "historical"


# ---------------------------------------------------------------------------
# should_include_in_mode
# ---------------------------------------------------------------------------


class TestShouldIncludeInMode:
    def test_current_mode_includes_current(self) -> None:
        assert should_include_in_mode("current", "current") is True

    def test_current_mode_excludes_historical(self) -> None:
        assert should_include_in_mode("historical", "current") is False

    def test_current_mode_excludes_future(self) -> None:
        assert should_include_in_mode("future", "current") is False

    def test_current_mode_includes_unknown(self) -> None:
        assert should_include_in_mode("unknown", "current") is True

    def test_historical_mode_includes_historical(self) -> None:
        assert should_include_in_mode("historical", "historical") is True

    def test_historical_mode_excludes_current(self) -> None:
        assert should_include_in_mode("current", "historical") is False

    def test_historical_mode_includes_unknown(self) -> None:
        assert should_include_in_mode("unknown", "historical") is True

    def test_all_mode_includes_everything(self) -> None:
        for status in ("current", "historical", "future", "unknown"):
            assert should_include_in_mode(status, "all") is True

    def test_auto_mode_includes_everything(self) -> None:
        for status in ("current", "historical", "future", "unknown"):
            assert should_include_in_mode(status, "auto") is True


# ---------------------------------------------------------------------------
# temporal_score_boost
# ---------------------------------------------------------------------------


class TestTemporalScoreBoost:
    def test_current_fact_in_current_mode(self) -> None:
        assert temporal_score_boost("current", "current") == 1.0

    def test_historical_fact_in_current_mode_penalized(self) -> None:
        assert temporal_score_boost("historical", "current") == 0.3

    def test_unknown_fact_in_current_mode(self) -> None:
        assert temporal_score_boost("unknown", "current") == 0.8

    def test_all_mode_neutral(self) -> None:
        for status in ("current", "historical", "future", "unknown"):
            assert temporal_score_boost(status, "all") == 1.0

    def test_auto_current_scores_highest(self) -> None:
        current = temporal_score_boost("current", "auto")
        historical = temporal_score_boost("historical", "auto")
        assert current > historical

    def test_historical_mode_prefers_historical(self) -> None:
        assert temporal_score_boost("historical", "historical") == 1.0
        assert temporal_score_boost("current", "historical") == 0.5


# ---------------------------------------------------------------------------
# detect_temporal_cues
# ---------------------------------------------------------------------------


class TestDetectTemporalCues:
    def test_auto_detects_historical_cue_in_2022(self) -> None:
        assert detect_temporal_cues("where did Alice live in 2022") == "historical"

    def test_auto_detects_historical_cue_formerly(self) -> None:
        assert detect_temporal_cues("what was formerly the config") == "historical"

    def test_auto_detects_current_cue_now(self) -> None:
        assert detect_temporal_cues("where does Alice live now") == "current"

    def test_auto_detects_current_cue_currently(self) -> None:
        assert detect_temporal_cues("what is the current version") == "current"

    def test_no_cue_returns_none(self) -> None:
        assert detect_temporal_cues("tell me about the database") is None

    def test_auto_detects_historical_cue_last_year(self) -> None:
        assert detect_temporal_cues("what happened last year") == "historical"

    def test_auto_detects_historical_cue_used_to(self) -> None:
        assert detect_temporal_cues("what config did we used to have") == "historical"


# ---------------------------------------------------------------------------
# Integration: temporal score boost affects ranking
# ---------------------------------------------------------------------------


class TestTemporalScoreBoostRanking:
    def test_current_fact_scores_higher_in_current_mode(self) -> None:
        """A current fact with base=0.5 should beat a historical fact with base=0.5."""
        current_score = 0.5 * temporal_score_boost("current", "current")
        historical_score = 0.5 * temporal_score_boost("historical", "current")
        assert current_score > historical_score

    def test_historical_fact_scores_higher_in_historical_mode(self) -> None:
        """A historical fact with base=0.5 should beat a current fact with base=0.5."""
        historical_score = 0.5 * temporal_score_boost("historical", "historical")
        current_score = 0.5 * temporal_score_boost("current", "historical")
        assert historical_score > current_score
