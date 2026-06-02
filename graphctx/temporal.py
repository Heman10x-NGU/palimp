"""Temporal validity filtering and classification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def classify_temporal_status(
    valid_from: Optional[str],
    valid_until: Optional[str],
    as_of: Optional[str] = None,
) -> tuple[str, str]:
    """Classify temporal status of a fact.

    Returns (status, reason).
    """
    ref_time = _parse_ts(as_of) if as_of else datetime.now(timezone.utc)

    vf = _parse_ts(valid_from) if valid_from else None
    vu = _parse_ts(valid_until) if valid_until else None

    if vf and ref_time < vf:
        return "future", f"valid_from {valid_from} is after reference time"
    if vu and ref_time > vu:
        return "historical", f"valid_until {valid_until} is before reference time"
    if vf and vu:
        return "current", f"valid between {valid_from} and {valid_until}"
    if vf and not vu:
        return "current", f"valid since {valid_from}, no expiry"
    if not vf and vu:
        if ref_time > vu:
            return "historical", f"expired at {valid_until}"
        return "current", f"valid until {valid_until}"
    return "unknown", "no temporal bounds"


def should_include_in_mode(temporal_status: str, temporal_mode: str) -> bool:
    """Whether to include a result given temporal mode."""
    if temporal_mode == "all":
        return True
    if temporal_mode == "current":
        return temporal_status in ("current", "unknown")
    if temporal_mode == "historical":
        return temporal_status in ("historical", "unknown")
    # auto: prefer current, include historical with lower priority
    return True


def temporal_score_boost(temporal_status: str, temporal_mode: str) -> float:
    """Score boost/penalty based on temporal match."""
    if temporal_mode == "all":
        return 1.0
    if temporal_mode == "current":
        if temporal_status == "current":
            return 1.0
        if temporal_status == "unknown":
            return 0.8
        return 0.3  # historical/future penalized
    if temporal_mode == "historical":
        if temporal_status == "historical":
            return 1.0
        if temporal_status == "unknown":
            return 0.8
        return 0.5  # current deprioritized
    # auto
    if temporal_status == "current":
        return 1.0
    if temporal_status == "historical":
        return 0.6
    return 0.7


def detect_temporal_cues(query: str) -> Optional[str]:
    """Detect temporal cues in query text."""
    q = query.lower()
    historical_cues = [
        "before", "previous", "formerly", "in 2022", "in 2023",
        "in 2024", "in 2025", "last year", "used to", "was",
        "were", "had been",
    ]
    current_cues = [
        "now", "currently", "today", "right now", "at present",
        "is", "are",
    ]

    for cue in historical_cues:
        if cue in q:
            return "historical"
    for cue in current_cues:
        if cue in q:
            return "current"
    return None


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
