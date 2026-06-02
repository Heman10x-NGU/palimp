"""Ebbinghaus forgetting curve for GraphCtx episode decay.

Retention model: R = e^(-t / S)
  t = days since last access
  S = stability (days; higher = slower decay)

Pinned items have infinite stability and never decay.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# Default stability: 30-day half-life (R = ~0.37 at 30 days)
DEFAULT_STABILITY = 30.0

# Pinned items never decay
PIN_STABILITY = float("inf")


def ebbinghaus_retention(
    last_accessed: str,
    stability: float = DEFAULT_STABILITY,
    pinned: bool = False,
) -> float:
    """Compute retention using Ebbinghaus forgetting curve.

    Parameters
    ----------
    last_accessed : str
        ISO-8601 timestamp of last access (or creation).
    stability : float
        Stability parameter in days. Higher values slow decay.
    pinned : bool
        If True, always returns 1.0 (item never decays).

    Returns
    -------
    float
        Retention score in [0.0, 1.0].
    """
    if pinned:
        return 1.0

    now = datetime.now(timezone.utc)
    try:
        accessed = datetime.fromisoformat(last_accessed)
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0

    t_days = max((now - accessed).total_seconds() / 86400.0, 0.0)
    return math.exp(-t_days / stability)


def compute_decay_score(
    created_at: str,
    last_accessed_at: str | None = None,
    stability: float = DEFAULT_STABILITY,
    pinned: bool = False,
) -> float:
    """Compute decay score for a memory/knowledge item.

    Uses last_accessed_at when available, otherwise falls back to created_at.
    """
    ref_time = last_accessed_at or created_at
    return ebbinghaus_retention(ref_time, stability, pinned)
