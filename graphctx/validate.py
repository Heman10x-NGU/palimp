"""Shared validation for REST, CLI, and MCP paths."""

from __future__ import annotations

import re

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ValidationError(Exception):
    """Raised when a validated field fails its check."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def validate_namespace(ns: str | None) -> str:
    """Validate namespace. Raises ValidationError if invalid.

    Accepts: 1-64 alphanumeric, hyphen, or underscore characters.
    Rejects: None, empty string, whitespace-only, invalid chars, too long.
    """
    if not isinstance(ns, str) or not ns.strip():
        raise ValidationError(
            "namespace",
            "Invalid namespace. Must be 1-64 alphanumeric/hyphen/underscore characters.",
        )
    ns = ns.strip()
    if not _NAMESPACE_RE.match(ns):
        raise ValidationError(
            "namespace",
            "Invalid namespace. Must be 1-64 alphanumeric/hyphen/underscore characters.",
        )
    return ns


def validate_content(content: str | None, field: str = "content") -> str:
    """Validate content is non-empty. Raises ValidationError if empty."""
    if not content or not content.strip():
        raise ValidationError(field, f"{field} is required and cannot be empty.")
    return content.strip()


def validate_batch_size(items: list, max_size: int = 50) -> list:
    """Validate batch size. Raises ValidationError if too large."""
    if len(items) > max_size:
        raise ValidationError(
            "items", f"Batch size {len(items)} exceeds maximum {max_size}."
        )
    return items
