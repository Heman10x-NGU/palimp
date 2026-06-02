"""Entity alias normalization and deduplication."""

import re

_STRIP_ARTICLES = re.compile(r"^(the|a|an)(\s+|$)", re.IGNORECASE)
_COLLAPSE_PUNCT = re.compile(r"[\s\-_]+")


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for dedup matching.

    Lowercases, strips leading articles (the/a/an), collapses
    whitespace/hyphens/underscores into single spaces, and trims.
    """
    n = name.strip().lower()
    n = _STRIP_ARTICLES.sub("", n)
    n = _COLLAPSE_PUNCT.sub(" ", n).strip()
    return n


def entities_compatible(type_a: str, type_b: str) -> bool:
    """Check if two entity types are compatible for merging.

    Returns True if either type is empty/None, or if both types match
    case-insensitively.
    """
    if not type_a or not type_b:
        return True
    return type_a.lower() == type_b.lower()
