"""Palimp custom exceptions."""


class PalimpError(Exception):
    """Base exception for all Palimp errors."""


class NamespaceRequiredError(PalimpError):
    """Raised when a namespace is required but not provided."""

    def __init__(self, msg: str = "Namespace is required for this operation."):
        super().__init__(msg)


class DimensionDriftError(PalimpError):
    """Raised when an embedding dimension does not match the existing dimension in the DB."""

    def __init__(self, expected: int, got: int):
        self.expected = expected
        self.got = got
        super().__init__(
            f"Embedding dimension drift: DB expects {expected}, got {got}."
        )


class ExtractionFailedError(PalimpError):
    """Raised when entity/edge/claim extraction fails."""

    def __init__(self, msg: str = "Extraction failed; source stored without graph facts."):
        super().__init__(msg)


class EpisodeNotFoundError(PalimpError):
    """Raised when an episode cannot be found."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        super().__init__(f"Episode not found: {episode_id}")
