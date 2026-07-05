"""Memory staging persistence service."""

from __future__ import annotations

from ..contracts.analysis import ConflictDecision
from ..contracts.memory import MemoryCandidate
from ..db.repository import SqliteRepository


class MemoryStagingService:
    """Stores memory candidates that require review."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def insert_memory_staging(self, candidate: MemoryCandidate, decision: ConflictDecision) -> str:
        """Persist a staging row."""

        return self._repository.insert_memory_staging(candidate, decision)
