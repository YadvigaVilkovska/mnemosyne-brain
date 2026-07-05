"""Memory graph node."""

from __future__ import annotations

from ...db.repository import SqliteRepository
from ...memory.write import MemoryWriter
from ..state import BrainGraphState


class ApplyMemoryCandidatesNode:
    """Apply candidate refs through the memory write pipeline."""

    def __init__(self, repository: SqliteRepository, writer: MemoryWriter) -> None:
        self._repository = repository
        self._writer = writer

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        for ref in state.get("memory_candidate_refs", []):
            candidate = self._repository.get_memory_candidate(ref["candidate_id"])
            self._writer.handle_candidate_write(candidate)
        return {}
