"""Conflict decision service."""

from __future__ import annotations

from ..contracts.analysis import ConflictAction, ConflictDecision
from ..contracts.memory import MemoryCandidate
from .dedupe import MemoryDeduper

REVIEW_REASON = "requires_review"
WRITE_REASON = "save_immediately_after_conflict_check"
DUPLICATE_REASON = "duplicate_dedupe_key"


class ConflictResolver:
    """Produces ConflictDecision before any candidate write is applied."""

    def __init__(self, deduper: MemoryDeduper) -> None:
        self._deduper = deduper

    def decide(self, candidate: MemoryCandidate) -> ConflictDecision:
        """Evaluate duplicates and recommended action."""

        conflicts = self._deduper.find_conflicts(candidate.dedupe_key)
        if conflicts:
            return ConflictDecision(
                action=ConflictAction.SKIP_DUPLICATE,
                reason=DUPLICATE_REASON,
                conflict_memory_ids=conflicts,
            )
        if candidate.recommended_action == "save_immediately":
            return ConflictDecision(action=ConflictAction.WRITE_MEMORY, reason=WRITE_REASON)
        return ConflictDecision(action=ConflictAction.STAGE_MEMORY, reason=REVIEW_REASON)
