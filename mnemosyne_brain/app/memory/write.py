"""Conflict-aware memory write pipeline."""

from __future__ import annotations

from ..contracts.analysis import ConflictAction, ConflictDecision
from ..contracts.identity import IdentifierAssignment
from ..contracts.memory import MemoryCandidate, MemoryWriteResult
from ..db.repository import SqliteRepository
from .conflicts import ConflictResolver
from .staging import MemoryStagingService


class MemoryWriter:
    """Applies MemoryCandidate through dedupe, conflicts and atomic persistence."""

    def __init__(
        self,
        repository: SqliteRepository,
        conflict_resolver: ConflictResolver,
        staging_service: MemoryStagingService,
    ) -> None:
        self._repository = repository
        self._conflict_resolver = conflict_resolver
        self._staging_service = staging_service

    def handle_candidate_write(
        self,
        candidate: MemoryCandidate,
        *,
        assignment: IdentifierAssignment | None = None,
    ) -> MemoryWriteResult:
        """Handle a candidate through ConflictDecision before applying it."""

        decision = self._conflict_resolver.decide(candidate)
        return self.apply_conflict_decision_atomic(candidate, decision, assignment=assignment)

    def apply_conflict_decision_atomic(
        self,
        candidate: MemoryCandidate,
        decision: ConflictDecision,
        *,
        assignment: IdentifierAssignment | None = None,
    ) -> MemoryWriteResult:
        """Atomically apply memory, identity and audit writes."""

        with self._repository.transaction():
            if decision.action is ConflictAction.SKIP_DUPLICATE:
                audit_id = self._repository.insert_audit_event(
                    event_type="memory_duplicate_skipped",
                    actor_type="system",
                    dialogue_id=candidate.dialogue_id,
                    track_id=candidate.track_id,
                    turn_id=candidate.turn_id,
                    target_type="memory_candidate",
                    target_id=candidate.candidate_id,
                    payload={"reason": decision.reason, "conflicts": decision.conflict_memory_ids},
                )
                return MemoryWriteResult(decision_action=decision.action.value, memory_id=audit_id)
            if decision.action is ConflictAction.STAGE_MEMORY:
                staging_id = self._staging_service.insert_memory_staging(candidate, decision)
                self._repository.insert_audit_event(
                    event_type="memory_staged",
                    actor_type="system",
                    dialogue_id=candidate.dialogue_id,
                    track_id=candidate.track_id,
                    turn_id=candidate.turn_id,
                    target_type="memory_staging",
                    target_id=staging_id,
                    payload={"candidate_id": candidate.candidate_id},
                )
                return MemoryWriteResult(
                    decision_action=decision.action.value,
                    staging_id=staging_id,
                )

            if assignment is not None:
                self._repository.insert_identifier_assignment(assignment)
            memory_id = self._repository.insert_memory_item(candidate)
            self._repository.insert_audit_event(
                event_type="memory_written",
                actor_type="system",
                dialogue_id=candidate.dialogue_id,
                track_id=candidate.track_id,
                turn_id=candidate.turn_id,
                target_type="memory_item",
                target_id=memory_id,
                payload={"candidate_id": candidate.candidate_id},
            )
            return MemoryWriteResult(decision_action=decision.action.value, memory_id=memory_id)


def handle_candidate_write(
    writer: MemoryWriter,
    candidate: MemoryCandidate,
    *,
    assignment: IdentifierAssignment | None = None,
) -> MemoryWriteResult:
    """Module-level function required by the MVP contract."""

    return writer.handle_candidate_write(candidate, assignment=assignment)


def apply_conflict_decision_atomic(
    writer: MemoryWriter,
    candidate: MemoryCandidate,
    decision: ConflictDecision,
    *,
    assignment: IdentifierAssignment | None = None,
) -> MemoryWriteResult:
    """Module-level function required by the MVP contract."""

    return writer.apply_conflict_decision_atomic(candidate, decision, assignment=assignment)
