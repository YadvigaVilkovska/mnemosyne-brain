"""Runtime analysis and routing contracts."""

from __future__ import annotations

from enum import StrEnum

from .base import EphemeralAnalysisModel


class ConflictAction(StrEnum):
    """Allowed outcomes of conflict handling."""

    WRITE_MEMORY = "write_memory"
    STAGE_MEMORY = "stage_memory"
    SKIP_DUPLICATE = "skip_duplicate"


class ConflictDecision(EphemeralAnalysisModel):
    """Decision produced before any memory candidate is applied."""

    action: ConflictAction
    reason: str
    conflict_memory_ids: list[str] = []


class RouterDecision(EphemeralAnalysisModel):
    """Graph route selected after analysis."""

    route: str
    reason: str


class ExecutorFeedbackAnalysis(EphemeralAnalysisModel):
    """Runtime-only interpretation of a persisted executor event."""

    event_id: str
    should_update_track: bool
    local_answer: str
