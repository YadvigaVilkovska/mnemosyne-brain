"""Runtime analysis and routing contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from .base import EphemeralAnalysisModel, PersistedModel


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


class Stage1Decision(PersistedModel):
    """Structured future LLM decision after receiving Stage 1 context."""

    decision_type: Literal["answer_directly", "request_memory"]
    selected_memory_ids: list[str] = []
    draft_answer: str | None = None
    extracted_facts: list[dict] = []
    memory_candidates: list[dict] = []
    rationale: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "Stage1Decision":
        """Enforce decision-specific memory selection rules."""

        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        for memory_id in self.selected_memory_ids:
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                deduped_ids.append(memory_id)
        object.__setattr__(self, "selected_memory_ids", deduped_ids)

        if self.decision_type == "answer_directly" and self.selected_memory_ids:
            raise ValueError("answer_directly decisions must not select memory ids")
        if self.decision_type == "request_memory" and not self.selected_memory_ids:
            raise ValueError("request_memory decisions require selected_memory_ids")
        return self
