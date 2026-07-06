"""Runtime analysis and routing contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from .base import SCHEMA_VERSION, EphemeralAnalysisModel, PersistedModel, StrictContractModel


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


class MemoryUpdateExtraction(StrictContractModel):
    """Diagnostic status for whether durable candidates were extracted."""

    status: Literal["ok", "fail"]
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Require a useful reason for both extraction outcomes."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("memory_update_extraction.reason must not be empty")
        return stripped


def _validate_schema_version(schema_version: str) -> None:
    """Reject stale explicit schema versions on persisted LLM decisions."""

    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")


def _validate_memory_update_extraction(memory_candidates: list[dict], memory_update_extraction: MemoryUpdateExtraction) -> None:
    """Keep memory_update_extraction.status aligned with emitted memory candidates."""

    has_candidates = bool(memory_candidates)
    if has_candidates and memory_update_extraction.status != "ok":
        raise ValueError("memory_update_extraction.status must be ok when memory_candidates is non-empty")
    if not has_candidates and memory_update_extraction.status != "fail":
        raise ValueError("memory_update_extraction.status must be fail when memory_candidates is empty")


class Stage0DialogueAct(StrEnum):
    """Allowed conversational acts detected in the current user message."""

    QUESTION = "question"
    COMPLAINT_OR_CHALLENGE = "complaint_or_challenge"
    MEMORY_INSTRUCTION = "memory_instruction"
    CORRECTION_OR_REFINEMENT = "correction_or_refinement"
    ALIAS_OR_EQUIVALENCE_PROPOSAL = "alias_or_equivalence_proposal"
    RELATIONSHIP_UPDATE = "relationship_update"
    BIOGRAPHICAL_CONTEXT_UPDATE = "biographical_context_update"
    PREFERENCE_OR_CONSTRAINT_UPDATE = "preference_or_constraint_update"
    CONFIRMATION_REQUEST = "confirmation_request"
    ANSWER_TO_CLARIFICATION = "answer_to_clarification"
    NO_NEW_DURABLE_INFO = "no_new_durable_info"
    OTHER = "other"


class Stage0EntityKind(StrEnum):
    """Allowed normalized entity kinds in the Stage 0 frame."""

    PERSON = "person"
    ALIAS = "alias"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    TOPIC = "topic"
    OTHER = "other"


class Stage0EntityRole(StrEnum):
    """Allowed entity roles in the Stage 0 frame."""

    SUBJECT = "subject"
    OBJECT = "object"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class Stage0CurrentSignalStatus(StrEnum):
    """Allowed Stage 0 interpretation signal outcomes."""

    NONE = "none"
    CLEAR = "clear"
    POSSIBLE = "possible"
    CORRECTION = "correction"


class Stage0CurrentSignalKind(StrEnum):
    """Allowed Stage 0 interpretation signal kinds in the NLU frame."""

    NONE = "none"
    PREFERENCE = "preference"
    PERSON = "person"
    ALIAS_EQUIVALENCE = "alias_equivalence"
    RELATIONSHIP = "relationship"
    BIOGRAPHICAL_CONTEXT = "biographical_context"
    MEMORY_INSTRUCTION = "memory_instruction"
    OTHER = "other"


class Stage0Entity(StrictContractModel):
    """Structured entity or reference extracted from the current user message."""

    surface: str
    kind: Stage0EntityKind
    role: Stage0EntityRole

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, value: str) -> str:
        """Require a non-empty entity surface string."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("surface must not be empty")
        return stripped


class Stage0CurrentSignal(StrictContractModel):
    """Lightweight interpretation signal derived from the current user message."""

    status: Stage0CurrentSignalStatus
    kind: Stage0CurrentSignalKind
    summary: str
    needs_confirmation: bool


class Stage0Clarification(StrictContractModel):
    """Optional single clarification question needed before confident interpretation."""

    needed: bool
    question: str

    @model_validator(mode="after")
    def validate_question(self) -> "Stage0Clarification":
        """Require one question only when clarification is needed."""

        if self.needed and not self.question.strip():
            raise ValueError("clarification.question must be non-empty when clarification.needed=true")
        if not self.needed and self.question:
            raise ValueError("clarification.question must be empty when clarification.needed=false")
        return self


class Stage0MemorySelectionHint(StrictContractModel):
    """Non-binding hint about whether later memory selection might be useful."""

    needed: bool
    reason: str
    query_terms: list[str]


class Stage0NLUFrame(StrictContractModel):
    """Runtime interpretation frame for the current user message before Stage 1."""

    schema_version: Literal["stage0_nlu_frame.v1"]
    normalized_intent: str
    dialogue_acts: list[Stage0DialogueAct]
    entities: list[Stage0Entity]
    current_signal: Stage0CurrentSignal
    clarification: Stage0Clarification
    memory_selection_hint: Stage0MemorySelectionHint

    @field_validator("normalized_intent")
    @classmethod
    def validate_normalized_intent(cls, value: str) -> str:
        """Require a non-empty normalized intent string."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("normalized_intent must not be empty")
        return stripped

    @field_validator("dialogue_acts")
    @classmethod
    def validate_dialogue_acts(cls, value: list[Stage0DialogueAct]) -> list[Stage0DialogueAct]:
        """Require at least one valid dialogue act."""

        if not value:
            raise ValueError("dialogue_acts must not be empty")
        return value


class Stage1Decision(PersistedModel):
    """Structured future LLM decision after receiving Stage 1 context."""

    decision_type: Literal["answer_directly", "request_memory"]
    selected_memory_ids: list[str] = []
    draft_answer: str | None = None
    extracted_facts: list[dict] = []
    memory_candidates: list[dict] = []
    memory_update_extraction: MemoryUpdateExtraction
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
        _validate_schema_version(self.schema_version)
        _validate_memory_update_extraction(self.memory_candidates, self.memory_update_extraction)
        return self


class Stage2Decision(PersistedModel):
    """Structured future LLM final decision after receiving Stage 2 context."""

    final_answer: str
    extracted_facts: list[dict] = []
    memory_candidates: list[dict] = []
    memory_update_extraction: MemoryUpdateExtraction
    used_memory_ids: list[str] = []
    rationale: str | None = None

    @field_validator("final_answer")
    @classmethod
    def validate_final_answer(cls, value: str) -> str:
        """Require a non-empty final answer after whitespace trimming."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("final_answer must not be empty")
        return stripped

    @model_validator(mode="after")
    def validate_decision(self) -> "Stage2Decision":
        """Validate schema, memory update extraction, and used memory ids."""

        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        for memory_id in self.used_memory_ids:
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                deduped_ids.append(memory_id)
        object.__setattr__(self, "used_memory_ids", deduped_ids)
        _validate_schema_version(self.schema_version)
        _validate_memory_update_extraction(self.memory_candidates, self.memory_update_extraction)
        return self
