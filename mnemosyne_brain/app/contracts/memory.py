"""Memory and dialogue contracts."""

from __future__ import annotations

from enum import StrEnum

from .base import PersistedModel, StrictContractModel
from .provenance import Provenance


class TrackStatus(StrEnum):
    """Durable track lifecycle states."""

    ACTIVE = "active"
    WAITING_FOR_EXECUTOR = "waiting_for_executor"
    CLOSED = "closed"


class DialogueTurn(PersistedModel):
    """Durable dialogue turn stored in SQLite."""

    turn_id: str
    dialogue_id: str
    track_id: str | None = None
    thread_id: str | None = None
    input_source: str
    role: str
    external_message_id: str | None = None
    content_text: str | None = None
    content_json: dict | list | str | int | float | bool | None = None
    created_at: str


class DialogueTrack(PersistedModel):
    """Durable dialogue track row."""

    track_id: str
    dialogue_id: str
    thread_id: str
    owner_user_id: str
    status: TrackStatus
    summary: str | None = None
    track_json: dict
    created_at: str
    updated_at: str
    last_turn_id: str | None = None


class MemoryCandidate(PersistedModel):
    """Candidate memory extracted from a turn or executor feedback."""

    candidate_id: str
    dialogue_id: str
    track_id: str
    turn_id: str
    candidate_type: str
    recommended_action: str
    confidence: float
    dedupe_key: str
    idempotency_key: str
    content_json: dict
    provenance_json: Provenance
    merge_target_memory_id: str | None = None
    created_at: str
    updated_at: str


class MemoryStaging(PersistedModel):
    """Staged memory record awaiting review."""

    staging_id: str
    candidate_id: str
    candidate_type: str
    status: str
    recommended_action: str
    confidence: float
    dedupe_key: str
    idempotency_key: str
    merge_target_memory_id: str | None
    conflict_memory_ids: list[str]
    content_json: dict
    provenance_json: Provenance
    review_reason: str
    reviewed_by: str | None
    reviewed_at: str | None
    created_at: str
    updated_at: str


class MemoryItem(PersistedModel):
    """Durable memory item."""

    memory_id: str
    memory_type: str
    status: str
    stability: str
    content_json: dict
    intent_tags: list[str]
    entity_keys: list[str]
    provenance_json: Provenance
    dedupe_key: str
    source_track_id: str | None
    source_turn_id: str | None
    valid_from: str | None
    valid_to: str | None
    observed_at: str | None
    confidence: float
    privacy_level: str
    created_at: str
    updated_at: str


class MemoryWriteResult(StrictContractModel):
    """Result of applying a memory candidate."""

    decision_action: str
    memory_id: str | None = None
    staging_id: str | None = None
