"""Executor contracts."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import field_validator

from .base import ImmutableEventModel, PersistedModel, StrictContractModel

JsonValue = Any


def validate_json_value(value: JsonValue) -> JsonValue:
    """Validate that a value is representable in a JSON column."""

    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON floats must be finite")
        return value
    if isinstance(value, list):
        return [validate_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: validate_json_value(item) for key, item in value.items()}
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}")


class ExecutionPolicy(StrictContractModel):
    """Executor scheduling policy for task creation."""

    executor: str = "hermes"
    max_attempts: int = 1


class ExecutorError(StrictContractModel):
    """Structured executor error payload."""

    code: str
    message: str


class ExecutorArtifact(StrictContractModel):
    """Structured executor artifact reference."""

    artifact_id: str
    artifact_type: str
    uri: str | None = None
    metadata: dict[str, JsonValue] = {}


class ExecutorTaskCapsule(PersistedModel):
    """Durable executor task capsule."""

    capsule_id: str
    source_track_id: str
    thread_id: str
    executor: str
    status: str
    idempotency_key: str
    attempt_count: int = 0
    capsule_json: dict[str, JsonValue]
    result_json: dict[str, JsonValue] | None = None
    last_error_json: dict[str, JsonValue] | None = None
    created_at: str
    updated_at: str


class ExecutorCallback(ImmutableEventModel):
    """Incoming callback payload before Brain adds received_at."""

    event_id: str
    capsule_id: str
    correlation_id: str
    executor: str
    status: str
    attempt: int
    is_final: bool
    payload: JsonValue
    error: ExecutorError | None = None
    artifacts: list[ExecutorArtifact] = []
    created_at: str

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: JsonValue) -> JsonValue:
        """Reject callback payloads that cannot be stored as JSON."""

        return validate_json_value(value)


class ExecutorEvent(ExecutorCallback):
    """Persisted executor callback event."""

    received_at: str
    applied: bool = False
    stale: bool = False
    applied_at: str | None = None


class ExecutorCallbackResult(StrEnum):
    """Callback service result statuses."""

    ACCEPTED = "accepted"
    DUPLICATE_ACCEPTED = "duplicate_accepted"
    ACCEPTED_STALE = "accepted_stale"
