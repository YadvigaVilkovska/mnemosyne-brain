"""Identity contracts and validation."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from .base import PersistedModel
from .provenance import Provenance


class IdentifierAssignment(PersistedModel):
    """Current or historical assignment of an identifier to a person/persona."""

    assignment_id: str
    identifier_key: str
    person_id: str | None = None
    persona_id: str | None = None
    resolution_status: str
    candidate_person_ids: list[str] = []
    assignment_scope: str = "individual"
    status: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float
    provenance_json: Provenance
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def validate_resolution_rules(self) -> "IdentifierAssignment":
        """Enforce explicit identity resolution invariants."""

        if self.resolution_status == "resolved" and self.person_id is None:
            raise ValueError("resolved identifier assignments require person_id")
        if self.resolution_status == "ambiguous" and not self.candidate_person_ids:
            raise ValueError("ambiguous identifier assignments require candidate_person_ids")
        if self.resolution_status == "unresolved" and self.person_id is not None:
            raise ValueError("unresolved identifier assignments must not have person_id")
        return self


class IdentityInput(PersistedModel):
    """Validated identity input used by application services."""

    raw_value: str
    identifier_type: str

    @field_validator("raw_value")
    @classmethod
    def validate_raw_value(cls, value: str) -> str:
        """Reject empty identifier values before persistence."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("identifier value must not be empty")
        return cleaned
