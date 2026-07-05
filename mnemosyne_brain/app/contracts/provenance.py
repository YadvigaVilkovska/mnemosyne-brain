"""Provenance contracts for durable writes."""

from __future__ import annotations

from .base import StrictContractModel


class Provenance(StrictContractModel):
    """Records where a durable fact or event came from."""

    source: str
    dialogue_id: str | None = None
    track_id: str | None = None
    turn_id: str | None = None
    capsule_id: str | None = None
    event_id: str | None = None
