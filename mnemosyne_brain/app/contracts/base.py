"""Base contracts and constants shared by all layers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.4.3"


def server_now() -> str:
    """Return a stable UTC timestamp string for persisted records."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Create an opaque text identifier with a domain prefix."""

    return f"{prefix}_{uuid4().hex}"


class StrictContractModel(BaseModel):
    """Base model that rejects accidental fields and mutable assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PersistedModel(StrictContractModel):
    """Base for objects that are durable business truth in SQLite."""

    schema_version: str = Field(default=SCHEMA_VERSION)


class ImmutableEventModel(PersistedModel):
    """Base for immutable event contracts."""


class EphemeralAnalysisModel(StrictContractModel):
    """Base for runtime-only analysis that must not become business truth."""


JsonValue = Any
