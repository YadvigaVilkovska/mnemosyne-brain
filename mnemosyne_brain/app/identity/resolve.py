"""Identity lookup services."""

from __future__ import annotations

from ..contracts.identity import IdentifierAssignment
from ..db.repository import SqliteRepository
from .normalize import IdentityNormalizer


class IdentityResolver:
    """Performs reverse and current identifier lookups through repository APIs."""

    def __init__(self, repository: SqliteRepository, normalizer: IdentityNormalizer) -> None:
        self._repository = repository
        self._normalizer = normalizer

    def resolve_by_phone(self, raw_phone: str) -> list[IdentifierAssignment]:
        """Normalize phone and resolve active assignments."""

        normalized = self._normalizer.normalize("phone", raw_phone)
        return self._repository.resolve_by_phone(normalized)

    def get_current_persona_phone(self, person_id: str, persona_id: str | None = None) -> str | None:
        """Return current active persona phone assignment."""

        return self._repository.get_current_persona_phone(person_id, persona_id)
