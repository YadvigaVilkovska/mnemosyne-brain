"""Memory dedupe checks."""

from __future__ import annotations

from ..db.repository import SqliteRepository


class MemoryDeduper:
    """Checks durable memory_items by dedupe key."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def find_conflicts(self, dedupe_key: str) -> list[str]:
        """Return conflicting active memory ids."""

        return self._repository.find_active_memory_by_dedupe_key(dedupe_key)
