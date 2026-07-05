"""Shared test helpers."""

from __future__ import annotations

import sqlite3

from mnemosyne_brain.app.db.migrate import run_migrations
from mnemosyne_brain.app.db.repository import SqliteRepository
from mnemosyne_brain.app.graph.graph import build_graph


def create_test_repository() -> SqliteRepository:
    """Create a migrated temporary SQLite repository."""

    connection = sqlite3.connect(":memory:")
    run_migrations(connection)
    return SqliteRepository(connection)


class SpyGraph:
    """Graph spy that records invocation and transaction state."""

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {"response": "ok"}
        self.calls: list[dict] = []
        self.in_transaction_values: list[bool] = []

    def invoke(self, state: dict) -> dict:
        """Record graph input state."""

        self.calls.append(state)
        return self.response


def create_graph_repository_pair():
    """Create a repository and compiled graph."""

    repository = create_test_repository()
    return repository, build_graph(repository)
