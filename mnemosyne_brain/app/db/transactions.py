"""SQLite transaction boundaries."""

from __future__ import annotations

import sqlite3


class SqliteUnitOfWork:
    """Explicit transaction wrapper that rolls back failed atomic writes."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> "SqliteUnitOfWork":
        self._connection.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._connection.commit()
            return
        self._connection.rollback()
