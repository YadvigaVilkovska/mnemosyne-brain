"""SQLite migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..contracts.base import server_now

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def run_migrations(connection: sqlite3.Connection) -> None:
    """Apply SQL migrations once, in filename order."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        )
        """
    )
    applied = {
        row[0] for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if migration_path.name in applied:
            continue
        connection.executescript(migration_path.read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
            (migration_path.name, server_now()),
        )
        connection.commit()
