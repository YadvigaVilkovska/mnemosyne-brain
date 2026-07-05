"""Migration tests."""

from __future__ import annotations

import sqlite3
import unittest

from mnemosyne_brain.app.db.migrate import run_migrations

APPROVED_TABLES = {
    "schema_migrations",
    "dialogue_threads",
    "dialogue_turns",
    "dialogue_tracks_temp",
    "memory_candidates",
    "memory_staging",
    "memory_items",
    "persons",
    "personas",
    "identifiers",
    "identifier_assignments",
    "name_aliases",
    "executor_tasks",
    "executor_events",
    "audit_events",
}

UNAPPROVED_TABLES = {
    "dialogue_tracks",
    "memory_provenance",
    "conflict_decisions",
    "executor_task_capsules",
    "identity_profiles",
    "identity_mutation_audit",
}


class MigrationTestCase(unittest.TestCase):
    """Verifies migrations are idempotent and create the approved schema."""

    def test_migrations_apply_cleanly_on_empty_db(self) -> None:
        connection = sqlite3.connect(":memory:")
        run_migrations(connection)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        self.assertTrue(APPROVED_TABLES.issubset(tables))
        self.assertTrue(UNAPPROVED_TABLES.isdisjoint(tables))

    def test_migrations_are_idempotent(self) -> None:
        connection = sqlite3.connect(":memory:")
        run_migrations(connection)
        run_migrations(connection)
        count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(4, count)
