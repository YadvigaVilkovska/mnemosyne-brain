"""Context debug CLI tests."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr

from mnemosyne_brain.app.context_debug import main
from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.contracts.memory import MemoryCandidate
from mnemosyne_brain.app.contracts.provenance import Provenance
from mnemosyne_brain.app.db.migrate import run_migrations
from mnemosyne_brain.app.db.repository import SqliteRepository


class ContextDebugTestCase(unittest.TestCase):
    """Verifies the read-only LLM context debug entrypoint."""

    def setUp(self) -> None:
        self._previous_db_path = os.environ.get("MNEMOSYNE_DB_PATH")
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._temp_dir.name, "mnemosyne_context_debug.sqlite3")
        os.environ["MNEMOSYNE_DB_PATH"] = self.db_path
        self.repository = self._create_repository(self.db_path)
        with self.repository.transaction():
            self.track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_context_debug",
                thread_id="thread_context_debug",
                owner_user_id="user_context_debug",
            )

    def tearDown(self) -> None:
        self.repository.connection.close()
        self._temp_dir.cleanup()
        if self._previous_db_path is None:
            os.environ.pop("MNEMOSYNE_DB_PATH", None)
        else:
            os.environ["MNEMOSYNE_DB_PATH"] = self._previous_db_path

    def _create_repository(self, db_path: str) -> SqliteRepository:
        connection = sqlite3.connect(db_path)
        run_migrations(connection)
        return SqliteRepository(connection)

    def _run_debug(self, argv: list[str]) -> dict:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(argv)
        self.assertEqual(0, exit_code)
        return json.loads(output.getvalue())

    def _add_memory(self, text: str) -> str:
        now = server_now()
        turn, _created = self.repository.persist_dialogue_turn(
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            thread_id=self.track.thread_id,
            input_source="user",
            role="user",
            content_text=f"memory source {text}",
        )
        candidate = MemoryCandidate(
            candidate_id=new_id("cand"),
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            turn_id=turn.turn_id,
            candidate_type="fact",
            recommended_action="save_immediately",
            confidence=0.9,
            dedupe_key=self.repository.stable_key("memory", text),
            idempotency_key=self.repository.stable_key("candidate", text),
            content_json={"text": text, "title": text, "entity_type": "person"},
            provenance_json=Provenance(
                source="test",
                dialogue_id=self.track.dialogue_id,
                track_id=self.track.track_id,
                turn_id=turn.turn_id,
            ),
            created_at=now,
            updated_at=now,
        )
        self.repository.persist_memory_candidate(candidate)
        return self.repository.insert_memory_item(candidate)

    def test_stage1_outputs_valid_json(self) -> None:
        context = self._run_debug(
            [
                "stage1",
                "--track-id",
                self.track.track_id,
                "--message",
                "test message",
            ]
        )
        self.assertEqual("stage1", context["stage"])
        self.assertEqual("test message", context["current_user_message"])

    def test_stage2_outputs_valid_json(self) -> None:
        memory_id = self._add_memory("Pav loves architecture diagrams")
        context = self._run_debug(
            [
                "stage2",
                "--track-id",
                self.track.track_id,
                "--message",
                "test message",
                "--memory-id",
                memory_id,
            ]
        )
        self.assertEqual("stage2", context["stage"])
        self.assertEqual(memory_id, context["selected_memory_context"][0]["memory_id"])

    def test_stage1_does_not_mutate_dialogue_turns(self) -> None:
        before = self.repository.count_rows("dialogue_turns")
        self._run_debug(
            [
                "stage1",
                "--track-id",
                self.track.track_id,
                "--message",
                "test message",
            ]
        )
        self.assertEqual(before, self.repository.count_rows("dialogue_turns"))

    def test_stage2_uses_context_builder_dedupe_for_repeated_memory_ids(self) -> None:
        first_id = self._add_memory("first debug memory")
        second_id = self._add_memory("second debug memory")
        context = self._run_debug(
            [
                "stage2",
                "--track-id",
                self.track.track_id,
                "--message",
                "test message",
                "--memory-id",
                second_id,
                "--memory-id",
                first_id,
                "--memory-id",
                second_id,
            ]
        )
        selected_ids = [item["memory_id"] for item in context["selected_memory_context"]]
        self.assertEqual([second_id, first_id], selected_ids)

    def test_invalid_stage_fails_with_nonzero_exit(self) -> None:
        error_output = io.StringIO()
        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(["invalid", "--track-id", self.track.track_id, "--message", "test message"])
        self.assertNotEqual(0, raised.exception.code)
