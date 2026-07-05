"""Memory pipeline tests."""

from __future__ import annotations

import sqlite3
import unittest

from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.contracts.identity import IdentifierAssignment
from mnemosyne_brain.app.contracts.memory import MemoryCandidate
from mnemosyne_brain.app.contracts.provenance import Provenance
from mnemosyne_brain.app.memory.conflicts import ConflictResolver
from mnemosyne_brain.app.memory.dedupe import MemoryDeduper
from mnemosyne_brain.app.memory.staging import MemoryStagingService
from mnemosyne_brain.app.memory.write import MemoryWriter
from tests.support import create_test_repository


class MemoryWriteTestCase(unittest.TestCase):
    """Verifies idempotency, ConflictDecision and atomic writes."""

    def setUp(self) -> None:
        self.repository = create_test_repository()
        with self.repository.transaction():
            self.track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_test",
                thread_id="thread_test",
                owner_user_id="user_test",
            )
            self.turn, _ = self.repository.persist_dialogue_turn(
                dialogue_id="dlg_test",
                track_id=self.track.track_id,
                thread_id=self.track.thread_id,
                input_source="user",
                role="user",
                content_text="remember: Alice likes tea",
            )
        self.writer = MemoryWriter(
            self.repository,
            ConflictResolver(MemoryDeduper(self.repository)),
            MemoryStagingService(self.repository),
        )

    def _candidate(self, *, dedupe_key: str = "memory:tea", idempotency_key: str = "idem:tea") -> MemoryCandidate:
        now = server_now()
        return MemoryCandidate(
            candidate_id=new_id("cand"),
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            turn_id=self.turn.turn_id,
            candidate_type="fact",
            recommended_action="save_immediately",
            confidence=0.9,
            dedupe_key=dedupe_key,
            idempotency_key=idempotency_key,
            content_json={"text": "Alice likes tea"},
            provenance_json=Provenance(
                source="user",
                dialogue_id=self.track.dialogue_id,
                track_id=self.track.track_id,
                turn_id=self.turn.turn_id,
            ),
            created_at=now,
            updated_at=now,
        )

    def test_duplicate_memory_candidate_idempotency_key_does_not_create_second_candidate(self) -> None:
        candidate = self._candidate()
        self.repository.persist_memory_candidate(candidate)
        duplicate = self._candidate()
        _stored, created = self.repository.persist_memory_candidate(duplicate)
        self.assertFalse(created)
        self.assertEqual(1, self.repository.count_rows("memory_candidates"))

    def test_save_immediately_still_passes_through_conflict_decision(self) -> None:
        candidate, _created = self.repository.persist_memory_candidate(self._candidate())
        result = self.writer.handle_candidate_write(candidate)
        self.assertEqual("write_memory", result.decision_action)
        self.assertEqual(1, self.repository.count_rows("memory_items"))

    def test_identity_memory_and_audit_rollback_together(self) -> None:
        candidate, _created = self.repository.persist_memory_candidate(self._candidate())
        self.writer.handle_candidate_write(candidate)
        duplicate = self._candidate(dedupe_key="memory:other", idempotency_key="idem:other")
        duplicate = duplicate.model_copy(update={"content_json": {"text": "Alice likes tea"}})
        duplicate, _created = self.repository.persist_memory_candidate(duplicate)
        now = server_now()
        assignment = IdentifierAssignment(
            assignment_id=new_id("assign"),
            identifier_key="phone:15551002000",
            person_id="person_missing",
            resolution_status="resolved",
            status="active",
            confidence=1.0,
            provenance_json=Provenance(source="test"),
            created_at=now,
            updated_at=now,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.writer.handle_candidate_write(duplicate, assignment=assignment)
        self.assertEqual(1, self.repository.count_rows("memory_items"))
        self.assertEqual(1, self.repository.count_rows("audit_events"))
        self.assertEqual(0, self.repository.count_rows("identifier_assignments"))

    def test_json_columns_are_serialized_once(self) -> None:
        candidate, _created = self.repository.persist_memory_candidate(self._candidate())
        stored = self.repository.connection.execute(
            "SELECT content_json FROM memory_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()[0]
        self.assertEqual('{"text":"Alice likes tea"}', stored)
