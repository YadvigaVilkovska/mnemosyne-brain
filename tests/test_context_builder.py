"""LLM context builder policy tests."""

from __future__ import annotations

import unittest

from mnemosyne_brain.app.context_builder import (
    MEMORY_MANIFEST_MAX_ITEMS,
    SELECTED_MEMORY_MAX_ITEMS,
    ContextBuilder,
)
from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.contracts.executor import ExecutorEvent, ExecutorTaskCapsule
from mnemosyne_brain.app.contracts.memory import MemoryCandidate, TrackStatus
from mnemosyne_brain.app.contracts.provenance import Provenance
from tests.support import create_test_repository


class ContextBuilderTestCase(unittest.TestCase):
    """Verifies the deterministic LLM Context Policy v0.4.3."""

    def setUp(self) -> None:
        self.repository = create_test_repository()
        with self.repository.transaction():
            self.track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_context",
                thread_id="thread_context",
                owner_user_id="user_context",
            )
        self.builder = ContextBuilder(self.repository)

    def _add_turn(self, track_id: str, text: str, role: str = "user") -> None:
        track = self.repository.get_track(track_id)
        self.repository.persist_dialogue_turn(
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            thread_id=track.thread_id,
            input_source="user",
            role=role,
            content_text=text,
        )

    def _add_memory(self, text: str, *, title: str | None = None) -> str:
        now = server_now()
        turn, _created = self.repository.persist_dialogue_turn(
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            thread_id=self.track.thread_id,
            input_source="user",
            role="user",
            content_text=f"memory source {text}",
        )
        content = {"text": text, "title": title or text[:20], "entity_type": "person"}
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
            content_json=content,
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

    def _contains_key(self, value, key: str) -> bool:
        if isinstance(value, dict):
            return key in value or any(self._contains_key(item, key) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_key(item, key) for item in value)
        return False

    def test_stage1_includes_raw_current_user_message(self) -> None:
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="raw current message",
        )
        self.assertEqual("raw current message", context["current_user_message"])

    def test_stage1_includes_at_most_12_recent_messages(self) -> None:
        for index in range(15):
            self._add_turn(self.track.track_id, f"message {index:02d}")
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.assertEqual(12, len(context["recent_messages"]))

    def test_recent_messages_come_only_from_current_active_track(self) -> None:
        with self.repository.transaction():
            other_track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_other",
                thread_id="thread_other",
                owner_user_id="user_other",
            )
        self._add_turn(other_track.track_id, "other track message")
        self._add_turn(self.track.track_id, "current track message")
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        texts = [message["content_text"] for message in context["recent_messages"]]
        self.assertEqual(["current track message"], texts)

    def test_closed_track_tail_is_not_pulled_into_new_active_track(self) -> None:
        self._add_turn(self.track.track_id, "closed track message")
        self.repository.update_track_status(self.track.track_id, TrackStatus.CLOSED)
        with self.repository.transaction():
            new_track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_new",
                thread_id="thread_new",
                owner_user_id="user_new",
            )
        context = self.builder.build_stage1_context(
            track_id=new_track.track_id,
            current_user_message="new current",
        )
        self.assertEqual([], context["recent_messages"])

    def test_stage1_includes_previous_saved_analysis_if_present(self) -> None:
        self.repository.insert_audit_event(
            event_type="track_analysis_saved",
            actor_type="system",
            target_type="dialogue_track",
            target_id=self.track.track_id,
            payload={"intent": "architecture"},
        )
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.assertEqual({"intent": "architecture"}, context["previous_analysis"])

    def test_stage1_includes_pinned_exact_messages_separately(self) -> None:
        self.repository.insert_audit_event(
            event_type="pinned_exact_message",
            actor_type="system",
            target_type="dialogue_track",
            target_id=self.track.track_id,
            payload={"text": "exact quote"},
        )
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.assertEqual("exact quote", context["pinned_exact_messages"][0]["text"])

    def test_stage1_memory_manifest_does_not_contain_full_memory_content(self) -> None:
        full_text = "Pav loves architecture diagrams with detailed system boundaries"
        memory_id = self._add_memory(full_text)
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        manifest_item = context["memory_manifest"][0]
        self.assertEqual(memory_id, manifest_item["memory_id"])
        self.assertNotIn("content", manifest_item)
        self.assertNotIn("content_json", manifest_item)

    def test_memory_manifest_is_bounded(self) -> None:
        for index in range(MEMORY_MANIFEST_MAX_ITEMS + 5):
            self._add_memory(f"memory {index}")
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.assertEqual(MEMORY_MANIFEST_MAX_ITEMS, len(context["memory_manifest"]))

    def test_stage2_includes_only_selected_validated_memory_ids(self) -> None:
        first_id = self._add_memory("first selected")
        second_id = self._add_memory("second unselected")
        context = self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=[first_id],
        )
        selected_ids = [item["memory_id"] for item in context["selected_memory_context"]]
        self.assertEqual([first_id], selected_ids)
        self.assertNotIn(second_id, selected_ids)

    def test_duplicate_selected_ids_are_deduped_preserving_order(self) -> None:
        first_id = self._add_memory("first")
        second_id = self._add_memory("second")
        context = self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=[second_id, first_id, second_id],
        )
        selected_ids = [item["memory_id"] for item in context["selected_memory_context"]]
        self.assertEqual([second_id, first_id], selected_ids)

    def test_unknown_selected_ids_are_reported_safely(self) -> None:
        valid_id = self._add_memory("valid")
        context = self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=["missing_memory", valid_id],
        )
        self.assertEqual(["missing_memory"], context["rejected_memory_ids"])

    def test_selected_memory_limit_is_enforced(self) -> None:
        memory_ids = [self._add_memory(f"selected {index}") for index in range(SELECTED_MEMORY_MAX_ITEMS + 3)]
        context = self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=memory_ids,
        )
        self.assertEqual(SELECTED_MEMORY_MAX_ITEMS, len(context["selected_memory_context"]))
        self.assertEqual(memory_ids[SELECTED_MEMORY_MAX_ITEMS:], context["rejected_memory_ids"])

    def test_summary_key_in_context_payload_raises_value_error(self) -> None:
        memory_id = self._add_memory("memory with summary")
        self.repository.insert_audit_event(
            event_type="track_analysis_saved",
            actor_type="system",
            target_type="dialogue_track",
            target_id=self.track.track_id,
            payload={"summary": "not allowed", "signal": "allowed"},
        )
        with self.assertRaises(ValueError):
            self.builder.build_stage2_context(
                track_id=self.track.track_id,
                current_user_message="current",
                selected_memory_ids=[memory_id],
            )

    def test_valid_context_does_not_contain_summary_key(self) -> None:
        memory_id = self._add_memory("memory without forbidden key")
        context = self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=[memory_id],
        )
        self.assertFalse(self._contains_key(context, "summary"))

    def test_overflow_drops_oldest_prior_dialogue_messages_first(self) -> None:
        for index in range(15):
            self._add_turn(self.track.track_id, f"message {index:02d} " + ("x" * 3000))
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current message is retained",
        )
        texts = [message["content_text"] for message in context["recent_messages"]]
        self.assertEqual("current message is retained", context["current_user_message"])
        self.assertNotIn("message 03 " + ("x" * 3000), texts)
        self.assertIn("message 14 " + ("x" * 3000), texts)

    def test_builder_does_not_mutate_stored_dialogue_turns(self) -> None:
        self._add_turn(self.track.track_id, "stored")
        before = self.repository.count_rows("dialogue_turns")
        self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.builder.build_stage2_context(
            track_id=self.track.track_id,
            current_user_message="current",
            selected_memory_ids=[],
        )
        self.assertEqual(before, self.repository.count_rows("dialogue_turns"))

    def test_raw_executor_event_is_not_treated_as_user_turn(self) -> None:
        now = server_now()
        capsule = {
            "capsule_id": "cap_context",
            "schema_version": "0.4.2",
            "source_track_id": self.track.track_id,
            "thread_id": self.track.thread_id,
            "executor": "hermes",
            "status": "queued",
            "idempotency_key": "cap_context_idem",
            "attempt_count": 0,
            "capsule_json": {"instruction": "do work"},
            "created_at": now,
            "updated_at": now,
        }
        self.repository.insert_executor_task(ExecutorTaskCapsule(**capsule))
        self.repository.insert_executor_event(
            ExecutorEvent(
                event_id="evt_context",
                capsule_id="cap_context",
                correlation_id="cap_context",
                executor="hermes",
                status="success",
                attempt=1,
                is_final=True,
                payload={"raw": "executor payload"},
                error=None,
                artifacts=[],
                created_at=now,
                received_at=now,
            )
        )
        context = self.builder.build_stage1_context(
            track_id=self.track.track_id,
            current_user_message="current",
        )
        self.assertEqual([], context["recent_messages"])
