"""LLM orchestrator tests."""

from __future__ import annotations

import unittest
from typing import Any

from mnemosyne_brain.app.contracts.analysis import Stage1Decision, Stage2Decision
from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.contracts.memory import MemoryCandidate
from mnemosyne_brain.app.contracts.provenance import Provenance
from mnemosyne_brain.app.llm_orchestrator import DeterministicLLMOrchestrator
from tests.support import create_test_repository


class FakeLLMAdapter:
    """Fake LLM adapter that records staged calls without network access."""

    def __init__(
        self,
        *,
        stage1_decision: Stage1Decision,
        stage2_decision: Stage2Decision | None = None,
    ) -> None:
        self.stage1_decision = stage1_decision
        self.stage2_decision = stage2_decision
        self.stage1_contexts: list[dict[str, Any]] = []
        self.stage2_contexts: list[dict[str, Any]] = []

    def decide_stage1(self, stage1_context: dict[str, Any]) -> Stage1Decision:
        """Record Stage 1 context and return the configured decision."""

        self.stage1_contexts.append(stage1_context)
        return self.stage1_decision

    def decide_stage2(self, stage2_context: dict[str, Any]) -> Stage2Decision:
        """Record Stage 2 context and return the configured decision."""

        self.stage2_contexts.append(stage2_context)
        if self.stage2_decision is None:
            raise AssertionError("Stage 2 should not have been called")
        return self.stage2_decision


class DeterministicLLMOrchestratorTestCase(unittest.TestCase):
    """Verifies staged orchestration without graph or provider side effects."""

    def setUp(self) -> None:
        self.repository = create_test_repository()
        with self.repository.transaction():
            self.track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_llm_orchestrator",
                thread_id="thread_llm_orchestrator",
                owner_user_id="user_llm_orchestrator",
            )

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

    def _contains_key(self, value: Any, key: str) -> bool:
        if isinstance(value, dict):
            return key in value or any(self._contains_key(item, key) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_key(item, key) for item in value)
        return False

    def test_answer_directly_returns_draft_answer_and_does_not_call_stage2(self) -> None:
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="answer_directly",
                draft_answer="  Direct local answer.  ",
            )
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual("answer_directly", result["route"])
        self.assertEqual("Direct local answer.", result["answer"])
        self.assertEqual([], result["selected_memory_ids"])
        self.assertEqual([], result["used_memory_ids"])
        self.assertEqual(1, len(adapter.stage1_contexts))
        self.assertEqual([], adapter.stage2_contexts)
        self.assertIsNone(result["stage2_decision"])

    def test_request_memory_builds_stage2_and_returns_final_answer(self) -> None:
        memory_id = self._add_memory("Pav loves architecture diagrams")
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="request_memory",
                selected_memory_ids=[memory_id],
            ),
            stage2_decision=Stage2Decision(
                final_answer="Pav loves architecture diagrams.",
                used_memory_ids=[memory_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual("used_selected_memory", result["route"])
        self.assertEqual("Pav loves architecture diagrams.", result["answer"])
        self.assertEqual(1, len(adapter.stage2_contexts))
        self.assertEqual("stage2", adapter.stage2_contexts[0]["stage"])

    def test_selected_memory_ids_pass_from_stage1_into_stage2(self) -> None:
        first_id = self._add_memory("first")
        second_id = self._add_memory("second")
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="request_memory",
                selected_memory_ids=[second_id, first_id],
            ),
            stage2_decision=Stage2Decision(final_answer="Done."),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        selected_context_ids = [
            item["memory_id"] for item in adapter.stage2_contexts[0]["selected_memory_context"]
        ]
        self.assertEqual([second_id, first_id], selected_context_ids)
        self.assertEqual([second_id, first_id], result["selected_memory_ids"])

    def test_used_memory_ids_are_returned_from_stage2(self) -> None:
        first_id = self._add_memory("first")
        second_id = self._add_memory("second")
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="request_memory",
                selected_memory_ids=[first_id, second_id],
            ),
            stage2_decision=Stage2Decision(
                final_answer="Done.",
                used_memory_ids=[second_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual([second_id], result["used_memory_ids"])

    def test_exclude_turn_id_reaches_context_builder_behavior(self) -> None:
        self.repository.persist_dialogue_turn(
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            thread_id=self.track.thread_id,
            input_source="user",
            role="user",
            content_text="previous message",
        )
        excluded_turn, _created = self.repository.persist_dialogue_turn(
            dialogue_id=self.track.dialogue_id,
            track_id=self.track.track_id,
            thread_id=self.track.thread_id,
            input_source="user",
            role="user",
            content_text="current persisted message",
        )
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "current persisted message",
            exclude_turn_id=excluded_turn.turn_id,
        )
        texts = [message["content_text"] for message in adapter.stage1_contexts[0]["recent_messages"]]
        self.assertEqual(["previous message"], texts)

    def test_empty_draft_answer_for_answer_directly_fails(self) -> None:
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="answer_directly",
                draft_answer="   ",
            )
        )
        with self.assertRaisesRegex(ValueError, "non-empty draft_answer"):
            DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
                self.track.track_id,
                "test message",
            )

    def test_orchestrator_does_not_mutate_dialogue_turns(self) -> None:
        before = self.repository.count_rows("dialogue_turns")
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(before, self.repository.count_rows("dialogue_turns"))

    def test_result_contains_no_summary_key(self) -> None:
        memory_id = self._add_memory("memory")
        adapter = FakeLLMAdapter(
            stage1_decision=Stage1Decision(
                decision_type="request_memory",
                selected_memory_ids=[memory_id],
            ),
            stage2_decision=Stage2Decision(
                final_answer="Done.",
                used_memory_ids=[memory_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertFalse(self._contains_key(result, "summary"))
