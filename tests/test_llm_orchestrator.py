"""LLM orchestrator tests."""

from __future__ import annotations

import unittest
from typing import Any

from mnemosyne_brain.app.contracts.analysis import (
    PhaseV1Stage0SignalExtraction,
    Stage0NLUFrame,
    Stage1Decision,
    Stage2Decision,
)
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
        stage0_frame: Stage0NLUFrame | None = None,
        stage1_decision: Stage1Decision,
        stage2_decision: Stage2Decision | None = None,
    ) -> None:
        self.stage0_frame = stage0_frame or Stage0NLUFrame.model_validate(
            {
                "schema_version": "stage0_nlu_frame.v1",
                "normalized_intent": "The user asks for a direct answer.",
                "dialogue_acts": ["question"],
                "entities": [],
                "current_signal": {
                    "status": "none",
                    "kind": "none",
                    "summary": "",
                    "needs_confirmation": False,
                },
                "clarification": {
                    "needed": False,
                    "question": "",
                },
                "memory_selection_hint": {
                    "needed": False,
                    "reason": "",
                    "query_terms": [],
                },
            }
        )
        self.stage1_decision = stage1_decision
        self.stage2_decision = stage2_decision
        self.stage0_contexts: list[dict[str, Any]] = []
        self.stage1_contexts: list[dict[str, Any]] = []
        self.stage2_contexts: list[dict[str, Any]] = []

    def run_stage0_nlu(self, context: dict[str, Any]) -> Stage0NLUFrame:
        """Record Stage 0 context and return the configured frame."""

        self.stage0_contexts.append(context)
        return self.stage0_frame

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

    def _memory_update_extraction_for(self, memory_candidates: list[dict] | None = None) -> dict:
        candidates = memory_candidates or []
        return {
            "status": "ok" if candidates else "fail",
            "reason": "Durable information extracted." if candidates else "No durable information extracted.",
        }

    def _stage1_decision(self, **kwargs: Any) -> Stage1Decision:
        kwargs.setdefault("memory_update_extraction", self._memory_update_extraction_for(kwargs.get("memory_candidates")))
        return Stage1Decision(**kwargs)

    def _stage2_decision(self, **kwargs: Any) -> Stage2Decision:
        kwargs.setdefault("memory_update_extraction", self._memory_update_extraction_for(kwargs.get("memory_candidates")))
        return Stage2Decision(**kwargs)

    def test_answer_directly_returns_draft_answer_and_does_not_call_stage2(self) -> None:
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
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
        self.assertEqual(1, len(adapter.stage0_contexts))
        self.assertEqual(1, len(adapter.stage1_contexts))
        self.assertEqual([], adapter.stage2_contexts)
        self.assertIsNone(result["stage2_decision"])

    def test_request_memory_builds_stage2_and_returns_final_answer(self) -> None:
        memory_id = self._add_memory("Pav loves architecture diagrams")
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="request_memory",
                selected_memory_ids=[memory_id],
            ),
            stage2_decision=self._stage2_decision(
                final_answer="Pav loves architecture diagrams.",
                used_memory_ids=[memory_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(1, len(adapter.stage0_contexts))
        self.assertEqual("used_selected_memory", result["route"])
        self.assertEqual("Pav loves architecture diagrams.", result["answer"])
        self.assertEqual(1, len(adapter.stage2_contexts))
        self.assertEqual("stage2", adapter.stage2_contexts[0]["stage"])

    def test_selected_memory_ids_pass_from_stage1_into_stage2(self) -> None:
        first_id = self._add_memory("first")
        second_id = self._add_memory("second")
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="request_memory",
                selected_memory_ids=[second_id, first_id],
            ),
            stage2_decision=self._stage2_decision(final_answer="Done."),
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

    def test_run_stage0_nlu_is_called_before_stage1(self) -> None:
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(1, len(adapter.stage0_contexts))
        self.assertEqual(1, len(adapter.stage1_contexts))
        self.assertNotIn("stage0_nlu_frame", adapter.stage0_contexts[0])
        self.assertIn("stage0_nlu_frame", adapter.stage1_contexts[0])

    def test_stage1_context_receives_stage0_nlu_frame(self) -> None:
        stage0_frame = Stage0NLUFrame.model_validate(
            {
                "schema_version": "stage0_nlu_frame.v1",
                "normalized_intent": "The user asks whether an alias can be remembered.",
                "dialogue_acts": ["question", "alias_or_equivalence_proposal"],
                "entities": [
                    {
                        "surface": "X",
                        "kind": "alias",
                        "role": "subject",
                    }
                ],
                "current_signal": {
                    "status": "possible",
                    "kind": "alias_equivalence",
                    "summary": "Possible alias equivalence.",
                    "needs_confirmation": True,
                },
                "clarification": {
                    "needed": True,
                    "question": "Do you mean the same person?",
                },
                "memory_selection_hint": {
                    "needed": True,
                    "reason": "Potential identity lookup may help later.",
                    "query_terms": ["X", "Y"],
                },
            }
        )
        adapter = FakeLLMAdapter(
            stage0_frame=stage0_frame,
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            ),
        )
        DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(
            stage0_frame.model_dump(mode="json"),
            adapter.stage1_contexts[0]["stage0_nlu_frame"],
        )

    def test_used_memory_ids_are_returned_from_stage2(self) -> None:
        first_id = self._add_memory("first")
        second_id = self._add_memory("second")
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="request_memory",
                selected_memory_ids=[first_id, second_id],
            ),
            stage2_decision=self._stage2_decision(
                final_answer="Done.",
                used_memory_ids=[second_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual([second_id], result["used_memory_ids"])

    def test_final_answer_still_comes_from_stage1_or_stage2(self) -> None:
        direct_adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        direct_result = DeterministicLLMOrchestrator(self.repository, direct_adapter).run_turn(
            self.track.track_id,
            "direct message",
        )
        self.assertEqual("Direct answer.", direct_result["answer"])

        memory_id = self._add_memory("memory")
        memory_adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="request_memory",
                selected_memory_ids=[memory_id],
            ),
            stage2_decision=self._stage2_decision(
                final_answer="Stage 2 answer.",
                used_memory_ids=[memory_id],
            ),
        )
        memory_result = DeterministicLLMOrchestrator(self.repository, memory_adapter).run_turn(
            self.track.track_id,
            "memory message",
        )
        self.assertEqual("Stage 2 answer.", memory_result["answer"])

    def test_memory_candidates_still_come_only_from_stage1_or_stage2(self) -> None:
        adapter = FakeLLMAdapter(
            stage0_frame=Stage0NLUFrame.model_validate(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "Potential alias proposal.",
                    "dialogue_acts": ["alias_or_equivalence_proposal"],
                    "entities": [],
                    "current_signal": {
                        "status": "possible",
                        "kind": "alias_equivalence",
                        "summary": "Possible alias.",
                        "needs_confirmation": True,
                    },
                    "clarification": {
                        "needed": True,
                        "question": "Do you mean the same person?",
                    },
                    "memory_selection_hint": {
                        "needed": False,
                        "reason": "",
                        "query_terms": [],
                    },
                }
            ),
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
                memory_candidates=[{"candidate_type": "name_alias", "content": {"raw_name": "X"}}],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(
            [{"candidate_type": "name_alias", "content": {"raw_name": "X"}}],
            result["stage1_decision"]["memory_candidates"],
        )
        self.assertIsNone(result["stage2_decision"])

    def test_stage0_does_not_write_memory(self) -> None:
        before_candidates = self.repository.count_rows("memory_candidates")
        before_items = self.repository.count_rows("memory_items")
        adapter = FakeLLMAdapter(
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(before_candidates, self.repository.count_rows("memory_candidates"))
        self.assertEqual(before_items, self.repository.count_rows("memory_items"))

    def test_optional_phase_v1_current_signal_is_included_in_result(self) -> None:
        current_signal = PhaseV1Stage0SignalExtraction(
            entities=[
                {
                    "id": "e1",
                    "mention": "Lena",
                    "entity_type": "person",
                    "source_span": "ты знаешь лену",
                    "resolution_status": "literal",
                    "resolved_to": None,
                }
            ],
            information_signals=[
                {
                    "id": "s1",
                    "source_span": "ты знаешь лену",
                    "signal_type": "person_mention",
                    "about_entity_ids": ["e1"],
                    "signal_scope": "current_message",
                    "polarity": "questioned",
                    "epistemic_status": "user_question",
                    "extraction_note": "The user asks about a mentioned person.",
                }
            ],
            unresolved_references=[],
            ambiguous_references=[],
        )

        class AdapterWithCurrentSignal(FakeLLMAdapter):
            def run_phase_v1_stage0_signal_extraction(self, context: dict[str, Any]) -> PhaseV1Stage0SignalExtraction:
                self.stage0_contexts.append(context)
                return current_signal

        adapter = AdapterWithCurrentSignal(
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual(current_signal.model_dump(mode="json"), result["current_signal"])
        self.assertIsNone(result["current_signal_audit_error"])

    def test_invalid_optional_phase_v1_current_signal_becomes_non_fatal_audit_error(self) -> None:
        class AdapterWithInvalidCurrentSignal(FakeLLMAdapter):
            def run_phase_v1_stage0_signal_extraction(self, context: dict[str, Any]) -> dict[str, Any]:
                self.stage0_contexts.append(context)
                return {
                    "entities": [
                        {
                            "id": "e1",
                            "mention": "Lena",
                            "entity_type": "person",
                            "source_span": "",
                            "resolution_status": "literal",
                            "resolved_to": None,
                        }
                    ],
                    "information_signals": [],
                    "unresolved_references": [],
                    "ambiguous_references": [],
                }

        adapter = AdapterWithInvalidCurrentSignal(
            stage1_decision=self._stage1_decision(
                decision_type="answer_directly",
                draft_answer="Direct answer.",
            )
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertEqual("Direct answer.", result["answer"])
        self.assertIsNone(result["current_signal"])
        self.assertIn("source_span", result["current_signal_audit_error"])

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
            stage1_decision=self._stage1_decision(
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
            stage1_decision=self._stage1_decision(
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
            stage1_decision=self._stage1_decision(
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
            stage1_decision=self._stage1_decision(
                decision_type="request_memory",
                selected_memory_ids=[memory_id],
            ),
            stage2_decision=self._stage2_decision(
                final_answer="Done.",
                used_memory_ids=[memory_id],
            ),
        )
        result = DeterministicLLMOrchestrator(self.repository, adapter).run_turn(
            self.track.track_id,
            "test message",
        )
        self.assertFalse(self._contains_key(result, "summary"))
