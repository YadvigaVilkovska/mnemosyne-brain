"""Stage 1 and Stage 0 contract tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.analysis import Stage0NLUFrame, Stage1Decision
from mnemosyne_brain.app.contracts.base import SCHEMA_VERSION


class Stage1DecisionTestCase(unittest.TestCase):
    """Verifies the future LLM Stage 1 decision contract."""

    def _fail_extraction(self) -> dict:
        return {"status": "fail", "reason": "No durable information extracted."}

    def _ok_extraction(self) -> dict:
        return {"status": "ok", "reason": "Durable information extracted."}

    def test_valid_answer_directly(self) -> None:
        decision = Stage1Decision(
            decision_type="answer_directly",
            draft_answer="Use a local answer.",
            memory_update_extraction=self._fail_extraction(),
        )
        self.assertEqual("answer_directly", decision.decision_type)
        self.assertEqual([], decision.selected_memory_ids)

    def test_valid_request_memory(self) -> None:
        decision = Stage1Decision(
            decision_type="request_memory",
            selected_memory_ids=["mem_1"],
            memory_update_extraction=self._fail_extraction(),
        )
        self.assertEqual(["mem_1"], decision.selected_memory_ids)

    def test_request_memory_without_selected_ids_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(decision_type="request_memory", memory_update_extraction=self._fail_extraction())

    def test_answer_directly_with_selected_ids_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                selected_memory_ids=["mem_1"],
                memory_update_extraction=self._fail_extraction(),
            )

    def test_duplicate_selected_ids_are_deduped_preserving_order(self) -> None:
        decision = Stage1Decision(
            decision_type="request_memory",
            selected_memory_ids=["mem_2", "mem_1", "mem_2", "mem_1"],
            memory_update_extraction=self._fail_extraction(),
        )
        self.assertEqual(["mem_2", "mem_1"], decision.selected_memory_ids)

    def test_extra_summary_field_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                memory_update_extraction=self._fail_extraction(),
                summary="not allowed",
            )

    def test_schema_version_exists(self) -> None:
        decision = Stage1Decision(decision_type="answer_directly", memory_update_extraction=self._fail_extraction())
        self.assertEqual(SCHEMA_VERSION, decision.schema_version)
        self.assertEqual("0.4.3", decision.schema_version)

    def test_explicit_stale_schema_version_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                schema_version="0.4.2",
                decision_type="answer_directly",
                memory_update_extraction=self._fail_extraction(),
            )

    def test_missing_memory_update_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(decision_type="answer_directly")

    def test_empty_memory_update_extraction_reason_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                memory_update_extraction={"status": "fail", "reason": "   "},
            )

    def test_empty_memory_candidates_with_fail_memory_update_extraction_passes(self) -> None:
        decision = Stage1Decision(
            decision_type="answer_directly",
            memory_update_extraction=self._fail_extraction(),
        )
        self.assertEqual("fail", decision.memory_update_extraction.status)

    def test_empty_memory_candidates_with_ok_memory_update_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                memory_update_extraction=self._ok_extraction(),
            )

    def test_non_empty_memory_candidates_with_ok_memory_update_extraction_passes(self) -> None:
        candidate = {"candidate_type": "fact", "content": {"text": "A durable fact."}}
        decision = Stage1Decision(
            decision_type="answer_directly",
            memory_candidates=[candidate],
            memory_update_extraction=self._ok_extraction(),
        )
        self.assertEqual([candidate], decision.memory_candidates)

    def test_non_empty_memory_candidates_with_fail_memory_update_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                memory_candidates=[{"candidate_type": "fact", "content": {"text": "A durable fact."}}],
                memory_update_extraction=self._fail_extraction(),
            )


class Stage0NLUFrameTestCase(unittest.TestCase):
    """Verifies the Stage 0 NLU frame contract."""

    def _valid_frame_payload(self) -> dict:
        return {
            "schema_version": "stage0_nlu_frame.v1",
            "normalized_intent": "The user asks whether an alias association can be used later.",
            "dialogue_acts": ["question", "alias_or_equivalence_proposal"],
            "entities": [
                {
                    "surface": "X",
                    "kind": "alias",
                    "role": "subject",
                }
            ],
            "new_information": {
                "status": "possible",
                "kind": "alias_equivalence",
                "summary": "The user may be proposing an alias equivalence.",
                "needs_confirmation": True,
            },
            "clarification": {
                "needed": True,
                "question": "Do you mean that X and Y refer to the same person?",
            },
            "memory_selection_hint": {
                "needed": False,
                "reason": "",
                "query_terms": [],
            },
        }

    def test_valid_stage0_nlu_frame_passes(self) -> None:
        frame = Stage0NLUFrame.model_validate(self._valid_frame_payload())
        self.assertEqual("stage0_nlu_frame.v1", frame.schema_version)
        self.assertEqual("possible", frame.new_information.status)

    def test_extra_top_level_field_fails(self) -> None:
        payload = self._valid_frame_payload() | {"unexpected": True}
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_empty_normalized_intent_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["normalized_intent"] = "   "
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_invalid_dialogue_act_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["dialogue_acts"] = ["not_allowed"]
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_invalid_entity_kind_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["entities"][0]["kind"] = "invalid"
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_invalid_new_information_status_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["new_information"]["status"] = "invalid"
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_invalid_new_information_kind_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["new_information"]["kind"] = "invalid"
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_clarification_needed_with_empty_question_fails(self) -> None:
        payload = self._valid_frame_payload()
        payload["clarification"]["question"] = ""
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_draft_answer_field_fails(self) -> None:
        payload = self._valid_frame_payload() | {"draft_answer": "not allowed"}
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_final_answer_field_fails(self) -> None:
        payload = self._valid_frame_payload() | {"final_answer": "not allowed"}
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_memory_candidates_field_fails(self) -> None:
        payload = self._valid_frame_payload() | {"memory_candidates": []}
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)

    def test_selected_memory_ids_field_fails(self) -> None:
        payload = self._valid_frame_payload() | {"selected_memory_ids": []}
        with self.assertRaises(ValidationError):
            Stage0NLUFrame.model_validate(payload)
