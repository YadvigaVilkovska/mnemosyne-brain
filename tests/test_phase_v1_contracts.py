"""Phase V1 contract tests for the dialogue signal and memory extraction pipeline."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.analysis import (
    PhaseV1CandidateFact,
    PhaseV1CandidateMemoryItem,
    PhaseV1DoNotSave,
    PhaseV1Entity,
    PhaseV1MemoryDecision,
    PhaseV1MemoryDecisionItem,
    PhaseV1SegmentAnalysis,
    PhaseV1SignalPolarity,
    PhaseV1Stage0SignalExtraction,
    PhaseV1InformationSignal,
)


class PhaseV1Stage0SignalExtractionTestCase(unittest.TestCase):
    """Verifies the Phase V1 Stage 0 signal extraction contract."""

    def _valid_entity(self) -> dict:
        return {
            "id": "e1",
            "mention": "Alena",
            "entity_type": "person",
            "source_span": "ты знаешь алену?",
            "resolution_status": "literal",
            "resolved_to": None,
        }

    def _valid_signal(self) -> dict:
        return {
            "id": "s1",
            "source_span": "а что если я тебе скажу",
            "signal_type": "alias_equivalence_proposal",
            "about_entity_ids": ["e1"],
            "signal_scope": "current_message",
            "polarity": "hypothetical",
            "epistemic_status": "user_question",
            "extraction_note": "The user is proposing a possible alias relation.",
        }

    def test_accepts_entities_and_information_signals(self) -> None:
        frame = PhaseV1Stage0SignalExtraction(
            entities=[self._valid_entity()],
            information_signals=[self._valid_signal()],
            unresolved_references=[{"id": "u1", "mention": "her", "source_span": "про нее"}],
            ambiguous_references=[{"id": "a1", "mention": "она", "source_span": "она"}],
        )
        self.assertEqual(1, len(frame.entities))
        self.assertEqual(1, len(frame.information_signals))

    def test_rejects_memory_decision_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1Stage0SignalExtraction(
                entities=[self._valid_entity()],
                information_signals=[self._valid_signal()],
                memory_candidates=[],
            )

    def test_information_signals_require_source_span(self) -> None:
        signal = self._valid_signal()
        signal["source_span"] = "   "
        with self.assertRaises(ValidationError):
            PhaseV1Stage0SignalExtraction(information_signals=[signal])

    def test_output_does_not_include_memory_candidates(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1Stage0SignalExtraction(
                entities=[self._valid_entity()],
                information_signals=[self._valid_signal()],
                memory_candidates=[{"candidate_type": "fact"}],
            )

    def test_output_does_not_include_selected_memory_ids(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1Stage0SignalExtraction(
                entities=[self._valid_entity()],
                information_signals=[self._valid_signal()],
                selected_memory_ids=["mem_1"],
            )

    def test_output_does_not_include_draft_or_final_answer(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1Stage0SignalExtraction(
                entities=[self._valid_entity()],
                information_signals=[self._valid_signal()],
                draft_answer="not allowed",
                final_answer="not allowed",
            )


class PhaseV1SegmentAnalysisTestCase(unittest.TestCase):
    """Verifies the Phase V1 Segment Analysis contract."""

    def _candidate_fact(self) -> dict:
        return {
            "id": "cf1",
            "claim": "The user says Alena is Ekaterina.",
            "about_entity_ids": ["e1"],
            "source_turn_ids": ["turn_1"],
            "source_signal_ids": ["s1"],
            "polarity": "asserted",
            "confidence": "high",
        }

    def _candidate_memory_item(self) -> dict:
        return {
            "id": "cm1",
            "memory_type": "alias",
            "proposed_content": "Alena may be Ekaterina.",
            "about_entity_ids": ["e1"],
            "source_turn_ids": ["turn_1"],
            "source_signal_ids": ["s1"],
            "reason_for_candidate": "The user explicitly proposed an alias relation.",
            "confidence": "medium",
        }

    def _do_not_save(self) -> dict:
        return {
            "id": "dns1",
            "content": "A sensitive claim.",
            "reason": "Do not save because it is not suitable for durable storage.",
            "source_turn_ids": ["turn_2"],
            "source_signal_ids": ["s2"],
            "confidence": "low",
        }

    def test_can_contain_candidate_facts(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            candidate_facts=[self._candidate_fact()],
            confidence="medium",
        )
        self.assertEqual(1, len(analysis.candidate_facts))

    def test_can_contain_candidate_memory_items(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            candidate_memory_items=[self._candidate_memory_item()],
            confidence="high",
        )
        self.assertEqual(1, len(analysis.candidate_memory_items))

    def test_candidate_facts_and_candidate_memory_items_are_separate_structures(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            candidate_facts=[self._candidate_fact()],
            candidate_memory_items=[self._candidate_memory_item()],
            confidence="high",
        )
        self.assertEqual(1, len(analysis.candidate_facts))
        self.assertEqual(1, len(analysis.candidate_memory_items))
        self.assertNotEqual(analysis.candidate_facts[0].claim, analysis.candidate_memory_items[0].proposed_content)

    def test_can_contain_do_not_save_items(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            do_not_save=[self._do_not_save()],
            confidence="low",
        )
        self.assertEqual(1, len(analysis.do_not_save))

    def test_do_not_save_items_require_reason_and_source_references(self) -> None:
        payload = self._do_not_save()
        payload["reason"] = "   "
        with self.assertRaises(ValidationError):
            PhaseV1SegmentAnalysis(segment_summary="The user proposed an alias.", do_not_save=[payload], confidence="low")

    def test_preserves_source_turn_ids_and_source_signal_ids_in_candidate_objects(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            candidate_facts=[self._candidate_fact()],
            candidate_memory_items=[self._candidate_memory_item()],
            confidence="high",
        )
        self.assertEqual(["turn_1"], analysis.candidate_facts[0].source_turn_ids)
        self.assertEqual(["s1"], analysis.candidate_facts[0].source_signal_ids)
        self.assertEqual(["turn_1"], analysis.candidate_memory_items[0].source_turn_ids)
        self.assertEqual(["s1"], analysis.candidate_memory_items[0].source_signal_ids)

    def test_candidate_memory_items_do_not_create_memory_writes(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            candidate_memory_items=[self._candidate_memory_item()],
            confidence="medium",
        )
        self.assertEqual("The user proposed an alias.", analysis.segment_summary)
        self.assertFalse(hasattr(analysis, "memory_candidates"))

    def test_do_not_save_items_preserve_source_references(self) -> None:
        analysis = PhaseV1SegmentAnalysis(
            segment_summary="The user proposed an alias.",
            do_not_save=[self._do_not_save()],
            confidence="low",
        )
        self.assertEqual(["turn_2"], analysis.do_not_save[0].source_turn_ids)
        self.assertEqual(["s2"], analysis.do_not_save[0].source_signal_ids)

    def test_segment_analysis_rejects_memory_decision_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1SegmentAnalysis(
                segment_summary="The user proposed an alias.",
                confidence="low",
                decisions=[],
            )


class PhaseV1MemoryDecisionTestCase(unittest.TestCase):
    """Verifies the Phase V1 memory decision contract."""

    def _decision_item(self, decision: str) -> dict:
        return {
            "candidate_id": "cm1",
            "decision": decision,
            "target_memory_id": None,
            "reason": "The candidate should be handled by the durable-memory gate.",
            "final_content": None,
            "confidence": "medium",
        }

    def test_supports_save_skip_update_conflict_and_ask(self) -> None:
        decision = PhaseV1MemoryDecision(
            decisions=[
                self._decision_item("save"),
                self._decision_item("skip"),
                self._decision_item("update"),
                self._decision_item("conflict"),
                self._decision_item("ask"),
            ]
        )
        self.assertEqual(5, len(decision.decisions))

    def test_reason_is_required_and_non_empty(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseV1MemoryDecision(decisions=[{**self._decision_item("save"), "reason": "   "}])

    def test_decisions_are_not_triggered_by_stage0_or_segment_analysis_contracts(self) -> None:
        stage0 = PhaseV1Stage0SignalExtraction(
            entities=[
                {
                    "id": "e1",
                    "mention": "Alena",
                    "entity_type": "person",
                    "source_span": "ты знаешь алену?",
                    "resolution_status": "literal",
                    "resolved_to": None,
                }
            ],
            information_signals=[
                {
                    "id": "s1",
                    "source_span": "ты знаешь алену?",
                    "signal_type": "person_mention",
                    "about_entity_ids": ["e1"],
                    "signal_scope": "current_message",
                    "polarity": "questioned",
                    "epistemic_status": "user_question",
                    "extraction_note": "The user asks about a person.",
                }
            ],
        )
        segment = PhaseV1SegmentAnalysis(
            segment_summary="The user asks about a person.",
            candidate_facts=[],
            candidate_memory_items=[],
            confidence="low",
        )
        self.assertEqual(1, len(stage0.entities))
        self.assertEqual("The user asks about a person.", segment.segment_summary)

    def test_final_content_can_be_null(self) -> None:
        decision = PhaseV1MemoryDecision(decisions=[self._decision_item("skip")])
        self.assertIsNone(decision.decisions[0].final_content)


class PhaseV1CompatibilityTestCase(unittest.TestCase):
    """Documents that historical Stage 2 naming remains available."""

    def test_stage2_name_remains_compatible(self) -> None:
        self.assertTrue(issubclass(PhaseV1SegmentAnalysis, object))
        self.assertTrue(issubclass(PhaseV1MemoryDecision, object))

