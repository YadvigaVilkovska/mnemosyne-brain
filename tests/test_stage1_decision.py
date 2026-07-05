"""Stage 1 decision contract tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.analysis import Stage1Decision
from mnemosyne_brain.app.contracts.base import SCHEMA_VERSION


class Stage1DecisionTestCase(unittest.TestCase):
    """Verifies the future LLM Stage 1 decision contract."""

    def test_valid_answer_directly(self) -> None:
        decision = Stage1Decision(
            decision_type="answer_directly",
            draft_answer="Use a local answer.",
        )
        self.assertEqual("answer_directly", decision.decision_type)
        self.assertEqual([], decision.selected_memory_ids)

    def test_valid_request_memory(self) -> None:
        decision = Stage1Decision(
            decision_type="request_memory",
            selected_memory_ids=["mem_1"],
        )
        self.assertEqual(["mem_1"], decision.selected_memory_ids)

    def test_request_memory_without_selected_ids_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(decision_type="request_memory")

    def test_answer_directly_with_selected_ids_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                selected_memory_ids=["mem_1"],
            )

    def test_duplicate_selected_ids_are_deduped_preserving_order(self) -> None:
        decision = Stage1Decision(
            decision_type="request_memory",
            selected_memory_ids=["mem_2", "mem_1", "mem_2", "mem_1"],
        )
        self.assertEqual(["mem_2", "mem_1"], decision.selected_memory_ids)

    def test_extra_summary_field_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage1Decision(
                decision_type="answer_directly",
                summary="not allowed",
            )

    def test_schema_version_exists(self) -> None:
        decision = Stage1Decision(decision_type="answer_directly")
        self.assertEqual(SCHEMA_VERSION, decision.schema_version)
