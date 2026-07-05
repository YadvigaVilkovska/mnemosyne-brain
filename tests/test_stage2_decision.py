"""Stage 2 decision contract tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.analysis import Stage2Decision
from mnemosyne_brain.app.contracts.base import SCHEMA_VERSION


class Stage2DecisionTestCase(unittest.TestCase):
    """Verifies the future LLM Stage 2 final decision contract."""

    def test_valid_final_answer(self) -> None:
        decision = Stage2Decision(final_answer="Architecture diagrams help Pav.")
        self.assertEqual("Architecture diagrams help Pav.", decision.final_answer)

    def test_final_answer_empty_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="   ")

    def test_used_memory_ids_are_deduped_preserving_order(self) -> None:
        decision = Stage2Decision(
            final_answer="Done.",
            used_memory_ids=["mem_2", "mem_1", "mem_2", "mem_3", "mem_1"],
        )
        self.assertEqual(["mem_2", "mem_1", "mem_3"], decision.used_memory_ids)

    def test_extra_summary_field_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="Done.", summary="not allowed")

    def test_schema_version_exists(self) -> None:
        decision = Stage2Decision(final_answer="Done.")
        self.assertEqual(SCHEMA_VERSION, decision.schema_version)

    def test_extracted_facts_default_to_empty_list(self) -> None:
        decision = Stage2Decision(final_answer="Done.")
        self.assertEqual([], decision.extracted_facts)

    def test_memory_candidates_default_to_empty_list(self) -> None:
        decision = Stage2Decision(final_answer="Done.")
        self.assertEqual([], decision.memory_candidates)
