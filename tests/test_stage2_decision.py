"""Stage 2 decision contract tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.analysis import Stage2Decision
from mnemosyne_brain.app.contracts.base import SCHEMA_VERSION


class Stage2DecisionTestCase(unittest.TestCase):
    """Verifies the future LLM Stage 2 final decision contract."""

    def _fail_extraction(self) -> dict:
        return {"status": "fail", "reason": "No durable information extracted."}

    def _ok_extraction(self) -> dict:
        return {"status": "ok", "reason": "Durable information extracted."}

    def test_valid_final_answer(self) -> None:
        decision = Stage2Decision(final_answer="Provider returned a final answer.", news_extraction=self._fail_extraction())
        self.assertEqual("Provider returned a final answer.", decision.final_answer)

    def test_final_answer_empty_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="   ", news_extraction=self._fail_extraction())

    def test_used_memory_ids_are_deduped_preserving_order(self) -> None:
        decision = Stage2Decision(
            final_answer="Done.",
            news_extraction=self._fail_extraction(),
            used_memory_ids=["mem_2", "mem_1", "mem_2", "mem_3", "mem_1"],
        )
        self.assertEqual(["mem_2", "mem_1", "mem_3"], decision.used_memory_ids)

    def test_extra_summary_field_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="Done.", news_extraction=self._fail_extraction(), summary="not allowed")

    def test_schema_version_exists(self) -> None:
        decision = Stage2Decision(final_answer="Done.", news_extraction=self._fail_extraction())
        self.assertEqual(SCHEMA_VERSION, decision.schema_version)
        self.assertEqual("0.4.3", decision.schema_version)

    def test_explicit_stale_schema_version_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(
                schema_version="0.4.2",
                final_answer="Done.",
                news_extraction=self._fail_extraction(),
            )

    def test_missing_news_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="Done.")

    def test_empty_news_extraction_reason_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(
                final_answer="Done.",
                news_extraction={"status": "fail", "reason": "   "},
            )

    def test_empty_memory_candidates_with_fail_news_extraction_passes(self) -> None:
        decision = Stage2Decision(final_answer="Done.", news_extraction=self._fail_extraction())
        self.assertEqual("fail", decision.news_extraction.status)

    def test_empty_memory_candidates_with_ok_news_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(final_answer="Done.", news_extraction=self._ok_extraction())

    def test_non_empty_memory_candidates_with_ok_news_extraction_passes(self) -> None:
        candidate = {"candidate_type": "fact", "content": {"text": "A durable fact."}}
        decision = Stage2Decision(
            final_answer="Done.",
            memory_candidates=[candidate],
            news_extraction=self._ok_extraction(),
        )
        self.assertEqual([candidate], decision.memory_candidates)

    def test_non_empty_memory_candidates_with_fail_news_extraction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            Stage2Decision(
                final_answer="Done.",
                memory_candidates=[{"candidate_type": "fact", "content": {"text": "A durable fact."}}],
                news_extraction=self._fail_extraction(),
            )

    def test_extracted_facts_default_to_empty_list(self) -> None:
        decision = Stage2Decision(final_answer="Done.", news_extraction=self._fail_extraction())
        self.assertEqual([], decision.extracted_facts)

    def test_memory_candidates_default_to_empty_list(self) -> None:
        decision = Stage2Decision(final_answer="Done.", news_extraction=self._fail_extraction())
        self.assertEqual([], decision.memory_candidates)
