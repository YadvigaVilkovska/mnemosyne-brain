"""Small coordinator for staged LLM context decisions."""

from __future__ import annotations

from typing import Any

from .context_builder import ContextBuilder
from .db.repository import SqliteRepository
from .llm_provider import LLMAdapter

ROUTE_ANSWER_DIRECTLY = "answer_directly"
ROUTE_USED_SELECTED_MEMORY = "used_selected_memory"


class DeterministicLLMOrchestrator:
    """Connect deterministic contexts to an injected LLM adapter."""

    def __init__(self, repository: SqliteRepository, adapter: LLMAdapter) -> None:
        self._context_builder = ContextBuilder(repository)
        self._adapter = adapter

    def run_turn(self, track_id: str, current_user_message: str) -> dict[str, Any]:
        """Run Stage 1 and, when requested, Stage 2 without mutating durable state."""

        stage1_context = self._context_builder.build_stage1_context(
            track_id=track_id,
            current_user_message=current_user_message,
        )
        stage1_decision = self._adapter.decide_stage1(stage1_context)
        if stage1_decision.decision_type == ROUTE_ANSWER_DIRECTLY:
            draft_answer = (stage1_decision.draft_answer or "").strip()
            if not draft_answer:
                raise ValueError("answer_directly decisions require a non-empty draft_answer")
            return {
                "route": ROUTE_ANSWER_DIRECTLY,
                "answer": draft_answer,
                "selected_memory_ids": [],
                "used_memory_ids": [],
                "stage1_decision": stage1_decision.model_dump(mode="json"),
                "stage2_decision": None,
            }

        stage2_context = self._context_builder.build_stage2_context(
            track_id=track_id,
            current_user_message=current_user_message,
            selected_memory_ids=stage1_decision.selected_memory_ids,
        )
        stage2_decision = self._adapter.decide_stage2(stage2_context)
        return {
            "route": ROUTE_USED_SELECTED_MEMORY,
            "answer": stage2_decision.final_answer,
            "selected_memory_ids": stage1_decision.selected_memory_ids,
            "used_memory_ids": stage2_decision.used_memory_ids,
            "stage1_decision": stage1_decision.model_dump(mode="json"),
            "stage2_decision": stage2_decision.model_dump(mode="json"),
        }
