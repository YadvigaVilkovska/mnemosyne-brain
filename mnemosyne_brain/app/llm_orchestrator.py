"""Small coordinator for staged LLM context decisions."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .contracts.analysis import PhaseV1Stage0SignalExtraction
from .context_builder import ContextBuilder
from .db.repository import SqliteRepository
from .llm_provider import LLMAdapter
from .stage0_current_signal_service import Stage0CurrentSignalService

ROUTE_ANSWER_DIRECTLY = "answer_directly"
ROUTE_USED_SELECTED_MEMORY = "used_selected_memory"


class DeterministicLLMOrchestrator:
    """Connect deterministic contexts to an injected LLM adapter."""

    def __init__(self, repository: SqliteRepository, adapter: LLMAdapter) -> None:
        self._context_builder = ContextBuilder(repository)
        self._adapter = adapter
        self._stage0_current_signal_service = Stage0CurrentSignalService()

    def run_turn(
        self,
        track_id: str,
        current_user_message: str,
        *,
        exclude_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Run Stage 1 and, when requested, Stage 2 without mutating durable state."""

        stage1_context = self._context_builder.build_stage1_context(
            track_id=track_id,
            current_user_message=current_user_message,
            exclude_turn_id=exclude_turn_id,
        )
        phase_v1_current_signal, phase_v1_current_signal_error = self._capture_phase_v1_current_signal(stage1_context)
        stage0_nlu_frame = self._adapter.run_stage0_nlu(dict(stage1_context))
        enriched_stage1_context = {
            **stage1_context,
            "stage0_nlu_frame": stage0_nlu_frame.model_dump(mode="json"),
        }
        stage1_decision = self._adapter.decide_stage1(enriched_stage1_context)
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
                "current_signal": phase_v1_current_signal,
                "current_signal_audit_error": phase_v1_current_signal_error,
            }

        stage2_context = self._context_builder.build_stage2_context(
            track_id=track_id,
            current_user_message=current_user_message,
            selected_memory_ids=stage1_decision.selected_memory_ids,
            exclude_turn_id=exclude_turn_id,
        )
        stage2_decision = self._adapter.decide_stage2(stage2_context)
        return {
            "route": ROUTE_USED_SELECTED_MEMORY,
            "answer": stage2_decision.final_answer,
            "selected_memory_ids": stage1_decision.selected_memory_ids,
            "used_memory_ids": stage2_decision.used_memory_ids,
            "stage1_decision": stage1_decision.model_dump(mode="json"),
            "stage2_decision": stage2_decision.model_dump(mode="json"),
            "current_signal": phase_v1_current_signal,
            "current_signal_audit_error": phase_v1_current_signal_error,
        }

    def _capture_phase_v1_current_signal(self, stage1_context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Collect optional Phase V1 Stage 0 signal data without affecting the answer path."""

        try:
            raw_signal = self._stage0_current_signal_service.extract_for_runtime(
                stage1_context,
                adapter=self._adapter,
            )
            if raw_signal is None:
                return None, None
            signal = (
                raw_signal
                if isinstance(raw_signal, PhaseV1Stage0SignalExtraction)
                else PhaseV1Stage0SignalExtraction.model_validate(raw_signal)
            )
            return signal.model_dump(mode="json"), None
        except (ValidationError, TypeError, ValueError) as error:
            return None, f"{error.__class__.__name__}: {error}"
        except Exception as error:
            # This hook is audit-only and must never block assistant answering.
            return None, f"{error.__class__.__name__}: {error}"
