"""Executor ingress graph nodes."""

from __future__ import annotations

from ...db.repository import SqliteRepository
from ..state import BrainGraphState


class LoadPersistedExecutorEventNode:
    """Load executor event by id and keep only a ref in graph state."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        event = self._repository.get_executor_event(state["event_id"])
        return {
            "executor_event_ref": [{"event_id": event.event_id}],
            "capsule_id": event.capsule_id,
        }


class LoadTrackByCapsuleNode:
    """Load track using the executor capsule reference."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        track = self._repository.load_track_by_capsule(state["capsule_id"])
        return {
            "dialogue_id": track.dialogue_id,
            "track_id": track.track_id,
            "thread_id": track.thread_id,
            "track_ref": [{"track_id": track.track_id}],
            "track_snapshot": {"status": track.status.value},
        }


class ValidateExecutorEventNode:
    """Validate event/task consistency from persisted rows."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        event = self._repository.get_executor_event(state["event_id"])
        task = self._repository.get_executor_task(event.capsule_id)
        if task.executor != event.executor:
            return {"errors": ["executor mismatch"], "route": "error_handler"}
        return {}


class ExecutorFeedbackAnalyzerNode:
    """Analyze executor feedback without invoking turn analysis."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        event = self._repository.get_executor_event(state["event_id"])
        payload_text = event.payload if isinstance(event.payload, str) else self._repository.to_json(event.payload)
        return {
            "executor_feedback_analysis": {
                "event_id": event.event_id,
                "should_update_track": event.is_final,
                "local_answer": f"Executor result received: {payload_text}",
            }
        }


class ExecutorFeedbackHandlerNode:
    """Apply executor feedback to task/track state."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        event = self._repository.get_executor_event(state["event_id"])
        if event.is_final:
            self._repository.update_executor_task_status(event.capsule_id, "completed", final=True)
        return {"response": state["executor_feedback_analysis"]["local_answer"]}
