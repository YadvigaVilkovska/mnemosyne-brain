"""Terminal executor and response graph nodes."""

from __future__ import annotations

from ...contracts.memory import TrackStatus
from ...contracts.routing import GraphRoute
from ...db.repository import SqliteRepository
from ...executors.hermes import HermesExecutor
from ..state import BrainGraphState


class RouterNode:
    """Select terminal route after user or executor processing."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        if state.get("route") == GraphRoute.ERROR_HANDLER.value:
            return {"router_decision": {"route": GraphRoute.ERROR_HANDLER.value, "reason": "error"}}
        if state.get("input_source") == "executor":
            return {
                "route": GraphRoute.LOCAL_ANSWER.value,
                "router_decision": {"route": GraphRoute.LOCAL_ANSWER.value, "reason": "executor_feedback"},
            }
        if state.get("turn_analysis", {}).get("should_call_executor"):
            return {
                "route": GraphRoute.CALL_EXECUTOR.value,
                "router_decision": {"route": GraphRoute.CALL_EXECUTOR.value, "reason": "delegation_requested"},
            }
        return {
            "route": GraphRoute.LOCAL_ANSWER.value,
            "router_decision": {"route": GraphRoute.LOCAL_ANSWER.value, "reason": "local_mvp"},
        }


class LocalAnswerNode:
    """Create a local answer without external calls."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        if state.get("response"):
            return {}
        return {"response": f"Local answer: {state.get('input_text', '')}"}


class CallExecutorNode:
    """Persist Hermes task and set the track waiting status."""

    def __init__(self, repository: SqliteRepository, hermes: HermesExecutor) -> None:
        self._repository = repository
        self._hermes = hermes

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        capsule = self._hermes.create_task(
            track_id=state["track_id"],
            thread_id=state["thread_id"],
            instruction=state.get("input_text", ""),
        )
        self._repository.update_track_status(state["track_id"], TrackStatus.WAITING_FOR_EXECUTOR)
        return {
            "capsule_id": capsule.capsule_id,
            "executor_task": {"capsule_id": capsule.capsule_id},
            "response": "Executor task accepted.",
        }


class AskClarificationNode:
    """Return a clarification answer."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        return {"response": "Please clarify your request."}


class CloseTrackNode:
    """Close the current track."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        self._repository.update_track_status(state["track_id"], TrackStatus.CLOSED)
        return {"response": "Track closed."}


class ErrorHandlerNode:
    """Return a local error response."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        return {"response": "Graph processing failed.", "errors": state.get("errors", [])}
