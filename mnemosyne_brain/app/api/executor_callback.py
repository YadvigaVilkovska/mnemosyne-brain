"""Pure executor callback service."""

from __future__ import annotations

from ..contracts.base import server_now
from ..contracts.executor import ExecutorCallback, ExecutorCallbackResult, ExecutorEvent
from ..contracts.routing import InputSource
from ..db.repository import SqliteRepository

FINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}


def handle_executor_callback(request_json: dict, *, db: SqliteRepository, graph) -> dict:
    """Persist executor callback before graph invocation and process idempotently."""

    callback = ExecutorCallback.model_validate(request_json)
    existing = db.find_executor_event(callback.event_id)
    if existing is not None:
        return {
            "status": ExecutorCallbackResult.DUPLICATE_ACCEPTED.value,
            "event_id": existing.event_id,
            "applied": existing.applied,
            "stale": existing.stale,
        }

    task = db.get_executor_task(callback.capsule_id)
    stale = callback.attempt < task.attempt_count and task.status in FINAL_TASK_STATUSES
    event = ExecutorEvent(
        **callback.model_dump(mode="json"),
        received_at=server_now(),
        applied=False,
        stale=stale,
        applied_at=None,
    )

    with db.transaction():
        db.insert_executor_event(event)
        if stale:
            db.insert_audit_event(
                event_type="executor_event_stale",
                actor_type="executor",
                actor_id=callback.executor,
                track_id=task.source_track_id,
                target_type="executor_event",
                target_id=event.event_id,
                payload={"capsule_id": task.capsule_id, "attempt": callback.attempt},
            )

    if stale:
        return {
            "status": ExecutorCallbackResult.ACCEPTED_STALE.value,
            "event_id": event.event_id,
            "applied": False,
            "stale": True,
        }

    result = graph.invoke(
        {
            "input_source": InputSource.EXECUTOR.value,
            "event_id": event.event_id,
            "capsule_id": event.capsule_id,
            "thread_id": task.thread_id,
        }
    )
    db.mark_executor_event_applied(event.event_id)
    return {
        "status": ExecutorCallbackResult.ACCEPTED.value,
        "event_id": event.event_id,
        "applied": True,
        "stale": False,
        "response": result.get("response"),
    }
