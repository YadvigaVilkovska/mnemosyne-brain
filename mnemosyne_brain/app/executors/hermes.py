"""Minimal Hermes executor adapter."""

from __future__ import annotations

from ..contracts.base import new_id, server_now
from ..contracts.executor import ExecutorTaskCapsule
from ..db.repository import SqliteRepository

HERMES_EXECUTOR_NAME = "hermes"


class HermesExecutor:
    """Creates Hermes task capsules without mutating memory or answering users."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def create_task(self, *, track_id: str, thread_id: str, instruction: str) -> ExecutorTaskCapsule:
        """Persist a Hermes task capsule for later callback processing."""

        now = server_now()
        idempotency_key = self._repository.stable_key(HERMES_EXECUTOR_NAME, track_id, instruction)
        capsule = ExecutorTaskCapsule(
            capsule_id=new_id("cap"),
            source_track_id=track_id,
            thread_id=thread_id,
            executor=HERMES_EXECUTOR_NAME,
            status="queued",
            idempotency_key=idempotency_key,
            attempt_count=0,
            capsule_json={"instruction": instruction},
            created_at=now,
            updated_at=now,
        )
        capsule, _created = self._repository.insert_executor_task(capsule)
        return capsule

    def enqueue_after_commit(self, capsule: ExecutorTaskCapsule) -> None:
        """Placeholder extension point for future outbox_events production pattern."""

        # TODO: Replace this no-op with an outbox_events producer after queues are approved.
        _ = capsule
