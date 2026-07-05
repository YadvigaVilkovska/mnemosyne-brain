"""Executor callback service tests."""

from __future__ import annotations

import unittest

from mnemosyne_brain.app.api.executor_callback import handle_executor_callback
from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.executors.hermes import HermesExecutor
from tests.support import SpyGraph, create_test_repository


class TransactionAwareGraph(SpyGraph):
    """Graph spy that records whether SQLite was in a transaction."""

    def __init__(self, repository) -> None:
        super().__init__({"response": "callback handled"})
        self.repository = repository

    def invoke(self, state: dict) -> dict:
        self.in_transaction_values.append(self.repository.connection.in_transaction)
        return super().invoke(state)


class ExecutorCallbackTestCase(unittest.TestCase):
    """Verifies callback idempotency and stale handling."""

    def setUp(self) -> None:
        self.repository = create_test_repository()
        with self.repository.transaction():
            track = self.repository.bootstrap_or_load_track(
                dialogue_id="dlg_exec",
                thread_id="thread_exec",
                owner_user_id="user_exec",
            )
        self.track = track
        self.capsule = HermesExecutor(self.repository).create_task(
            track_id=track.track_id,
            thread_id=track.thread_id,
            instruction="delegate: collect data",
        )

    def _request(self, event_id: str | None = None, attempt: int = 1) -> dict:
        return {
            "event_id": event_id or new_id("evt"),
            "capsule_id": self.capsule.capsule_id,
            "correlation_id": self.capsule.capsule_id,
            "executor": "hermes",
            "status": "success",
            "attempt": attempt,
            "is_final": True,
            "payload": {"answer": "Done"},
            "error": None,
            "artifacts": [],
            "created_at": "2026-07-05T12:00:00Z",
        }

    def test_duplicate_executor_event_returns_duplicate_accepted_and_does_not_invoke_graph(self) -> None:
        graph = SpyGraph()
        request = self._request("evt_duplicate")
        first = handle_executor_callback(request, db=self.repository, graph=graph)
        second = handle_executor_callback(request, db=self.repository, graph=graph)
        self.assertEqual("accepted", first["status"])
        self.assertEqual("duplicate_accepted", second["status"])
        self.assertEqual(1, len(graph.calls))
        self.assertEqual(1, self.repository.count_rows("executor_events"))

    def test_stale_executor_event_is_persisted_not_applied(self) -> None:
        self.repository.update_executor_task_status(self.capsule.capsule_id, "completed", final=True)
        graph = SpyGraph()
        result = handle_executor_callback(self._request(attempt=0), db=self.repository, graph=graph)
        event = self.repository.get_executor_event(result["event_id"])
        self.assertEqual("accepted_stale", result["status"])
        self.assertTrue(event.stale)
        self.assertFalse(event.applied)
        self.assertEqual([], graph.calls)

    def test_graph_invoke_is_not_called_inside_db_transaction(self) -> None:
        graph = TransactionAwareGraph(self.repository)
        handle_executor_callback(self._request(), db=self.repository, graph=graph)
        self.assertEqual([False], graph.in_transaction_values)
