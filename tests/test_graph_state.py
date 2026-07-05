"""Graph state tests."""

from __future__ import annotations

import unittest

from mnemosyne_brain.app.api.executor_callback import handle_executor_callback
from mnemosyne_brain.app.api.user_message import handle_user_message
from mnemosyne_brain.app.contracts.memory import TrackStatus
from mnemosyne_brain.app.executors.hermes import HermesExecutor
from mnemosyne_brain.app.graph.state import dedupe_ref_dicts
from tests.support import create_graph_repository_pair


class GraphStateTestCase(unittest.TestCase):
    """Verifies reference-only state and route behavior."""

    def test_dedupe_reducer_rejects_payload_shaped_dicts(self) -> None:
        with self.assertRaises(ValueError):
            dedupe_ref_dicts([], [{"event_id": "evt_1", "payload": "not allowed"}])

    def test_duplicate_external_message_id_does_not_create_second_turn(self) -> None:
        repository, graph = create_graph_repository_pair()
        request = {
            "dialogue_id": "dlg_user",
            "thread_id": "thread_user",
            "external_message_id": "msg_1",
            "input_text": "hello",
        }
        handle_user_message(request, graph=graph)
        handle_user_message(request, graph=graph)
        self.assertEqual(1, repository.count_rows("dialogue_turns"))

    def test_call_executor_sets_track_waiting_and_ends_route(self) -> None:
        repository, graph = create_graph_repository_pair()
        response = handle_user_message(
            {
                "dialogue_id": "dlg_exec_route",
                "thread_id": "thread_exec_route",
                "external_message_id": "msg_delegate",
                "input_text": "remember: use tea preference. delegate: collect context",
            },
            graph=graph,
        )
        track = repository.get_track(response["track_id"])
        self.assertEqual(TrackStatus.WAITING_FOR_EXECUTOR, track.status)
        self.assertIsNotNone(response["capsule_id"])

    def test_executor_callback_does_not_call_turn_analyzer(self) -> None:
        repository, graph = create_graph_repository_pair()
        with repository.transaction():
            track = repository.bootstrap_or_load_track(
                dialogue_id="dlg_callback",
                thread_id="thread_callback",
                owner_user_id="user_callback",
            )
        capsule = HermesExecutor(repository).create_task(
            track_id=track.track_id,
            thread_id=track.thread_id,
            instruction="delegate",
        )
        result = handle_executor_callback(
            {
                "event_id": "evt_callback",
                "capsule_id": capsule.capsule_id,
                "correlation_id": capsule.capsule_id,
                "executor": "hermes",
                "status": "success",
                "attempt": 1,
                "is_final": True,
                "payload": {"answer": "done"},
                "error": None,
                "artifacts": [],
                "created_at": "2026-07-05T12:00:00Z",
            },
            db=repository,
            graph=graph,
        )
        event = repository.get_executor_event(result["event_id"])
        self.assertTrue(event.applied)
        self.assertEqual(0, repository.count_rows("dialogue_turns"))

    def test_langgraph_state_stores_refs_not_executor_payload(self) -> None:
        repository, graph = create_graph_repository_pair()
        with repository.transaction():
            track = repository.bootstrap_or_load_track(
                dialogue_id="dlg_state",
                thread_id="thread_state",
                owner_user_id="user_state",
            )
        capsule = HermesExecutor(repository).create_task(
            track_id=track.track_id,
            thread_id=track.thread_id,
            instruction="delegate",
        )
        handle_executor_callback(
            {
                "event_id": "evt_state",
                "capsule_id": capsule.capsule_id,
                "correlation_id": capsule.capsule_id,
                "executor": "hermes",
                "status": "success",
                "attempt": 1,
                "is_final": True,
                "payload": {"secret": "payload must stay in sqlite"},
                "error": None,
                "artifacts": [],
                "created_at": "2026-07-05T12:00:00Z",
            },
            db=repository,
            graph=graph,
        )
        stored = repository.connection.execute(
            "SELECT payload_json FROM executor_events WHERE event_id = 'evt_state'"
        ).fetchone()[0]
        self.assertEqual('{"secret":"payload must stay in sqlite"}', stored)
