"""Local demo for the Mnemosyne Brain MVP."""

from __future__ import annotations

import sqlite3

from .api.executor_callback import handle_executor_callback
from .api.user_message import handle_user_message
from .config import load_config
from .contracts.base import new_id, server_now
from .db.migrate import run_migrations
from .db.repository import SqliteRepository
from .graph.graph import build_graph


def main() -> None:
    """Run the local vertical slice demo."""

    config = load_config()
    connection = sqlite3.connect(config.db_path)
    run_migrations(connection)
    repository = SqliteRepository(connection)
    graph = build_graph(repository)

    user_response = handle_user_message(
        {
            "dialogue_id": new_id("dlg"),
            "thread_id": new_id("thread"),
            "external_message_id": new_id("msg"),
            "input_text": "remember: Alice prefers green tea. delegate: collect context",
        },
        graph=graph,
    )
    capsule_id = user_response["capsule_id"]
    callback_response = handle_executor_callback(
        {
            "event_id": new_id("evt"),
            "capsule_id": capsule_id,
            "correlation_id": capsule_id,
            "executor": "hermes",
            "status": "success",
            "attempt": 1,
            "is_final": True,
            "payload": {"answer": "Context collected locally"},
            "error": None,
            "artifacts": [],
            "created_at": server_now(),
        },
        db=repository,
        graph=graph,
    )
    print(f"Local answer: {callback_response['response']}")
    print(f"DB path: {config.db_path}")
    print(f"Track: {user_response['track_id']}")
    print(f"Capsule: {capsule_id}")


if __name__ == "__main__":
    main()
