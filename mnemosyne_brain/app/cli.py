"""Command-line entrypoint for sending one local user message."""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Sequence

from .api.user_message import handle_user_message
from .config import load_config, load_project_env
from .contracts.base import new_id
from .db.migrate import run_migrations
from .db.repository import SqliteRepository
from .graph.graph import build_graph
from .llm_orchestrator import DeterministicLLMOrchestrator
from .llm_provider import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    OpenAICompatibleLLMProvider,
)

REQUIRED_LLM_ENV_VARS = (LLM_BASE_URL_ENV, LLM_API_KEY_ENV, LLM_MODEL_ENV)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for one positional message."""

    parser = argparse.ArgumentParser(prog="python3 -m mnemosyne_brain.app.cli")
    parser.add_argument("--thread-id")
    parser.add_argument("message")
    return parser


def run_message(message: str, thread_id: str | None = None) -> dict:
    """Send a single message through the existing graph and repository flow."""

    config = load_config()
    connection = sqlite3.connect(config.db_path)
    try:
        run_migrations(connection)
        repository = SqliteRepository(connection)
        if llm_env_is_configured():
            return run_llm_message(message, repository, thread_id=thread_id)
        return run_local_message(message, repository, thread_id=thread_id)
    finally:
        connection.close()


def llm_env_is_configured() -> bool:
    """Return true only when all required LLM provider variables are present."""

    load_project_env()
    return all(os.environ.get(name, "").strip() for name in REQUIRED_LLM_ENV_VARS)


def run_local_message(message: str, repository: SqliteRepository, thread_id: str | None = None) -> dict:
    """Preserve the existing graph-backed local fallback behavior."""

    graph = build_graph(repository)
    return handle_user_message(
        {
            "dialogue_id": new_id("dlg"),
            "thread_id": thread_id or new_id("thread"),
            "external_message_id": new_id("msg"),
            "input_text": message,
        },
        graph=graph,
    )


def run_llm_message(message: str, repository: SqliteRepository, thread_id: str | None = None) -> dict:
    """Run one message through the staged LLM orchestrator path."""

    dialogue_id = new_id("dlg")
    effective_thread_id = thread_id or new_id("thread")
    owner_user_id = new_id("user")
    with repository.transaction():
        track = repository.bootstrap_or_load_track(
            dialogue_id=dialogue_id,
            thread_id=effective_thread_id,
            owner_user_id=owner_user_id,
        )
        repository.persist_dialogue_turn(
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            thread_id=track.thread_id,
            input_source="user",
            role="user",
            content_text=message,
        )

    adapter = OpenAICompatibleLLMProvider.from_env()
    orchestrator = DeterministicLLMOrchestrator(repository, adapter)
    result = orchestrator.run_turn(track.track_id, message)
    with repository.transaction():
        assistant_turn, _created = repository.persist_dialogue_turn(
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            thread_id=track.thread_id,
            input_source="llm",
            role="assistant",
            content_text=result["answer"],
        )
    return {
        "dialogue_id": track.dialogue_id,
        "thread_id": track.thread_id,
        "track_id": track.track_id,
        "turn_id": assistant_turn.turn_id,
        "capsule_id": None,
        "response": result["answer"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and print a concise user-facing result."""

    args = build_parser().parse_args(argv)
    result = run_message(args.message, thread_id=args.thread_id)
    print(f"Assistant: {result.get('response')}")
    print(f"Track: {result['track_id']}")
    print(f"Capsule: {result.get('capsule_id') or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
