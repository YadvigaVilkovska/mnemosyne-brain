"""Command-line entrypoint for sending one local user message."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence

from .api.user_message import handle_user_message
from .config import load_config
from .contracts.base import new_id
from .db.migrate import run_migrations
from .db.repository import SqliteRepository
from .graph.graph import build_graph


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for one positional message."""

    parser = argparse.ArgumentParser(prog="python3 -m mnemosyne_brain.app.cli")
    parser.add_argument("message")
    return parser


def run_message(message: str) -> dict:
    """Send a single message through the existing graph and repository flow."""

    config = load_config()
    connection = sqlite3.connect(config.db_path)
    run_migrations(connection)
    repository = SqliteRepository(connection)
    graph = build_graph(repository)
    return handle_user_message(
        {
            "dialogue_id": new_id("dlg"),
            "thread_id": new_id("thread"),
            "external_message_id": new_id("msg"),
            "input_text": message,
        },
        graph=graph,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and print a concise user-facing result."""

    args = build_parser().parse_args(argv)
    result = run_message(args.message)
    print(f"Assistant: {result.get('response')}")
    print(f"Track: {result['track_id']}")
    print(f"Capsule: {result.get('capsule_id') or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
