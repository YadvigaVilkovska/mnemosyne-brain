"""Read-only CLI for inspecting deterministic LLM context payloads."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from .config import load_config
from .context_builder import ContextBuilder
from .db.migrate import run_migrations
from .db.repository import SqliteRepository

STAGE1_COMMAND = "stage1"
STAGE2_COMMAND = "stage2"


def build_parser() -> argparse.ArgumentParser:
    """Create the context debug parser with explicit stage subcommands."""

    parser = argparse.ArgumentParser(prog="python3 -m mnemosyne_brain.app.context_debug")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    stage1 = subparsers.add_parser(STAGE1_COMMAND)
    stage1.add_argument("--track-id", required=True)
    stage1.add_argument("--message", required=True)

    stage2 = subparsers.add_parser(STAGE2_COMMAND)
    stage2.add_argument("--track-id", required=True)
    stage2.add_argument("--message", required=True)
    stage2.add_argument("--memory-id", action="append", default=[])
    return parser


def build_context(args: argparse.Namespace, builder: ContextBuilder) -> dict[str, Any]:
    """Delegate the selected stage to ContextBuilder without mutating dialogue data."""

    if args.stage == STAGE1_COMMAND:
        return builder.build_stage1_context(
            track_id=args.track_id,
            current_user_message=args.message,
        )
    if args.stage == STAGE2_COMMAND:
        return builder.build_stage2_context(
            track_id=args.track_id,
            current_user_message=args.message,
            selected_memory_ids=args.memory_id,
        )
    raise ValueError(f"Unsupported context stage: {args.stage}")


def main(argv: Sequence[str] | None = None) -> int:
    """Print one deterministic LLM context payload as JSON."""

    args = build_parser().parse_args(argv)
    config = load_config()
    connection = sqlite3.connect(config.db_path)
    try:
        run_migrations(connection)
        repository = SqliteRepository(connection)
        context = build_context(args, ContextBuilder(repository))
        print(json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
