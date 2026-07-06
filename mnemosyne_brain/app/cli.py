"""Command-line entrypoint for sending one local user message."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from .api.user_message import handle_user_message
from .config import load_config, load_project_env
from .contracts.analysis import PhaseV1Stage0SignalExtraction
from .contracts.base import new_id, server_now
from .contracts.memory import MemoryCandidate
from .contracts.provenance import Provenance
from .db.migrate import run_migrations
from .db.repository import SqliteRepository
from .graph.graph import build_graph
from .llm_orchestrator import DeterministicLLMOrchestrator
from .llm_provider import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    OpenAICompatibleLLMProvider,
    ProviderConfigError,
    ProviderResponseError,
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
        user_turn, _created = repository.persist_dialogue_turn(
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            thread_id=track.thread_id,
            input_source="user",
            role="user",
            content_text=message,
        )

    adapter = OpenAICompatibleLLMProvider.from_env()
    orchestrator = DeterministicLLMOrchestrator(repository, adapter)
    result = orchestrator.run_turn(track.track_id, message, exclude_turn_id=user_turn.turn_id)
    analysis_payload = build_llm_analysis_payload(result)
    with repository.transaction():
        assistant_turn, _created = repository.persist_dialogue_turn(
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            thread_id=track.thread_id,
            input_source="llm",
            role="assistant",
            content_text=result["answer"],
        )
        repository.insert_audit_event(
            event_type="track_analysis_saved",
            actor_type="llm",
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            turn_id=assistant_turn.turn_id,
            target_type="dialogue_track",
            target_id=track.track_id,
            payload=analysis_payload,
        )
        for candidate in build_llm_memory_candidates(
            result,
            repository=repository,
            dialogue_id=track.dialogue_id,
            track_id=track.track_id,
            turn_id=user_turn.turn_id,
        ):
            repository.persist_memory_candidate(candidate)
    return {
        "dialogue_id": track.dialogue_id,
        "thread_id": track.thread_id,
        "track_id": track.track_id,
        "turn_id": assistant_turn.turn_id,
        "capsule_id": None,
        "response": result["answer"],
    }


def build_llm_memory_candidates(
    result: dict,
    *,
    repository: SqliteRepository,
    dialogue_id: str,
    track_id: str,
    turn_id: str,
) -> list[MemoryCandidate]:
    """Convert valid raw LLM memory candidates into durable candidate rows."""

    candidates: list[MemoryCandidate] = []
    seen_dedupe_keys = list_existing_track_candidate_dedupe_keys(repository, track_id)
    for raw_candidate in iter_raw_llm_memory_candidates(result):
        candidate_type = raw_candidate.get("candidate_type")
        content_json = raw_candidate.get("content")
        confidence = raw_candidate.get("confidence")
        if not isinstance(candidate_type, str) or not candidate_type.strip():
            continue
        if not isinstance(content_json, dict):
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        if not 0.0 <= float(confidence) <= 1.0:
            continue

        recommended_action = raw_candidate.get("recommended_action")
        if recommended_action not in {"stage", "save_immediately"}:
            recommended_action = "stage"

        normalized_type = candidate_type.strip()
        dedupe_key = repository.stable_key("llm_memory_candidate", normalized_type, content_json)
        if dedupe_key in seen_dedupe_keys:
            continue
        seen_dedupe_keys.add(dedupe_key)
        now = server_now()
        candidates.append(
            MemoryCandidate(
                candidate_id=new_id("cand"),
                dialogue_id=dialogue_id,
                track_id=track_id,
                turn_id=turn_id,
                candidate_type=normalized_type,
                recommended_action=recommended_action,
                confidence=float(confidence),
                dedupe_key=dedupe_key,
                idempotency_key=repository.stable_key(
                    "llm_memory_candidate",
                    track_id,
                    turn_id,
                    normalized_type,
                    content_json,
                ),
                content_json=content_json,
                provenance_json=Provenance(
                    source="llm",
                    dialogue_id=dialogue_id,
                    track_id=track_id,
                    turn_id=turn_id,
                ),
                created_at=now,
                updated_at=now,
            )
        )
    return candidates


def list_existing_track_candidate_dedupe_keys(repository: SqliteRepository, track_id: str) -> set[str]:
    """Return existing semantic candidate dedupe keys for one track."""

    rows = repository.connection.execute(
        """
        SELECT dedupe_key
        FROM memory_candidates
        WHERE track_id = ?
        """,
        (track_id,),
    ).fetchall()
    return {
        row[0]
        for row in rows
        if row and isinstance(row[0], str) and row[0]
    }


def iter_raw_llm_memory_candidates(result: dict) -> list[dict]:
    """Return raw LLM memory candidates from Stage 1 and Stage 2 in order."""

    raw_candidates: list[dict] = []
    stage1_decision = result["stage1_decision"]
    raw_candidates.extend(
        candidate
        for candidate in stage1_decision.get("memory_candidates", [])
        if isinstance(candidate, dict)
    )
    stage2_decision = result["stage2_decision"]
    if stage2_decision is not None:
        raw_candidates.extend(
            candidate
            for candidate in stage2_decision.get("memory_candidates", [])
            if isinstance(candidate, dict)
        )
    return raw_candidates


def build_llm_analysis_payload(result: dict) -> dict:
    """Build the track analysis audit payload from a successful LLM result."""

    stage1_decision = result["stage1_decision"]
    stage2_decision = result["stage2_decision"]
    extracted_facts = list(stage1_decision.get("extracted_facts", []))
    memory_candidates = list(stage1_decision.get("memory_candidates", []))
    if stage2_decision is not None:
        extracted_facts.extend(stage2_decision.get("extracted_facts", []))
        memory_candidates.extend(stage2_decision.get("memory_candidates", []))
    payload = {
        "route": result["route"],
        "selected_memory_ids": list(result.get("selected_memory_ids", [])),
        "used_memory_ids": list(result.get("used_memory_ids", [])),
        "stage1_decision": stage1_decision,
        "stage2_decision": stage2_decision,
        "extracted_facts": extracted_facts,
        "memory_candidates": memory_candidates,
    }
    payload.update(build_phase_v1_current_signal_audit(result))
    return payload


def build_phase_v1_current_signal_audit(result: dict) -> dict:
    """Return audit-only Phase V1 Stage 0 signal data without affecting answer persistence."""

    raw_current_signal = result.get("current_signal")
    current_signal_error = result.get("current_signal_audit_error")
    audit_payload: dict[str, object] = {}
    if raw_current_signal is not None:
        try:
            signal = (
                raw_current_signal
                if isinstance(raw_current_signal, PhaseV1Stage0SignalExtraction)
                else PhaseV1Stage0SignalExtraction.model_validate(raw_current_signal)
            )
            audit_payload["current_signal"] = signal.model_dump(mode="json")
        except (ValidationError, TypeError, ValueError) as error:
            current_signal_error = f"{error.__class__.__name__}: {error}"
    if isinstance(current_signal_error, str) and current_signal_error.strip():
        audit_payload["current_signal_error"] = current_signal_error.strip()
    return audit_payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and print a concise user-facing result."""

    args = build_parser().parse_args(argv)
    try:
        result = run_message(args.message, thread_id=args.thread_id)
    except (ProviderConfigError, ProviderResponseError, ValueError) as error:
        print(
            f"LLM failed: {error}. user turn saved; assistant turn not saved.",
            file=sys.stderr,
        )
        return 1
    print(f"Assistant: {result.get('response')}")
    print(f"Track: {result['track_id']}")
    print(f"Capsule: {result.get('capsule_id') or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
