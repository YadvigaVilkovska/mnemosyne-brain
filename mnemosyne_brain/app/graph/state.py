"""Reference-only LangGraph state."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

REF_KEY_BY_FIELD = {
    "executor_event_ref": "event_id",
    "track_ref": "track_id",
    "memory_candidate_refs": "candidate_id",
    "retrieved_memory_refs": "memory_id",
}


def dedupe_ref_dicts(left: list[dict[str, str]] | None, right: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Merge reference dictionaries by their only key and reject payload-shaped dicts."""

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in (left or []) + (right or []):
        if len(item) != 1:
            raise ValueError("reference state items must contain exactly one key")
        key, value = next(iter(item.items()))
        if key not in {"event_id", "track_id", "candidate_id", "memory_id"}:
            raise ValueError(f"unsupported reference key: {key}")
        marker = (key, value)
        if marker not in seen:
            seen.add(marker)
            merged.append(item)
    return merged


def dedupe_errors(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Merge runtime errors without duplication."""

    merged: list[str] = []
    for item in (left or []) + (right or []):
        if item not in merged:
            merged.append(item)
    return merged


class BrainGraphState(TypedDict, total=False):
    """LangGraph state containing runtime data and durable references only."""

    schema_version: str
    thread_id: str
    dialogue_id: str
    owner_user_id: str
    track_id: str
    run_id: str
    turn_id: str
    turn_created: bool
    input_source: str
    input_text: str
    external_message_id: str
    event_id: str
    capsule_id: str
    executor_event_ref: Annotated[list[dict[str, str]], dedupe_ref_dicts]
    track_ref: Annotated[list[dict[str, str]], dedupe_ref_dicts]
    track_snapshot: dict[str, Any]
    memory_candidate_refs: Annotated[list[dict[str, str]], dedupe_ref_dicts]
    retrieved_memory_refs: Annotated[list[dict[str, str]], dedupe_ref_dicts]
    retrieved_memory_summary: str
    turn_analysis: dict[str, Any]
    executor_feedback_analysis: dict[str, Any]
    executor_task: dict[str, str]
    router_decision: dict[str, str]
    route: str
    response: str
    errors: Annotated[list[str], dedupe_errors]
