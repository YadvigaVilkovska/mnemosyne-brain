"""Pure user message service."""

from __future__ import annotations

from ..contracts.base import SCHEMA_VERSION, new_id
from ..contracts.routing import InputSource


def handle_user_message(request_json: dict, *, graph) -> dict:
    """Invoke the graph for a user message using runtime-only state."""

    dialogue_id = request_json.get("dialogue_id") or new_id("dlg")
    thread_id = request_json.get("thread_id") or new_id("thread")
    result = graph.invoke(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": new_id("run"),
            "input_source": InputSource.USER.value,
            "dialogue_id": dialogue_id,
            "thread_id": thread_id,
            "owner_user_id": request_json.get("owner_user_id") or new_id("user"),
            "input_text": request_json["input_text"],
            "external_message_id": request_json.get("external_message_id"),
        }
    )
    return {
        "dialogue_id": result["dialogue_id"],
        "thread_id": result["thread_id"],
        "track_id": result["track_id"],
        "turn_id": result["turn_id"],
        "capsule_id": result.get("capsule_id"),
        "response": result.get("response"),
    }
