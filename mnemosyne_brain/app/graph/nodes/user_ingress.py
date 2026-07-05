"""User/system graph nodes."""

from __future__ import annotations

from ...contracts.base import new_id, server_now
from ...contracts.memory import MemoryCandidate
from ...contracts.provenance import Provenance
from ...db.repository import SqliteRepository
from ..state import BrainGraphState

class PersistDialogueTurnNode:
    """Persist a user or system turn."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        turn, created = self._repository.persist_dialogue_turn(
            dialogue_id=state["dialogue_id"],
            input_source=state["input_source"],
            role="user" if state["input_source"] == "user" else "system",
            external_message_id=state.get("external_message_id"),
            content_text=state.get("input_text"),
        )
        return {"turn_id": turn.turn_id, "turn_created": created}


class BootstrapOrLoadTrackNode:
    """Create or load the durable track for the thread."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        track = self._repository.bootstrap_or_load_track(
            dialogue_id=state["dialogue_id"],
            thread_id=state["thread_id"],
            owner_user_id=state["owner_user_id"],
        )
        self._repository.attach_turn_to_track(state["turn_id"], track.track_id, track.thread_id)
        self._repository.update_track_status(track.track_id, track.status, last_turn_id=state["turn_id"])
        return {
            "track_id": track.track_id,
            "track_ref": [{"track_id": track.track_id}],
            "track_snapshot": {"status": track.status.value},
        }


class MemoryRetrievalNode:
    """Retrieve memory refs for the current track."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        return {"retrieved_memory_refs": [], "retrieved_memory_summary": ""}


class TurnAnalyzerNode:
    """Analyze user text for MVP memory and executor intent."""

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        text = state.get("input_text", "")
        if state.get("turn_created") is False:
            return {
                "turn_analysis": {
                    "should_write_memory": False,
                    "should_call_executor": False,
                }
            }
        return {
            "turn_analysis": {
                "should_write_memory": bool(text.strip()),
                "should_call_executor": "delegate:" in text.lower(),
            }
        }


class PersistMemoryCandidatesNode:
    """Persist candidate memory rows and return refs only."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        if not state.get("turn_analysis", {}).get("should_write_memory"):
            return {"memory_candidate_refs": []}
        text = state.get("input_text", "").strip()
        content = {"text": text}
        dedupe_key = self._repository.stable_key("memory", text.lower())
        now = server_now()
        candidate = MemoryCandidate(
            candidate_id=new_id("cand"),
            dialogue_id=state["dialogue_id"],
            track_id=state["track_id"],
            turn_id=state["turn_id"],
            candidate_type="fact",
            recommended_action="save_immediately" if text.lower().startswith("remember:") else "stage",
            confidence=0.8,
            dedupe_key=dedupe_key,
            idempotency_key=self._repository.stable_key(state["turn_id"], dedupe_key),
            content_json=content,
            provenance_json=Provenance(
                source=state["input_source"],
                dialogue_id=state["dialogue_id"],
                track_id=state["track_id"],
                turn_id=state["turn_id"],
            ),
            created_at=now,
            updated_at=now,
        )
        candidate, _created = self._repository.persist_memory_candidate(candidate)
        return {"memory_candidate_refs": [{"candidate_id": candidate.candidate_id}]}


class TrackUpdaterNode:
    """Refresh runtime track snapshot from durable storage."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def __call__(self, state: BrainGraphState) -> BrainGraphState:
        track = self._repository.get_track(state["track_id"])
        return {"track_snapshot": {"status": track.status.value}}
