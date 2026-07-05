"""Deterministic LLM context policy v0.4.3."""

from __future__ import annotations

from typing import Any

from .db.repository import SqliteRepository

LLM_CONTEXT_LAST_MESSAGES = 12
LLM_CONTEXT_MAX_CHARS = 24000
MEMORY_MANIFEST_MAX_ITEMS = 80
SELECTED_MEMORY_MAX_ITEMS = 20
PINNED_EXACT_MESSAGES_MAX = 40


class ContextBuilder:
    """Builds deterministic Stage 1 and Stage 2 LLM contexts without calling an LLM."""

    def __init__(self, repository: SqliteRepository) -> None:
        self._repository = repository

    def build_stage1_context(self, *, track_id: str, current_user_message: str) -> dict[str, Any]:
        """Build the manifest-only first-stage context for the current active track."""

        context = {
            "stage": "stage1",
            "track_id": track_id,
            "current_user_message": current_user_message,
            "recent_messages": self._recent_messages(track_id),
            "previous_analysis": self._repository.get_latest_track_analysis(track_id),
            "pinned_exact_messages": self._repository.list_pinned_exact_messages(
                track_id,
                limit=PINNED_EXACT_MESSAGES_MAX,
            ),
            "memory_manifest": self._repository.list_memory_manifest_items(
                limit=MEMORY_MANIFEST_MAX_ITEMS,
            ),
            "limits": self._limits(),
        }
        self._validate_no_summary_keys(context)
        return self._apply_overflow_policy(context)

    def build_stage2_context(
        self,
        *,
        track_id: str,
        current_user_message: str,
        selected_memory_ids: list[str],
    ) -> dict[str, Any]:
        """Build a second-stage context with only validated selected memories."""

        deduped_ids = self._dedupe_preserving_order(selected_memory_ids)
        limited_ids = deduped_ids[:SELECTED_MEMORY_MAX_ITEMS]
        memory_by_id = self._repository.get_memory_context_items(limited_ids)
        selected_memory_context = [
            memory_by_id[memory_id]
            for memory_id in limited_ids
            if memory_id in memory_by_id
        ]
        rejected_memory_ids = [
            memory_id
            for memory_id in deduped_ids
            if memory_id not in memory_by_id or memory_id not in limited_ids
        ]
        context = {
            "stage": "stage2",
            "track_id": track_id,
            "current_user_message": current_user_message,
            "recent_messages": self._recent_messages(track_id),
            "previous_analysis": self._repository.get_latest_track_analysis(track_id),
            "pinned_exact_messages": self._repository.list_pinned_exact_messages(
                track_id,
                limit=PINNED_EXACT_MESSAGES_MAX,
            ),
            "selected_memory_context": selected_memory_context,
            "rejected_memory_ids": rejected_memory_ids,
            "limits": self._limits(),
        }
        self._validate_no_summary_keys(context)
        return self._apply_overflow_policy(context)

    def _recent_messages(self, track_id: str) -> list[dict[str, Any]]:
        turns = self._repository.list_recent_turns_for_active_track(
            track_id,
            limit=LLM_CONTEXT_LAST_MESSAGES,
        )
        return [
            {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "input_source": turn.input_source,
                "content_text": turn.content_text,
                "created_at": turn.created_at,
            }
            for turn in turns
        ]

    def _apply_overflow_policy(self, context: dict[str, Any]) -> dict[str, Any]:
        while (
            self._context_size(context) > LLM_CONTEXT_MAX_CHARS
            and context["recent_messages"]
        ):
            context["recent_messages"].pop(0)
        if self._context_size(context) > LLM_CONTEXT_MAX_CHARS and not context["current_user_message"]:
            raise ValueError("current_user_message must not be dropped")
        return context

    def _context_size(self, context: dict[str, Any]) -> int:
        return len(self._repository.to_json(context))

    def _dedupe_preserving_order(self, memory_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for memory_id in memory_ids:
            if memory_id not in seen:
                seen.add(memory_id)
                deduped.append(memory_id)
        return deduped

    def _validate_no_summary_keys(self, value: Any) -> None:
        if isinstance(value, dict):
            if "summary" in value:
                raise ValueError("LLM context must not contain a summary key")
            for item in value.values():
                self._validate_no_summary_keys(item)
            return
        if isinstance(value, list):
            for item in value:
                self._validate_no_summary_keys(item)

    def _limits(self) -> dict[str, int]:
        return {
            "LLM_CONTEXT_LAST_MESSAGES": LLM_CONTEXT_LAST_MESSAGES,
            "LLM_CONTEXT_MAX_CHARS": LLM_CONTEXT_MAX_CHARS,
            "MEMORY_MANIFEST_MAX_ITEMS": MEMORY_MANIFEST_MAX_ITEMS,
            "SELECTED_MEMORY_MAX_ITEMS": SELECTED_MEMORY_MAX_ITEMS,
            "PINNED_EXACT_MESSAGES_MAX": PINNED_EXACT_MESSAGES_MAX,
        }
