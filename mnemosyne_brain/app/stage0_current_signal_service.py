"""Contract: provide a tiny reusable service for Phase V1 Stage 0 current-signal extraction."""

from __future__ import annotations

import re
from typing import Any

from .contracts.analysis import PhaseV1Stage0SignalExtraction

QUESTION_LEAD_WORDS = {
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "какая",
    "какие",
    "какой",
    "какое",
    "кто",
    "что",
}
WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*")
COMPLEXITY_REACTION_MARKERS = ("bureaucr", "complex", "complicated", "бюрократ")


class Stage0CurrentSignalService:
    """Build audit-only Stage 0 current-signal payloads without writing memory."""

    def extract_for_runtime(
        self,
        stage1_context: dict[str, Any],
        *,
        adapter: Any,
    ) -> PhaseV1Stage0SignalExtraction | dict[str, Any] | None:
        """Return optional runtime signal data from an adapter hook when available."""

        extractor = getattr(adapter, "run_phase_v1_stage0_signal_extraction", None)
        if not callable(extractor):
            return None
        return extractor(dict(stage1_context))

    def extract_debug(self, message: str) -> PhaseV1Stage0SignalExtraction:
        """Return a deterministic debug/demo extraction for one message."""

        cleaned_message = message.strip()
        entities = self._build_entities(cleaned_message)
        information_signals = self._build_information_signals(cleaned_message, entities)
        return PhaseV1Stage0SignalExtraction(
            entities=entities,
            information_signals=information_signals,
            unresolved_references=[],
            ambiguous_references=[],
        )

    def _build_entities(self, message: str) -> list[dict[str, Any]]:
        """Return a single literal person mention when the first token looks like a name."""

        matches = WORD_PATTERN.findall(message)
        if not matches:
            return []
        first_word = matches[0]
        if first_word.lower() in QUESTION_LEAD_WORDS:
            return []
        if not self._looks_like_name(first_word):
            return []
        return [
            {
                "id": "e1",
                "mention": first_word,
                "entity_type": "person",
                "source_span": message,
                "resolution_status": "literal",
                "resolved_to": None,
            }
        ]

    def _build_information_signals(self, message: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one deterministic signal shaped for manual inspection."""

        lowered = message.lower()
        if any(marker in lowered for marker in COMPLEXITY_REACTION_MARKERS):
            return [
                {
                    "id": "s1",
                    "source_span": message,
                    "signal_type": "dissatisfaction_with_complexity",
                    "about_entity_ids": [],
                    "signal_scope": "about_current_interaction",
                    "polarity": "asserted",
                    "epistemic_status": "user_reaction",
                    "extraction_note": "Debug-only signal for dissatisfaction with interaction complexity.",
                }
            ]

        about_entity_ids = [entity["id"] for entity in entities]
        signal_type = "possible_connection_between_entities" if about_entity_ids else "user_statement"
        signal_scope = "about_other_person" if about_entity_ids else "about_current_message"
        return [
            {
                "id": "s1",
                "source_span": message,
                "signal_type": signal_type,
                "about_entity_ids": about_entity_ids,
                "signal_scope": signal_scope,
                "polarity": "asserted",
                "epistemic_status": "user_claim",
                "extraction_note": "Debug-only deterministic signal for manual inspection.",
            }
        ]

    def _looks_like_name(self, value: str) -> bool:
        """Return true when a token looks name-like enough for the debug extractor."""

        return any(character.isupper() for character in value)
