"""OpenAI-compatible LLM provider adapter for structured decisions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from .config import load_project_env
from .contracts.analysis import Stage1Decision, Stage2Decision

LLM_BASE_URL_ENV = "MNEMOSYNE_LLM_BASE_URL"
LLM_API_KEY_ENV = "MNEMOSYNE_LLM_API_KEY"
LLM_MODEL_ENV = "MNEMOSYNE_LLM_MODEL"
CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30.0
STAGE1_SYSTEM_PROMPT = (
    "Return one JSON object only for Stage 1. "
    "Do not wrap in Stage1Decision. "
    "Do not return prose. "
    "Do not use markdown. "
    "Do not include summary. "
    "Allowed JSON shape: "
    '{"decision_type":"answer_directly|request_memory",'
    '"selected_memory_ids":[],'
    '"draft_answer":null,'
    '"extracted_facts":[],'
    '"memory_candidates":[],'
    '"rationale":null}. '
    'If answering directly, use decision_type="answer_directly", keep selected_memory_ids empty, '
    "and put the user-facing answer in draft_answer. "
    'If current_user_message semantically asks the assistant to retain information for future use, create at least one memory_candidates item. '
    "Treat this as intent recognition, not keyword matching, and apply it across languages. "
    'When no memory read is needed, use decision_type="answer_directly". '
    'For this semantic memory capture case, use a memory_candidates item with this exact shape: '
    '{"candidate_type":"fact","content":{"text":"<concise fact extracted from the user message>"},"recommended_action":"stage","confidence":0.8}. '
    "The content.text value must contain only the concise fact extracted from the user message, not the instruction itself. "
    "The draft_answer may only acknowledge that the information was captured, noted, or recorded as a memory candidate. "
    "Do not say or imply the information was remembered, will be remembered, saved, stored, committed, written to memory, permanently saved, or applied to long-term memory. "
    "Preserve the user's language in draft_answer when practical, but keep these prompt instructions in English. "
    'If memory_manifest is empty, use decision_type="answer_directly". '
    'Never choose decision_type="request_memory" with empty selected_memory_ids. '
    'Only choose decision_type="request_memory" when selected_memory_ids contains at least one memory_id copied exactly from memory_manifest. '
    'For questions about recent conversation history, such as "what did I just ask" or "what did I just say", use recent_messages and answer_directly. '
    'If the answer can be produced from current_user_message and recent_messages, use answer_directly.'
)
STAGE2_SYSTEM_PROMPT = (
    "Return one JSON object only for Stage 2. "
    "Do not wrap in Stage2Decision. "
    "Do not return prose. "
    "Do not use markdown. "
    "Do not include summary. "
    "Allowed JSON shape: "
    '{"final_answer":"",'
    '"extracted_facts":[],'
    '"memory_candidates":[],'
    '"used_memory_ids":[],'
    '"rationale":null}. '
    "Put the final user-facing answer in final_answer."
)

DecisionModel = TypeVar("DecisionModel", Stage1Decision, Stage2Decision)


class ProviderConfigError(RuntimeError):
    """Raised when required provider configuration is missing or invalid."""


class ProviderResponseError(RuntimeError):
    """Raised when the provider returns data that cannot satisfy contracts."""


class LLMAdapter(Protocol):
    """Protocol consumed by future orchestration code for LLM decisions."""

    def decide_stage1(self, stage1_context: dict[str, Any]) -> Stage1Decision:
        """Return a structured Stage 1 decision."""

    def decide_stage2(self, stage2_context: dict[str, Any]) -> Stage2Decision:
        """Return a structured Stage 2 decision."""


class HttpTransport(Protocol):
    """Small HTTP boundary that lets tests inject fake network behavior."""

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Post JSON and return decoded response JSON."""


class UrllibHttpTransport:
    """Stdlib HTTP transport for OpenAI-compatible JSON POST requests."""

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Send one JSON request without logging secrets or headers."""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.URLError as error:
            raise ProviderResponseError(f"LLM provider request failed: {error.reason}") from error

        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise ProviderResponseError("LLM provider returned invalid response JSON") from error
        if not isinstance(decoded, dict):
            raise ProviderResponseError("LLM provider response must be a JSON object")
        return decoded


@dataclass(frozen=True)
class OpenAICompatibleLLMProvider(LLMAdapter):
    """Adapter for OpenAI-compatible chat completions APIs."""

    base_url: str
    api_key: str
    model: str
    transport: HttpTransport
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(
        cls,
        *,
        transport: HttpTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> "OpenAICompatibleLLMProvider":
        """Create a provider from environment variables only."""

        load_project_env()
        base_url = os.environ.get(LLM_BASE_URL_ENV, "").strip()
        api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
        model = os.environ.get(LLM_MODEL_ENV, "").strip()
        missing = [
            name
            for name, value in (
                (LLM_BASE_URL_ENV, base_url),
                (LLM_API_KEY_ENV, api_key),
                (LLM_MODEL_ENV, model),
            )
            if not value
        ]
        if missing:
            raise ProviderConfigError(f"Missing required LLM env vars: {', '.join(missing)}")
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            transport=transport or UrllibHttpTransport(),
            timeout_seconds=timeout_seconds,
        )

    def decide_stage1(self, stage1_context: dict[str, Any]) -> Stage1Decision:
        """Ask the provider for a structured Stage 1 decision."""

        content = self._request_decision(STAGE1_SYSTEM_PROMPT, stage1_context)
        return self._parse_decision(content, Stage1Decision)

    def decide_stage2(self, stage2_context: dict[str, Any]) -> Stage2Decision:
        """Ask the provider for a structured Stage 2 final decision."""

        content = self._request_decision(STAGE2_SYSTEM_PROMPT, stage2_context)
        return self._parse_decision(content, Stage2Decision)

    def _request_decision(self, system_prompt: str, context: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
            ],
        }
        response = self.transport.post_json(
            url=f"{self.base_url}{CHAT_COMPLETIONS_PATH}",
            headers=self._headers(),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        return self._extract_message_content(response)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_message_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderResponseError("LLM provider response missing choices[0].message.content") from error
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("LLM provider message content must be a non-empty JSON string")
        return content

    def _parse_decision(self, content: str, model_type: type[DecisionModel]) -> DecisionModel:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ProviderResponseError("LLM provider returned invalid decision JSON") from error
        if not isinstance(payload, dict):
            raise ProviderResponseError("LLM provider decision JSON must be an object")
        try:
            return model_type.model_validate(payload)
        except ValidationError as error:
            raise ProviderResponseError(f"LLM provider decision failed contract validation: {error}") from error
