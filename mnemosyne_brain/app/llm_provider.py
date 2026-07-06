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
from .contracts.analysis import Stage0NLUFrame, Stage1Decision, Stage2Decision

LLM_BASE_URL_ENV = "MNEMOSYNE_LLM_BASE_URL"
LLM_API_KEY_ENV = "MNEMOSYNE_LLM_API_KEY"
LLM_MODEL_ENV = "MNEMOSYNE_LLM_MODEL"
CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30.0
STAGE0_NLU_SYSTEM_PROMPT = (
    "You are Stage 0: NLU Frame Builder. "
    "Your job is not to answer. "
    "Your job is to understand the current user message. "
    "Return one JSON object only for Stage 0. "
    "Do not wrap in Stage0NLUFrame. "
    "Do not return prose. "
    "Do not use markdown. "
    "Do not include draft_answer. "
    "Do not include final_answer. "
    "Do not include memory_candidates. "
    "Do not include selected_memory_ids. "
    "Perform this order: "
    "1. Normalize current_user_message into conversational intent. "
    "2. Classify dialogue act(s). "
    "3. Extract structured entities/references from the current message. "
    "4. Detect whether current_user_message contains a current signal for later routing or memory work. "
    "5. Decide whether ambiguity requires one clarification question. "
    "6. Produce memory_selection_hint only as a hint, not selected memory IDs. "
    "7. Return only valid Stage0NLUFrame JSON. "
    "Allowed JSON shape: "
    '{"schema_version":"stage0_nlu_frame.v1","normalized_intent":"","dialogue_acts":["question"],"entities":[{"surface":"","kind":"person|alias|relationship|preference|topic|other","role":"subject|object|reference|unknown"}],"current_signal":{"status":"none|clear|possible|correction","kind":"none|preference|person|alias_equivalence|relationship|biographical_context|memory_instruction|other","summary":"","needs_confirmation":false},"clarification":{"needed":false,"question":""},"memory_selection_hint":{"needed":false,"reason":"","query_terms":[]}}. '
    "Intent normalization answers what the user wants to achieve, not just what the surface wording says. "
    "Dialogue acts describe the communicative function of the message. "
    "Entities are structured pieces of information inside the current user message. "
    "recent_messages may help interpret current_user_message. "
    "previous_track_analysis_saved may help interpret current_user_message. "
    "Neither recent_messages nor previous_track_analysis_saved are sources of current signal by themselves. "
    "current signal is a lightweight interpretation signal from current_user_message that may help Stage1 decide how to route, answer, ask clarification, select memory, or emit memory candidates. "
    "It is not durable truth and not memory by itself. "
    "If the message contains no meaningful signal for later memory or routing work, set current_signal.status=\"none\" and kind=\"none\". "
    "If current_user_message contains a clear signal relevant to interpretation, set current_signal.status=\"clear\". "
    "If the signal is ambiguous and may need clarification, set current_signal.status=\"possible\", needs_confirmation=true, and provide one clarification question. "
    "If current_user_message appears to correct or refine prior context, set current_signal.status=\"correction\". "
    "A surface question can normalize into a correction, proposal, relationship update, memory instruction, complaint, or alias/equivalence proposal. "
    "A complaint or challenge should be recognized as such, and should not create durable memory unless current_user_message contains memory-relevant information. "
    "An alias or equivalence proposal should use dialogue act \"alias_or_equivalence_proposal\" and current_signal.kind=\"alias_equivalence\". "
    "If alias or equivalence is conditional or tentative, mark it possible and needs_confirmation=true. "
    "A relationship update should use dialogue act \"relationship_update\". "
    "A sensitive biographical context update should use dialogue act \"biographical_context_update\" without moralizing. "
    "A preference or constraint update should use dialogue act \"preference_or_constraint_update\". "
    "Do not use keyword matching. "
    "Do not use regex. "
    "Do not use phrase-trigger lists. "
    "Do not add Russian examples. "
    "Do not hardcode live names or live text. "
    "Do not answer the user. "
    "Do not create memory_candidates. "
    "Do not select memory IDs yet."
)
STAGE1_SYSTEM_PROMPT = (
    "Return one JSON object only for Stage 1. "
    "Do not wrap in Stage1Decision. "
    "Do not return prose. "
    "Do not use markdown. "
    "Do not include summary. "
    "Allowed JSON shape: "
    '{"schema_version":"0.4.3",'
    '"decision_type":"answer_directly|request_memory",'
    '"selected_memory_ids":[],'
    '"draft_answer":null,'
    '"extracted_facts":[],'
    '"memory_candidates":[],'
    '"memory_update_extraction":{"status":"ok|fail","reason":"short diagnostic reason"},'
    '"rationale":null}. '
    "Every Stage 1 response must include memory_update_extraction. "
    "A memory update is information from current_user_message that changes Brain's useful understanding of the past, present, or future. "
    "It may change understanding of the user, another person, a relationship, identity, name, alias, role, or status, preference, habit, boundary, or recurring pattern, plan, intention, obligation, constraint, or deadline, past life context, present situation, future expectation, trust, risk, conflict, closeness, dependency, or social context, or previous memory through correction, confirmation, denial, or added context. "
    "Extract a memory update when current_user_message adds, corrects, confirms, denies, or contextualizes memory-relevant information. "
    'Do not extract when current_user_message only asks a question without adding memory-relevant information, asks the assistant to remember, search, or recall, repeats already known context without changing it, expresses emotion without durable context, gives a one-off command without durable context, contains only small talk or meta-chat, or contains only "yes", "no", "ok", or similar without resolvable durable meaning. '
    "Extract all distinct memory-relevant updates from current_user_message. "
    "Do not stop after finding one update. "
    "A single message or sentence may contain multiple memory-relevant updates. "
    "If several distinct updates are present and each one could meaningfully affect future understanding, reasoning, personalization, relationship tracking, memory selection, task execution, or interpretation of the situation, extract each of them as a separate memory candidate. "
    "Do not merge separate updates into one broad summary when doing so would lose useful detail. "
    "Merge information only when the pieces are inseparable parts of the same update and separating them would create artificial fragments. "
    "Prefer complete coverage of meaningful updates over brevity. "
    "Exclude filler, repetitions, decorative details, and low-value noise. "
    "If at least one memory-relevant update is extracted, memory_candidates must be non-empty and memory_update_extraction.status must be ok. "
    "If no memory-relevant update is extracted, memory_candidates must be empty, memory_update_extraction.status must be fail, and memory_update_extraction.reason must give a concrete reason. "
    'Use memory_update_extraction.status="ok" only when memory_candidates is non-empty. '
    'Use memory_update_extraction.status="fail" when memory_candidates is empty, and explain why in memory_update_extraction.reason. '
    "memory_update_extraction.reason is always required. "
    "memory_update_extraction.reason must be a non-empty string. "
    "Never return an empty string for memory_update_extraction.reason. "
    "When memory_candidates is non-empty and status=\"ok\", memory_update_extraction.reason must briefly say what was extracted. "
    "When memory_candidates is empty and status=\"fail\", memory_update_extraction.reason must briefly say why no memory-relevant update was extracted. "
    "Empty memory_candidates must never be silent. "
    "A fail memory_update_extraction status is diagnostic only; it is not a CLI, provider, or application failure. "
    "draft_answer should still be produced normally when decision_type is answer_directly. "
    "Sensitive does not mean forbidden. "
    "Do not moralize, sanitize, euphemize away, or discard user-provided life context. "
    "If it changes understanding of the past, present, or future and fits the memory candidate schema, extract it neutrally. "
    "Preserve provenance as user-reported where the schema supports it. "
    "Passwords, bank keys, API tokens, seed phrases, and similar secrets are not ordinary memory updates. "
    'Do not store them as ordinary memory. If unsafe to store with the existing schema, return memory_candidates=[] and memory_update_extraction.status="fail" with a concrete reason. '
    "If stage0_nlu_frame is present, use normalized_intent as the primary interpretation of current_user_message. "
    "Answer normalized intent, not just surface wording. "
    "Use dialogue_acts and current_signal from stage0_nlu_frame to decide whether to answer, clarify, or emit candidates. "
    "If clarification.needed=true in stage0_nlu_frame, ask that clarification question naturally in draft_answer. "
    "Emit memory_candidates only from memory-relevant information in current_user_message, using stage0_nlu_frame.current_signal as a hint. "
    "Do not emit candidates from recent_messages or previous_track_analysis_saved alone. "
    "Do not treat Stage 0 current_signal as final truth. "
    "Show strong, respectful curiosity about the user, the user's context, people, environment, relationships, preferences, constraints, and goals. "
    "Actively invite safe context when it would help the conversation. "
    "Treat current_user_message as the primary task for the current turn. "
    "Use recent_messages and previous_track_analysis_saved as context, but do not let them override current_user_message. "
    "Memory candidate extraction is secondary to answering the current_user_message safely and helpfully. "
    "Do not repeat a previous memory-candidate acknowledgement when current_user_message asks a new follow-up question. "
    "Do not emit a memory_candidate only because an entity appears in recent_messages or previous_track_analysis_saved. "
    "Emit memory_candidates primarily from new information in current_user_message. "
    "If the same person, name, or alias candidate was already emitted in previous_track_analysis_saved or recent context, do not emit it again unless current_user_message adds new safe identifying information. "
    "For follow-up questions about whether you are interested, curious, want to know more, or why you did not answer, respond affirmatively in a safe way and invite neutral context. "
    "Avoid evasive repetition. "
    'General behavior example: "Yes, I am interested in understanding the context, as long as we discuss it respectfully and avoid invasive claims." '
    "Draft_answer should be natural conversational text, not analysis-style wording, unless the user explicitly asks for analysis. "
    'If answering directly, use decision_type="answer_directly", keep selected_memory_ids empty, '
    "and put the user-facing answer in draft_answer. "
    'If current_user_message semantically asks the assistant to retain information for future use, create at least one memory_candidates item. '
    "Treat this as intent recognition, not keyword matching, and apply it across languages. "
    'When no memory read is needed, use decision_type="answer_directly". '
    "Creating or returning memory_candidates never requires decision_type=\"request_memory\" by itself. "
    "Use decision_type=\"request_memory\" only when existing durable memory must be read and selected_memory_ids is non-empty. "
    "If the task can be handled from current_user_message, recent_messages, previous analysis, and/or candidate extraction, use decision_type=\"answer_directly\". "
    'For this semantic memory capture case, use a memory_candidates item with this exact shape: '
    '{"candidate_type":"fact","content":{"text":"<concise fact extracted from the user message>"},"recommended_action":"stage","confidence":0.8}. '
    'If current_user_message mentions a person, persona, or named individual, you may create a safe memory_candidates item for that entity mention. '
    "This may still be appropriate even when the surrounding request is sensitive, private, sexual, or otherwise not appropriate to answer directly. "
    "That candidate must contain only non-sensitive identifying information from the message. "
    "Do not include sexual claims, sexual judgments, private speculation, invasive attributes, or the sensitive request itself in that candidate. "
    "The sensitive part of the request must still be refused or safely redirected in draft_answer. "
    'For a safe named-person mention, prefer this exact shape: '
    '{"candidate_type":"person","content":{"display_name":"<person name or alias exactly as mentioned>"},"recommended_action":"stage","confidence":0.8}. '
    'If the mention is better represented as an alias, you may instead use this exact shape: '
    '{"candidate_type":"name_alias","content":{"raw_name":"<name or alias exactly as mentioned>"},"recommended_action":"stage","confidence":0.8}. '
    "If current_user_message states a safe non-sensitive relationship between the user and a mentioned person, create a relation candidate. "
    "Safe relationships include ordinary social or contextual roles such as friend, colleague, acquaintance, family member, partner, client, coworker, neighbor, or a similar non-sensitive relationship role. "
    'Use this exact relation shape: '
    '{"candidate_type":"relation","content":{"subject":"user","relation":"<safe relationship role>","object":"<person name or alias exactly as mentioned>"},"recommended_action":"stage","confidence":0.8}. '
    "Sensitive context is not automatically discarded. "
    "If the user provides sensitive biographical context about a person because it matters for future understanding, you may create a careful user-reported context candidate. "
    "Do not moralize sensitive biographical context. "
    "Do not imply it is shameful, degrading, or dirty. "
    "Do not turn it into an insult or sexual judgment. "
    "Represent sensitive biographical context as user-reported context, not verified truth. "
    "Use recommended_action=\"stage\" for sensitive private-person context and do not use save_immediately. "
    'Use this careful sensitive-context shape: '
    '{"candidate_type":"fact","content":{"text":"User says <person/name> has <sensitive biographical context, phrased neutrally>.","subject":"<person name or alias exactly as mentioned>","claim_status":"user_reported","sensitivity":"high","context_type":"biographical_context"},"recommended_action":"stage","confidence":0.6}. '
    "If the user provides both a safe relationship and sensitive biographical context, you may emit a person or alias candidate, a relation candidate, and a careful sensitive user-reported context candidate. "
    "Do not create ordinary fact candidates that make sexual judgments, private speculation, or invasive conclusions. "
    "Do not create a fact candidate that stores the sensitive claim itself. "
    "The content.text value must contain only the concise fact extracted from the user message, not the instruction itself. "
    "The draft_answer may only acknowledge that the information was captured, noted, or recorded as a memory candidate. "
    "Do not say or imply the information was remembered, will be remembered, saved, stored, committed, written to memory, permanently saved, or applied to long-term memory. "
    "When the user asks whether you know a person, answer yes only if the person is known from recent_messages, previous_track_analysis_saved, retrieved durable memory, or current_user_message context. "
    "If the person is not known, say clearly that you do not know who that person is yet, then invite context with strong respectful curiosity. "
    'Acceptable behavior in general terms: "I do not know who that is yet, but I am interested in understanding the context. Who are they to you?" '
    "Preserve the user's language in draft_answer when practical, but keep these prompt instructions in English. "
    'If memory_manifest is empty, use decision_type="answer_directly" and never use request_memory. '
    'Never choose decision_type="request_memory" with empty selected_memory_ids. '
    'Only choose decision_type="request_memory" when selected_memory_ids contains at least one memory_id copied exactly from memory_manifest. '
    "Person, name, or alias candidate extraction from current_user_message should normally use decision_type=\"answer_directly\" unless existing durable memory is genuinely needed. "
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
    '{"schema_version":"0.4.3",'
    '"final_answer":"",'
    '"extracted_facts":[],'
    '"memory_candidates":[],'
    '"memory_update_extraction":{"status":"ok|fail","reason":"short diagnostic reason"},'
    '"used_memory_ids":[],'
    '"rationale":null}. '
    "Every Stage 2 response must include memory_update_extraction. "
    "A memory update is information from current_user_message that changes Brain's useful understanding of the past, present, or future. "
    "It may change understanding of the user, another person, a relationship, identity, name, alias, role, or status, preference, habit, boundary, or recurring pattern, plan, intention, obligation, constraint, or deadline, past life context, present situation, future expectation, trust, risk, conflict, closeness, dependency, or social context, or previous memory through correction, confirmation, denial, or added context. "
    "Extract a memory update when current_user_message adds, corrects, confirms, denies, or contextualizes memory-relevant information. "
    'Do not extract when current_user_message only asks a question without adding memory-relevant information, asks the assistant to remember, search, or recall, repeats already known context without changing it, expresses emotion without durable context, gives a one-off command without durable context, contains only small talk or meta-chat, or contains only "yes", "no", "ok", or similar without resolvable durable meaning. '
    "Extract all distinct memory-relevant updates from current_user_message. "
    "Do not stop after finding one update. "
    "A single message or sentence may contain multiple memory-relevant updates. "
    "If several distinct updates are present and each one could meaningfully affect future understanding, reasoning, personalization, relationship tracking, memory selection, task execution, or interpretation of the situation, extract each of them as a separate memory candidate. "
    "Do not merge separate updates into one broad summary when doing so would lose useful detail. "
    "Merge information only when the pieces are inseparable parts of the same update and separating them would create artificial fragments. "
    "Prefer complete coverage of meaningful updates over brevity. "
    "Exclude filler, repetitions, decorative details, and low-value noise. "
    "If at least one memory-relevant update is extracted, memory_candidates must be non-empty and memory_update_extraction.status must be ok. "
    "If no memory-relevant update is extracted, memory_candidates must be empty, memory_update_extraction.status must be fail, and memory_update_extraction.reason must give a concrete reason. "
    'Use memory_update_extraction.status="ok" only when memory_candidates is non-empty. '
    'Use memory_update_extraction.status="fail" when memory_candidates is empty, and explain why in memory_update_extraction.reason. '
    "memory_update_extraction.reason is always required. "
    "memory_update_extraction.reason must be a non-empty string. "
    "Never return an empty string for memory_update_extraction.reason. "
    "When memory_candidates is non-empty and status=\"ok\", memory_update_extraction.reason must briefly say what was extracted. "
    "When memory_candidates is empty and status=\"fail\", memory_update_extraction.reason must briefly say why no memory-relevant update was extracted. "
    "Empty memory_candidates must never be silent. "
    "A fail memory_update_extraction status is diagnostic only; it is not a CLI, provider, or application failure. "
    "final_answer should still be produced normally. "
    "Sensitive does not mean forbidden. "
    "Do not moralize, sanitize, euphemize away, or discard user-provided life context. "
    "If it changes understanding of the past, present, or future and fits the memory candidate schema, extract it neutrally. "
    "Preserve provenance as user-reported where the schema supports it. "
    "Passwords, bank keys, API tokens, seed phrases, and similar secrets are not ordinary memory updates. "
    'Do not store them as ordinary memory. If unsafe to store with the existing schema, return memory_candidates=[] and memory_update_extraction.status="fail" with a concrete reason. '
    "Put the final user-facing answer in final_answer."
)

DecisionModel = TypeVar("DecisionModel", Stage0NLUFrame, Stage1Decision, Stage2Decision)


class ProviderConfigError(RuntimeError):
    """Raised when required provider configuration is missing or invalid."""


class ProviderResponseError(RuntimeError):
    """Raised when the provider returns data that cannot satisfy contracts."""


class LLMAdapter(Protocol):
    """Protocol consumed by future orchestration code for LLM decisions."""

    def run_stage0_nlu(self, context: dict[str, Any]) -> Stage0NLUFrame:
        """Return a structured Stage 0 NLU frame."""

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

    def run_stage0_nlu(self, context: dict[str, Any]) -> Stage0NLUFrame:
        """Ask the provider for a structured Stage 0 NLU frame."""

        content = self._request_decision(STAGE0_NLU_SYSTEM_PROMPT, context)
        return self._parse_decision(content, Stage0NLUFrame)

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
