"""LLM provider adapter tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mnemosyne_brain.app.contracts.analysis import Stage0NLUFrame, Stage1Decision, Stage2Decision
from mnemosyne_brain.app.llm_provider import (
    CHAT_COMPLETIONS_PATH,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    OpenAICompatibleLLMProvider,
    ProviderConfigError,
    ProviderResponseError,
)


class FakeTransport:
    """Captures provider requests and returns a configured fake response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Record the request instead of making a network call."""

        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


class LLMProviderTestCase(unittest.TestCase):
    """Verifies the OpenAI-compatible provider remains isolated and strict."""

    @contextmanager
    def _working_directory(self, path: str):
        previous = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    def _provider(self, response_content: str) -> tuple[OpenAICompatibleLLMProvider, FakeTransport]:
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "message": {
                            "content": response_content,
                        }
                    }
                ]
            }
        )
        provider = OpenAICompatibleLLMProvider(
            base_url="https://llm.example.test/v1",
            api_key="test_key",
            model="test_model",
            transport=transport,
            timeout_seconds=1.5,
        )
        return provider, transport

    def _stage1_response(self, *, memory_candidates: list[dict] | None = None) -> str:
        candidates = memory_candidates or []
        return json.dumps(
            {
                "schema_version": "0.4.3",
                "decision_type": "answer_directly",
                "draft_answer": "Done.",
                "memory_candidates": candidates,
                "memory_update_extraction": {
                    "status": "ok" if candidates else "fail",
                    "reason": "Durable information extracted." if candidates else "No durable information extracted.",
                },
            }
        )

    def _stage2_response(self, *, memory_candidates: list[dict] | None = None) -> str:
        candidates = memory_candidates or []
        return json.dumps(
            {
                "schema_version": "0.4.3",
                "final_answer": "Done.",
                "memory_candidates": candidates,
                "memory_update_extraction": {
                    "status": "ok" if candidates else "fail",
                    "reason": "Durable information extracted." if candidates else "No durable information extracted.",
                },
            }
        )

    def test_missing_env_vars_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._working_directory(temp_dir):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ProviderConfigError, LLM_BASE_URL_ENV):
                        OpenAICompatibleLLMProvider.from_env(transport=FakeTransport({}))

    def test_stage1_valid_fake_http_response_returns_decision(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "schema_version": "0.4.3",
                    "decision_type": "answer_directly",
                    "draft_answer": "Already enough context.",
                    "memory_candidates": [],
                    "memory_update_extraction": {
                        "status": "fail",
                        "reason": "No durable information extracted.",
                    },
                }
            )
        )
        decision = provider.decide_stage1({"stage": "stage1", "current_user_message": "hello"})
        self.assertIsInstance(decision, Stage1Decision)
        self.assertEqual("answer_directly", decision.decision_type)
        self.assertEqual(1, len(transport.calls))
        self.assertTrue(transport.calls[0]["url"].endswith(CHAT_COMPLETIONS_PATH))
        self.assertEqual("test_model", transport.calls[0]["payload"]["model"])
        self.assertEqual({"type": "json_object"}, transport.calls[0]["payload"]["response_format"])

    def test_stage0_valid_fake_http_response_returns_frame(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "The user asks whether an alias association can be remembered.",
                    "dialogue_acts": ["question", "alias_or_equivalence_proposal"],
                    "entities": [
                        {
                            "surface": "X",
                            "kind": "alias",
                            "role": "subject",
                        }
                    ],
                    "new_information": {
                        "status": "possible",
                        "kind": "alias_equivalence",
                        "summary": "Possible alias equivalence proposal.",
                        "needs_confirmation": True,
                    },
                    "clarification": {
                        "needed": True,
                        "question": "Do you mean that X and Y refer to the same person?",
                    },
                    "memory_selection_hint": {
                        "needed": False,
                        "reason": "",
                        "query_terms": [],
                    },
                }
            )
        )
        frame = provider.run_stage0_nlu({"stage": "stage1", "current_user_message": "hello"})
        self.assertIsInstance(frame, Stage0NLUFrame)
        self.assertEqual("stage0_nlu_frame.v1", frame.schema_version)
        self.assertEqual(1, len(transport.calls))

    def test_stage1_prompt_rejects_wrapped_contract_and_includes_memory_selection_rules(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not wrap in Stage1Decision", prompt)
        self.assertIn('"schema_version":"0.4.3"', prompt)
        self.assertIn("decision_type", prompt)
        self.assertIn("draft_answer", prompt)
        self.assertIn("If memory_manifest is empty, use decision_type=\"answer_directly\"", prompt)
        self.assertIn("Never choose decision_type=\"request_memory\" with empty selected_memory_ids", prompt)
        self.assertIn(
            "selected_memory_ids contains at least one memory_id copied exactly from memory_manifest",
            prompt,
        )
        self.assertIn('never use request_memory', prompt)
        self.assertIn("recent_messages and answer_directly", prompt)

    def test_stage1_prompt_requires_memory_update_extraction_status(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Every Stage 1 response must include memory_update_extraction", prompt)
        self.assertIn("A memory update is information", prompt)
        self.assertIn("changes Brain's useful understanding of the past, present, or future", prompt)
        self.assertIn("understanding of the past", prompt)
        self.assertIn("present situation", prompt)
        self.assertIn("future expectation", prompt)
        self.assertIn("Extract all distinct memory-relevant updates", prompt)
        self.assertIn("Do not stop after finding one update", prompt)
        self.assertIn("separate memory candidate", prompt)
        self.assertIn("Exclude filler, repetitions, decorative details, and low-value noise", prompt)
        self.assertIn('memory_update_extraction.status="ok" only when memory_candidates is non-empty', prompt)
        self.assertIn('memory_update_extraction.status="fail" when memory_candidates is empty', prompt)
        self.assertIn("Empty memory_candidates must never be silent", prompt)
        self.assertIn("diagnostic only; it is not a CLI, provider, or application failure", prompt)
        self.assertIn("draft_answer should still be produced normally", prompt)
        self.assertIn("Sensitive does not mean forbidden", prompt)
        self.assertIn("Passwords, bank keys, API tokens, seed phrases, and similar secrets are not ordinary memory updates", prompt)

    def test_stage0_prompt_contains_nlu_frame_guidance(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "Normalized intent.",
                    "dialogue_acts": ["question"],
                    "entities": [],
                    "new_information": {
                        "status": "none",
                        "kind": "none",
                        "summary": "",
                        "needs_confirmation": False,
                    },
                    "clarification": {"needed": False, "question": ""},
                    "memory_selection_hint": {"needed": False, "reason": "", "query_terms": []},
                }
            )
        )
        provider.run_stage0_nlu({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Normalize current_user_message into conversational intent", prompt)
        self.assertIn("Classify dialogue act", prompt)
        self.assertIn("Extract structured entities/references", prompt)
        self.assertIn("Detect whether current_user_message introduces new durable information", prompt)
        self.assertIn("ambiguity requires one clarification question", prompt)
        self.assertIn("recent_messages may help interpret current_user_message", prompt)
        self.assertIn("previous_track_analysis_saved may help interpret current_user_message", prompt)
        self.assertIn("Neither recent_messages nor previous_track_analysis_saved are sources of new information", prompt)
        self.assertIn("New durable information must come from current_user_message after normalization", prompt)
        self.assertIn("Do not use keyword matching", prompt)
        self.assertIn("Do not use regex", prompt)
        self.assertIn("Do not use phrase-trigger lists", prompt)
        self.assertIn("Do not add Russian examples", prompt)
        self.assertIn("Do not hardcode live names or live text", prompt)
        self.assertIn("Your job is not to answer", prompt)
        self.assertIn("Do not create memory_candidates", prompt)
        self.assertIn("Do not select memory IDs yet", prompt)

    def test_stage1_prompt_uses_stage0_frame_when_present(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("If stage0_nlu_frame is present, use normalized_intent as the primary interpretation", prompt)
        self.assertIn("Answer normalized intent, not just surface wording", prompt)
        self.assertIn("Use dialogue_acts and new_information from stage0_nlu_frame", prompt)
        self.assertIn("If clarification.needed=true in stage0_nlu_frame, ask that clarification question naturally", prompt)
        self.assertIn("Do not treat Stage 0 as final truth; it is an interpretation frame", prompt)

    def test_stage1_prompt_keeps_candidate_extraction_on_answer_directly_route(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('memory_candidates never requires decision_type="request_memory" by itself', prompt)
        self.assertIn('only when existing durable memory must be read and selected_memory_ids is non-empty', prompt)
        self.assertIn(
            'If the task can be handled from current_user_message, recent_messages, previous analysis, and/or candidate extraction, use decision_type="answer_directly"',
            prompt,
        )
        self.assertIn(
            'Person, name, or alias candidate extraction from current_user_message should normally use decision_type="answer_directly" unless existing durable memory is genuinely needed',
            prompt,
        )

    def test_stage1_prompt_requires_non_empty_selected_memory_ids_for_request_memory(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('If memory_manifest is empty, use decision_type="answer_directly" and never use request_memory', prompt)
        self.assertIn('Never choose decision_type="request_memory" with empty selected_memory_ids', prompt)
        self.assertIn(
            'Only choose decision_type="request_memory" when selected_memory_ids contains at least one memory_id copied exactly from memory_manifest',
            prompt,
        )

    def test_stage1_prompt_contains_semantic_memory_capture_rule(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("semantically asks", prompt)
        self.assertIn("retain information for future use", prompt)
        self.assertIn("not keyword matching", prompt)
        self.assertIn("across languages", prompt)
        self.assertIn("create at least one memory_candidates item", prompt)

    def test_stage1_prompt_contains_required_memory_candidate_shape(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('"candidate_type":"fact"', prompt)
        self.assertIn('"content":{"text":"<concise fact extracted from the user message>"}', prompt)
        self.assertIn('"recommended_action":"stage"', prompt)
        self.assertIn('"confidence":0.8', prompt)

    def test_stage1_prompt_supports_safe_person_mention_candidates(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("mentions a person, persona, or named individual", prompt)
        self.assertIn(
            "even when the surrounding request is sensitive, private, sexual, or otherwise not appropriate to answer directly",
            prompt,
        )
        self.assertIn("only non-sensitive identifying information", prompt)
        self.assertIn(
            "Do not include sexual claims, sexual judgments, private speculation, invasive attributes, or the sensitive request itself",
            prompt,
        )
        self.assertIn("must still be refused or safely redirected in draft_answer", prompt)

    def test_stage1_prompt_supports_safe_relation_candidates(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("If current_user_message states a safe non-sensitive relationship", prompt)
        self.assertIn("friend, colleague, acquaintance, family member, partner, client, coworker, neighbor", prompt)
        self.assertIn('"candidate_type":"relation"', prompt)
        self.assertIn('"subject":"user"', prompt)
        self.assertIn('"relation":"<safe relationship role>"', prompt)
        self.assertIn('"object":"<person name or alias exactly as mentioned>"', prompt)
        self.assertIn('"recommended_action":"stage"', prompt)
        self.assertIn('"confidence":0.8', prompt)

    def test_stage1_prompt_supports_sensitive_biographical_context_without_moralizing(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Sensitive context is not automatically discarded", prompt)
        self.assertIn("may create a careful user-reported context candidate", prompt)
        self.assertIn("Represent sensitive biographical context as user-reported context, not verified truth", prompt)
        self.assertIn('Do not imply it is shameful, degrading, or dirty', prompt)
        self.assertIn('"claim_status":"user_reported"', prompt)
        self.assertIn('"sensitivity":"high"', prompt)
        self.assertIn('"context_type":"biographical_context"', prompt)
        self.assertIn('"confidence":0.6', prompt)
        self.assertIn('do not use save_immediately', prompt)

    def test_stage1_prompt_contains_safe_person_and_alias_candidate_shapes(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('"candidate_type":"person"', prompt)
        self.assertIn('"content":{"display_name":"<person name or alias exactly as mentioned>"}', prompt)
        self.assertIn('"candidate_type":"name_alias"', prompt)
        self.assertIn('"content":{"raw_name":"<name or alias exactly as mentioned>"}', prompt)

    def test_stage1_prompt_forbids_fact_candidate_for_sensitive_claim(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not create a fact candidate that stores the sensitive claim itself", prompt)

    def test_stage1_prompt_warns_against_claiming_permanent_memory_application(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("captured, noted, or recorded as a memory candidate", prompt)
        self.assertIn("remembered", prompt)
        self.assertIn("will be remembered", prompt)
        self.assertIn("saved", prompt)
        self.assertIn("stored", prompt)
        self.assertIn("committed", prompt)
        self.assertIn("written to memory", prompt)
        self.assertIn("permanently saved", prompt)
        self.assertIn("applied to long-term memory", prompt)

    def test_stage1_prompt_does_not_contain_hardcoded_user_fact_examples(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        forbidden_user_fact_examples = (
            "\u041f\u0430\u0432",
            "\u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u043d\u044b\u0435 \u0434\u0438\u0430\u0433\u0440\u0430\u043c\u043c\u044b",
            "\u0050\u0061\u0076",
            "\u0061\u0072\u0063\u0068\u0069\u0074\u0065\u0063\u0074\u0075\u0072\u0065 \u0064\u0069\u0061\u0067\u0072\u0061\u006d\u0073",
        )
        for forbidden in forbidden_user_fact_examples:
            self.assertNotIn(forbidden, prompt)

    def test_stage1_prompt_keeps_provider_instructions_in_english_only(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("keep these prompt instructions in English", prompt)
        self.assertNotRegex(prompt, r"[\u0400-\u04FF]")

    def test_stage1_prompt_guides_curious_safe_follow_up_behavior(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Show strong, respectful curiosity about the user", prompt)
        self.assertIn("Actively invite safe context when it would help the conversation", prompt)
        self.assertIn("Treat current_user_message as the primary task for the current turn", prompt)
        self.assertIn("Memory candidate extraction is secondary to answering the current_user_message safely and helpfully", prompt)
        self.assertIn(
            "Do not repeat a previous memory-candidate acknowledgement when current_user_message asks a new follow-up question",
            prompt,
        )

    def test_stage1_prompt_uses_context_without_re_emitting_candidates_from_old_context(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn(
            "Use recent_messages and previous_track_analysis_saved as context, but do not let them override current_user_message",
            prompt,
        )
        self.assertIn(
            "Do not emit a memory_candidate only because an entity appears in recent_messages or previous_track_analysis_saved",
            prompt,
        )
        self.assertIn("Emit memory_candidates primarily from new information in current_user_message", prompt)
        self.assertIn(
            "do not emit it again unless current_user_message adds new safe identifying information",
            prompt,
        )

    def test_stage1_prompt_answers_interest_follow_ups_naturally_with_safe_curiosity(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn(
            "respond affirmatively in a safe way and invite neutral context",
            prompt,
        )
        self.assertIn(
            'General behavior example: "Yes, I am interested in understanding the context, as long as we discuss it respectfully and avoid invasive claims."',
            prompt,
        )
        self.assertIn(
            "Draft_answer should be natural conversational text, not analysis-style wording, unless the user explicitly asks for analysis",
            prompt,
        )
        self.assertIn("Avoid evasive repetition", prompt)

    def test_stage1_prompt_keeps_sensitive_boundaries_while_allowing_safe_candidates(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn(
            "The sensitive part of the request must still be refused or safely redirected in draft_answer",
            prompt,
        )
        self.assertIn("Do not create a fact candidate that stores the sensitive claim itself", prompt)
        self.assertIn(
            "Do not create ordinary fact candidates that make sexual judgments, private speculation, or invasive conclusions",
            prompt,
        )

    def test_stage1_prompt_guides_do_you_know_behavior(self) -> None:
        provider, transport = self._provider(self._stage1_response())
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("answer yes only if the person is known from recent_messages", prompt)
        self.assertIn("previous_track_analysis_saved", prompt)
        self.assertIn("retrieved durable memory", prompt)
        self.assertIn("If the person is not known, say clearly that you do not know who that person is yet", prompt)
        self.assertIn(
            'I do not know who that is yet, but I am interested in understanding the context. Who are they to you?',
            prompt,
        )

    def test_stage2_valid_fake_http_response_returns_decision(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "schema_version": "0.4.3",
                    "final_answer": "Provider returned a final answer.",
                    "memory_candidates": [],
                    "memory_update_extraction": {
                        "status": "fail",
                        "reason": "No durable information extracted.",
                    },
                    "used_memory_ids": ["mem_1"],
                }
            )
        )
        decision = provider.decide_stage2({"stage": "stage2", "selected_memory_context": []})
        self.assertIsInstance(decision, Stage2Decision)
        self.assertEqual("Provider returned a final answer.", decision.final_answer)
        self.assertEqual(["mem_1"], decision.used_memory_ids)
        self.assertEqual(1, len(transport.calls))

    def test_stage2_prompt_rejects_wrapped_contract_and_names_final_answer(self) -> None:
        provider, transport = self._provider(self._stage2_response())
        provider.decide_stage2({"stage": "stage2"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not wrap in Stage2Decision", prompt)
        self.assertIn('"schema_version":"0.4.3"', prompt)
        self.assertIn("final_answer", prompt)

    def test_stage2_prompt_requires_memory_update_extraction_status(self) -> None:
        provider, transport = self._provider(self._stage2_response())
        provider.decide_stage2({"stage": "stage2"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Every Stage 2 response must include memory_update_extraction", prompt)
        self.assertIn("A memory update is information", prompt)
        self.assertIn("changes Brain's useful understanding of the past, present, or future", prompt)
        self.assertIn("understanding of the past", prompt)
        self.assertIn("present situation", prompt)
        self.assertIn("future expectation", prompt)
        self.assertIn("Extract all distinct memory-relevant updates", prompt)
        self.assertIn("Do not stop after finding one update", prompt)
        self.assertIn("separate memory candidate", prompt)
        self.assertIn("Exclude filler, repetitions, decorative details, and low-value noise", prompt)
        self.assertIn('memory_update_extraction.status="ok" only when memory_candidates is non-empty', prompt)
        self.assertIn('memory_update_extraction.status="fail" when memory_candidates is empty', prompt)
        self.assertIn("Empty memory_candidates must never be silent", prompt)
        self.assertIn("diagnostic only; it is not a CLI, provider, or application failure", prompt)
        self.assertIn("final_answer should still be produced normally", prompt)
        self.assertIn("Sensitive does not mean forbidden", prompt)
        self.assertIn("Passwords, bank keys, API tokens, seed phrases, and similar secrets are not ordinary memory updates", prompt)

    def test_stage0_response_with_draft_answer_fails(self) -> None:
        provider, _transport = self._provider(
            json.dumps(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "Intent.",
                    "dialogue_acts": ["question"],
                    "entities": [],
                    "new_information": {
                        "status": "none",
                        "kind": "none",
                        "summary": "",
                        "needs_confirmation": False,
                    },
                    "clarification": {"needed": False, "question": ""},
                    "memory_selection_hint": {"needed": False, "reason": "", "query_terms": []},
                    "draft_answer": "not allowed",
                }
            )
        )
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.run_stage0_nlu({"stage": "stage1"})

    def test_stage0_response_with_memory_candidates_fails(self) -> None:
        provider, _transport = self._provider(
            json.dumps(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "Intent.",
                    "dialogue_acts": ["question"],
                    "entities": [],
                    "new_information": {
                        "status": "none",
                        "kind": "none",
                        "summary": "",
                        "needs_confirmation": False,
                    },
                    "clarification": {"needed": False, "question": ""},
                    "memory_selection_hint": {"needed": False, "reason": "", "query_terms": []},
                    "memory_candidates": [],
                }
            )
        )
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.run_stage0_nlu({"stage": "stage1"})

    def test_stage0_response_with_selected_memory_ids_fails(self) -> None:
        provider, _transport = self._provider(
            json.dumps(
                {
                    "schema_version": "stage0_nlu_frame.v1",
                    "normalized_intent": "Intent.",
                    "dialogue_acts": ["question"],
                    "entities": [],
                    "new_information": {
                        "status": "none",
                        "kind": "none",
                        "summary": "",
                        "needs_confirmation": False,
                    },
                    "clarification": {"needed": False, "question": ""},
                    "memory_selection_hint": {"needed": False, "reason": "", "query_terms": []},
                    "selected_memory_ids": [],
                }
            )
        )
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.run_stage0_nlu({"stage": "stage1"})

    def test_invalid_json_response_fails_clearly(self) -> None:
        provider, _transport = self._provider("not json")
        with self.assertRaisesRegex(ProviderResponseError, "invalid decision JSON"):
            provider.decide_stage1({"stage": "stage1"})

    def test_contract_invalid_response_fails_clearly(self) -> None:
        provider, _transport = self._provider(json.dumps({"decision_type": "request_memory"}))
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.decide_stage1({"stage": "stage1"})

    def test_no_real_network_call_is_made_in_tests(self) -> None:
        provider, transport = self._provider(self._stage2_response())
        provider.decide_stage2({"stage": "stage2"})
        self.assertEqual(1, len(transport.calls))
        self.assertIsInstance(transport, FakeTransport)

    def test_provider_rejects_returned_summary_field_through_contract(self) -> None:
        provider, _transport = self._provider(
            json.dumps(
                {
                    "final_answer": "Done.",
                    "summary": "not allowed",
                }
            )
        )
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.decide_stage2({"stage": "stage2"})

    def test_from_env_uses_env_without_printing_or_real_network(self) -> None:
        transport = FakeTransport({"choices": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._working_directory(temp_dir):
                env = {
                    LLM_BASE_URL_ENV: "https://llm.example.test/v1/",
                    LLM_API_KEY_ENV: "test_api_key",
                    LLM_MODEL_ENV: "test_model",
                }
                with patch.dict(os.environ, env, clear=True):
                    provider = OpenAICompatibleLLMProvider.from_env(transport=transport)
        self.assertEqual("https://llm.example.test/v1", provider.base_url)
        self.assertEqual("test_api_key", provider.api_key)
        self.assertEqual("test_model", provider.model)

    def test_from_env_loads_project_env_file_when_present(self) -> None:
        transport = FakeTransport({"choices": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "MNEMOSYNE_LLM_BASE_URL=https://llm.example.test/v1",
                        "MNEMOSYNE_LLM_API_KEY=file_key",
                        "MNEMOSYNE_LLM_MODEL=file_model",
                    ]
                ),
                encoding="utf-8",
            )
            with self._working_directory(temp_dir):
                with patch.dict(os.environ, {}, clear=True):
                    provider = OpenAICompatibleLLMProvider.from_env(transport=transport)
        self.assertEqual("https://llm.example.test/v1", provider.base_url)
        self.assertEqual("file_key", provider.api_key)
        self.assertEqual("file_model", provider.model)
