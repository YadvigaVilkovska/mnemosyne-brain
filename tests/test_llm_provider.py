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

from mnemosyne_brain.app.contracts.analysis import Stage1Decision, Stage2Decision
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
                    "decision_type": "answer_directly",
                    "draft_answer": "Already enough context.",
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

    def test_stage1_prompt_rejects_wrapped_contract_and_includes_memory_selection_rules(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not wrap in Stage1Decision", prompt)
        self.assertIn("decision_type", prompt)
        self.assertIn("draft_answer", prompt)
        self.assertIn("If memory_manifest is empty, use decision_type=\"answer_directly\"", prompt)
        self.assertIn("Never choose decision_type=\"request_memory\" with empty selected_memory_ids", prompt)
        self.assertIn(
            "selected_memory_ids contains at least one memory_id copied exactly from memory_manifest",
            prompt,
        )
        self.assertIn("recent_messages and answer_directly", prompt)

    def test_stage1_prompt_contains_semantic_memory_capture_rule(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("semantically asks", prompt)
        self.assertIn("retain information for future use", prompt)
        self.assertIn("not keyword matching", prompt)
        self.assertIn("across languages", prompt)
        self.assertIn("create at least one memory_candidates item", prompt)

    def test_stage1_prompt_contains_required_memory_candidate_shape(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('"candidate_type":"fact"', prompt)
        self.assertIn('"content":{"text":"<concise fact extracted from the user message>"}', prompt)
        self.assertIn('"recommended_action":"stage"', prompt)
        self.assertIn('"confidence":0.8', prompt)

    def test_stage1_prompt_supports_safe_person_mention_candidates(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
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

    def test_stage1_prompt_contains_safe_person_and_alias_candidate_shapes(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn('"candidate_type":"person"', prompt)
        self.assertIn('"content":{"display_name":"<person name or alias exactly as mentioned>"}', prompt)
        self.assertIn('"candidate_type":"name_alias"', prompt)
        self.assertIn('"content":{"raw_name":"<name or alias exactly as mentioned>"}', prompt)

    def test_stage1_prompt_forbids_fact_candidate_for_sensitive_claim(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not create a fact candidate that stores the sensitive claim itself", prompt)

    def test_stage1_prompt_warns_against_claiming_permanent_memory_application(self) -> None:
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
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
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
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
        provider, transport = self._provider(json.dumps({"decision_type": "answer_directly", "draft_answer": "Done."}))
        provider.decide_stage1({"stage": "stage1"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("keep these prompt instructions in English", prompt)
        self.assertNotRegex(prompt, r"[\u0400-\u04FF]")

    def test_stage2_valid_fake_http_response_returns_decision(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "final_answer": "Provider returned a final answer.",
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
        provider, transport = self._provider(json.dumps({"final_answer": "Done."}))
        provider.decide_stage2({"stage": "stage2"})
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("Do not wrap in Stage2Decision", prompt)
        self.assertIn("final_answer", prompt)

    def test_invalid_json_response_fails_clearly(self) -> None:
        provider, _transport = self._provider("not json")
        with self.assertRaisesRegex(ProviderResponseError, "invalid decision JSON"):
            provider.decide_stage1({"stage": "stage1"})

    def test_contract_invalid_response_fails_clearly(self) -> None:
        provider, _transport = self._provider(json.dumps({"decision_type": "request_memory"}))
        with self.assertRaisesRegex(ProviderResponseError, "contract validation"):
            provider.decide_stage1({"stage": "stage1"})

    def test_no_real_network_call_is_made_in_tests(self) -> None:
        provider, transport = self._provider(json.dumps({"final_answer": "Done."}))
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
