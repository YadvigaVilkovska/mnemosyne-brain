"""LLM provider adapter tests."""

from __future__ import annotations

import json
import os
import unittest
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

    def test_stage2_valid_fake_http_response_returns_decision(self) -> None:
        provider, transport = self._provider(
            json.dumps(
                {
                    "final_answer": "Pav loves architecture diagrams.",
                    "used_memory_ids": ["mem_1"],
                }
            )
        )
        decision = provider.decide_stage2({"stage": "stage2", "selected_memory_context": []})
        self.assertIsInstance(decision, Stage2Decision)
        self.assertEqual("Pav loves architecture diagrams.", decision.final_answer)
        self.assertEqual(["mem_1"], decision.used_memory_ids)
        self.assertEqual(1, len(transport.calls))

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
