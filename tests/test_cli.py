"""CLI tests."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import redirect_stdout
from unittest.mock import patch

from mnemosyne_brain.app.cli import main
from mnemosyne_brain.app.llm_provider import ProviderResponseError


class CliTestCase(unittest.TestCase):
    """Verifies the CLI sends one message through the graph."""

    def _run_cli_with_env(self, message: str, env: dict[str, str]) -> tuple[int, str]:
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with redirect_stdout(output):
                exit_code = main([message])
        return exit_code, output.getvalue()

    def test_cli_prints_response_track_and_no_capsule_for_local_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli.sqlite3")
            exit_code, rendered = self._run_cli_with_env(
                "Remember that Pav loves architecture diagrams",
                {"MNEMOSYNE_DB_PATH": db_path},
            )

        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: Remember that Pav loves architecture diagrams", rendered)
        self.assertIn("Track: trk_", rendered)
        self.assertIn("Capsule: none", rendered)

    def test_missing_llm_env_vars_use_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_fallback.sqlite3")
            exit_code, rendered = self._run_cli_with_env(
                "fallback message",
                {
                    "MNEMOSYNE_DB_PATH": db_path,
                    "MNEMOSYNE_LLM_BASE_URL": "https://llm.example.test/v1",
                },
            )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: fallback message", rendered)

    def test_all_llm_env_vars_use_llm_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_llm.sqlite3")
            env = self._llm_env(db_path)
            with self._patched_llm_path({"answer": "Provider answer."}) as calls:
                exit_code, rendered = self._run_cli_with_env("hello", env)

        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Provider answer.", rendered)
        self.assertIn("Track: trk_", rendered)
        self.assertIn("Capsule: none", rendered)
        self.assertEqual(1, calls["provider_from_env"])
        self.assertEqual(["hello"], calls["messages"])

    def test_answer_directly_result_prints_provider_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_llm_path({"answer": "Direct provider answer."}):
                _exit_code, rendered = self._run_cli_with_env(
                    "hello",
                    self._llm_env(os.path.join(temp_dir, "mnemosyne_cli_direct.sqlite3")),
                )
        self.assertIn("Assistant: Direct provider answer.", rendered)

    def test_request_memory_result_prints_stage2_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_llm_path({"answer": "Stage 2 final answer."}):
                _exit_code, rendered = self._run_cli_with_env(
                    "hello",
                    self._llm_env(os.path.join(temp_dir, "mnemosyne_cli_stage2.sqlite3")),
                )
        self.assertIn("Assistant: Stage 2 final answer.", rendered)

    def test_configured_provider_failure_is_not_silently_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_failure.sqlite3")
            with patch(
                "mnemosyne_brain.app.cli.OpenAICompatibleLLMProvider.from_env",
                side_effect=ProviderResponseError("configured provider failed"),
            ):
                with self.assertRaisesRegex(ProviderResponseError, "configured provider failed"):
                    self._run_cli_with_env("hello", self._llm_env(db_path))

    def test_no_real_network_call_in_llm_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_llm_path({"answer": "No network answer."}) as calls:
                self._run_cli_with_env(
                    "hello",
                    self._llm_env(os.path.join(temp_dir, "mnemosyne_cli_no_network.sqlite3")),
                )
        self.assertEqual(1, calls["provider_from_env"])
        self.assertEqual(1, calls["orchestrator_runs"])

    def _llm_env(self, db_path: str) -> dict[str, str]:
        return {
            "MNEMOSYNE_DB_PATH": db_path,
            "MNEMOSYNE_LLM_BASE_URL": "https://llm.example.test/v1",
            "MNEMOSYNE_LLM_API_KEY": "test_api_key",
            "MNEMOSYNE_LLM_MODEL": "test_model",
        }

    @contextmanager
    def _patched_llm_path(self, result: dict) -> Iterator[dict]:
        calls = {"provider_from_env": 0, "orchestrator_runs": 0, "messages": []}

        class FakeProvider:
            """Sentinel provider that performs no HTTP."""

        class FakeProviderFactory:
            """Factory matching OpenAICompatibleLLMProvider.from_env."""

            @staticmethod
            def from_env():
                calls["provider_from_env"] += 1
                return FakeProvider()

        class FakeOrchestrator:
            """Orchestrator fake that records input and returns a configured answer."""

            def __init__(self, repository, adapter) -> None:
                self._repository = repository
                self._adapter = adapter

            def run_turn(self, track_id: str, current_user_message: str) -> dict:
                calls["orchestrator_runs"] += 1
                calls["messages"].append(current_user_message)
                return result

        with patch.multiple(
            "mnemosyne_brain.app.cli",
            OpenAICompatibleLLMProvider=FakeProviderFactory,
            DeterministicLLMOrchestrator=FakeOrchestrator,
        ):
            yield calls
