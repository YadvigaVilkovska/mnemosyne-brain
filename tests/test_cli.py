"""CLI tests."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mnemosyne_brain.app.cli import llm_env_is_configured, main
from mnemosyne_brain.app.llm_provider import OpenAICompatibleLLMProvider, ProviderResponseError


class CliTestCase(unittest.TestCase):
    """Verifies the CLI sends one message through the graph."""

    @contextmanager
    def _working_directory(self, path: str) -> Iterator[None]:
        previous = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    def _run_cli_with_env(self, message: str, env: dict[str, str]) -> tuple[int, str]:
        return self._run_cli_args_with_env([message], env)

    def _run_cli_args_with_env(self, argv: list[str], env: dict[str, str]) -> tuple[int, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(argv)
        return exit_code, stdout.getvalue() + stderr.getvalue()

    def test_cli_prints_response_track_and_no_capsule_for_local_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli.sqlite3")
            with self._working_directory(temp_dir):
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
            with self._working_directory(temp_dir):
                exit_code, rendered = self._run_cli_with_env(
                    "fallback message",
                    {
                        "MNEMOSYNE_DB_PATH": db_path,
                        "MNEMOSYNE_LLM_BASE_URL": "https://llm.example.test/v1",
                    },
                )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: fallback message", rendered)

    def test_cli_sees_llm_env_vars_from_project_env_file_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_envfile.sqlite3")
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "# comment",
                        "",
                        "MNEMOSYNE_LLM_BASE_URL=https://llm.example.test/v1",
                        "MNEMOSYNE_LLM_API_KEY=env_file_key",
                        "MNEMOSYNE_LLM_MODEL=test_model",
                    ]
                ),
                encoding="utf-8",
            )
            with self._working_directory(temp_dir):
                with self._patched_llm_path({"answer": "Provider answer from env file."}) as calls:
                    exit_code, rendered = self._run_cli_with_env(
                        "hello",
                        {"MNEMOSYNE_DB_PATH": db_path},
                    )

        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Provider answer from env file.", rendered)
        self.assertEqual(1, calls["provider_from_env"])

    def test_llm_env_is_configured_reads_project_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "MNEMOSYNE_LLM_BASE_URL=https://llm.example.test/v1",
                        "MNEMOSYNE_LLM_API_KEY=env_file_key",
                        "MNEMOSYNE_LLM_MODEL=test_model",
                    ]
                ),
                encoding="utf-8",
            )
            with self._working_directory(temp_dir):
                with patch.dict(os.environ, {}, clear=True):
                    self.assertTrue(llm_env_is_configured())

    def test_existing_process_env_overrides_project_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_env_override.sqlite3")
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
            observed = {}
            with self._working_directory(temp_dir):
                def fail_with_observed_env():
                    observed["base_url"] = os.environ.get("MNEMOSYNE_LLM_BASE_URL")
                    observed["api_key"] = os.environ.get("MNEMOSYNE_LLM_API_KEY")
                    observed["model"] = os.environ.get("MNEMOSYNE_LLM_MODEL")
                    raise ProviderResponseError("process value wins")

                with patch(
                    "mnemosyne_brain.app.cli.OpenAICompatibleLLMProvider.from_env",
                    side_effect=fail_with_observed_env,
                ) as provider_from_env:
                    exit_code, rendered = self._run_cli_with_env(
                        "hello",
                        {
                            "MNEMOSYNE_DB_PATH": db_path,
                            "MNEMOSYNE_LLM_BASE_URL": "https://process.example.test/v1",
                            "MNEMOSYNE_LLM_API_KEY": "process_key",
                            "MNEMOSYNE_LLM_MODEL": "process_model",
                        },
                    )
        self.assertEqual(1, exit_code)
        self.assertIn("LLM failed", rendered)
        self.assertIn("user turn saved", rendered)
        self.assertIn("assistant turn not saved", rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertEqual(1, provider_from_env.call_count)
        self.assertEqual("https://process.example.test/v1", observed["base_url"])
        self.assertEqual("process_key", observed["api_key"])
        self.assertEqual("process_model", observed["model"])

    def test_missing_project_env_does_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_missing_env.sqlite3")
            with self._working_directory(temp_dir):
                exit_code, rendered = self._run_cli_with_env(
                    "fallback message",
                    {"MNEMOSYNE_DB_PATH": db_path},
                )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: fallback message", rendered)

    def test_comments_and_blank_lines_in_project_env_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_env_comments.sqlite3")
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "",
                        "# comment",
                        "MNEMOSYNE_LLM_BASE_URL=https://llm.example.test/v1",
                        "",
                        "MNEMOSYNE_LLM_API_KEY=env_file_key",
                        "   ",
                        "MNEMOSYNE_LLM_MODEL=test_model",
                    ]
                ),
                encoding="utf-8",
            )
            with self._working_directory(temp_dir):
                with self._patched_llm_path({"answer": "Provider answer from env file."}) as calls:
                    exit_code, rendered = self._run_cli_with_env(
                        "hello",
                        {"MNEMOSYNE_DB_PATH": db_path},
                    )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Provider answer from env file.", rendered)
        self.assertEqual(1, calls["provider_from_env"])

    def test_no_secrets_are_printed_when_loading_project_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_env_secret.sqlite3")
            Path(temp_dir, ".env").write_text(
                "\n".join(
                    [
                        "MNEMOSYNE_LLM_BASE_URL=https://llm.example.test/v1",
                        "MNEMOSYNE_LLM_API_KEY=super_secret_value",
                        "MNEMOSYNE_LLM_MODEL=test_model",
                    ]
                ),
                encoding="utf-8",
            )
            with self._working_directory(temp_dir):
                with self._patched_llm_path({"answer": "Safe answer."}):
                    _exit_code, rendered = self._run_cli_with_env(
                        "hello",
                        {"MNEMOSYNE_DB_PATH": db_path},
                    )
        self.assertNotIn("super_secret_value", rendered)

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

    def test_thread_id_argument_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_thread_arg.sqlite3")
            with self._patched_llm_path({"answer": "Thread answer."}):
                exit_code, rendered = self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "hello"],
                    self._llm_env(db_path),
                )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Thread answer.", rendered)

    def test_two_llm_runs_with_same_thread_id_reuse_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_same_thread.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                _first_exit, first_rendered = self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "First message"],
                    self._llm_env(db_path),
                )
                _second_exit, second_rendered = self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "Continue this"],
                    self._llm_env(db_path),
                )
        self.assertEqual(self._track_from_output(first_rendered), self._track_from_output(second_rendered))

    def test_second_llm_run_has_previous_turns_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_context_thread.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}) as calls:
                self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "First message"],
                    self._llm_env(db_path),
                )
                self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "Continue this"],
                    self._llm_env(db_path),
                )
        self.assertEqual(
            ["First message", "Assistant answer."],
            calls["recent_texts_at_run"][1],
        )

    def test_llm_path_without_thread_id_creates_new_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_new_threads.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                _first_exit, first_rendered = self._run_cli_with_env("First message", self._llm_env(db_path))
                _second_exit, second_rendered = self._run_cli_with_env("Second message", self._llm_env(db_path))
        self.assertNotEqual(self._track_from_output(first_rendered), self._track_from_output(second_rendered))

    def test_local_fallback_accepts_thread_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_fallback_thread.sqlite3")
            with self._working_directory(temp_dir):
                exit_code, rendered = self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "fallback message"],
                    {"MNEMOSYNE_DB_PATH": db_path},
                )
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: fallback message", rendered)
        self.assertIn("Track: trk_", rendered)

    def test_llm_path_persists_one_user_turn_and_one_assistant_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_turns.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
        self.assertEqual(
            [("user", "user", "User message."), ("llm", "assistant", "Assistant answer.")],
            [(turn["input_source"], turn["role"], turn["content_text"]) for turn in turns],
        )
        self.assertEqual(1, len(analysis_events))
        self.assertEqual("track_analysis_saved", analysis_events[0]["event_type"])

    def test_user_turn_is_persisted_before_orchestrator_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_user_before.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}) as calls:
                self._run_cli_with_env("User message.", self._llm_env(db_path))
        self.assertEqual([1], calls["turn_counts_at_run"])

    def test_assistant_turn_is_persisted_after_successful_orchestrator_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_assistant_after.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                _exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            turns = self._list_turns(db_path)
        assistant_turns = [turn for turn in turns if turn["role"] == "assistant"]
        self.assertEqual(1, len(assistant_turns))
        self.assertIn(f"Track: {assistant_turns[0]['track_id']}", rendered)

    def test_successful_llm_turn_saves_track_analysis_audit_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_analysis.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                _exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
        track_id = self._track_from_output(rendered)
        self.assertEqual(2, len(turns))
        self.assertEqual(1, len([turn for turn in turns if turn["role"] == "user"]))
        self.assertEqual(1, len([turn for turn in turns if turn["role"] == "assistant"]))
        self.assertEqual(1, len(analysis_events))
        self.assertEqual("llm", analysis_events[0]["actor_type"])
        self.assertEqual("dialogue_track", analysis_events[0]["target_type"])
        self.assertEqual(track_id, analysis_events[0]["target_id"])
        self.assertEqual(track_id, analysis_events[0]["track_id"])

    def test_saved_analysis_payload_contains_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_analysis_keys.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            payload = self._list_track_analysis_events(db_path)[0]["payload"]
        self.assertEqual(
            {
                "route",
                "selected_memory_ids",
                "used_memory_ids",
                "stage1_decision",
                "stage2_decision",
                "extracted_facts",
                "memory_candidates",
            },
            set(payload),
        )

    def test_answer_directly_analysis_uses_stage1_facts_and_candidates(self) -> None:
        stage1_facts = [{"fact": "stage1_fact"}]
        stage1_candidates = [{"candidate": "stage1_candidate"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_analysis_direct.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Direct answer.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Direct answer.",
                        "extracted_facts": stage1_facts,
                        "memory_candidates": stage1_candidates,
                        "rationale": "enough context",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            payload = self._list_track_analysis_events(db_path)[0]["payload"]
        self.assertEqual("answer_directly", payload["route"])
        self.assertIsNone(payload["stage2_decision"])
        self.assertEqual(stage1_facts, payload["extracted_facts"])
        self.assertEqual(stage1_candidates, payload["memory_candidates"])
        self.assertEqual(
            {"status": "ok", "reason": "Durable information extracted."},
            payload["stage1_decision"]["memory_update_extraction"],
        )

    def test_stage2_analysis_combines_facts_and_candidates_in_order(self) -> None:
        stage1_facts = [{"fact": "stage1_a"}, {"fact": "stage1_b"}]
        stage2_facts = [{"fact": "stage2_a"}]
        stage1_candidates = [{"candidate": "stage1_a"}]
        stage2_candidates = [{"candidate": "stage2_a"}, {"candidate": "stage2_b"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_analysis_stage2.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Stage 2 answer.",
                    "route": "used_selected_memory",
                    "selected_memory_ids": ["mem_1"],
                    "used_memory_ids": ["mem_1"],
                    "stage1_decision": {
                        "decision_type": "request_memory",
                        "selected_memory_ids": ["mem_1"],
                        "draft_answer": None,
                        "extracted_facts": stage1_facts,
                        "memory_candidates": stage1_candidates,
                        "rationale": "needs memory",
                    },
                    "stage2_decision": {
                        "final_answer": "Stage 2 answer.",
                        "extracted_facts": stage2_facts,
                        "memory_candidates": stage2_candidates,
                        "used_memory_ids": ["mem_1"],
                        "rationale": "used memory",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            payload = self._list_track_analysis_events(db_path)[0]["payload"]
        self.assertEqual("used_selected_memory", payload["route"])
        self.assertIsNotNone(payload["stage2_decision"])
        self.assertEqual(stage1_facts + stage2_facts, payload["extracted_facts"])
        self.assertEqual(stage1_candidates + stage2_candidates, payload["memory_candidates"])
        self.assertEqual(
            {"status": "ok", "reason": "Durable information extracted."},
            payload["stage1_decision"]["memory_update_extraction"],
        )
        self.assertEqual(
            {"status": "ok", "reason": "Durable information extracted."},
            payload["stage2_decision"]["memory_update_extraction"],
        )

    def test_successful_llm_turn_persists_one_valid_stage1_memory_candidate(self) -> None:
        content = {"subject": "Pav", "predicate": "likes", "object": "architecture diagrams"}
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_stage1_candidate.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Saved as candidate.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Saved as candidate.",
                        "extracted_facts": [],
                        "memory_candidates": [
                            {
                                "candidate_type": "fact",
                                "content": content,
                                "recommended_action": "save_immediately",
                                "confidence": 0.92,
                            }
                        ],
                        "rationale": "candidate found",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
            turns = self._list_turns(db_path)
        self.assertEqual(1, len(candidates))
        self.assertEqual("fact", candidates[0]["candidate_type"])
        self.assertEqual("save_immediately", candidates[0]["recommended_action"])
        self.assertEqual(0.92, candidates[0]["confidence"])
        self.assertEqual(content, candidates[0]["content"])
        self.assertEqual("llm", candidates[0]["provenance"]["source"])
        self.assertEqual(turns[0]["track_id"], candidates[0]["track_id"])

    def test_successful_llm_turn_persists_stage1_and_stage2_candidates_in_order(self) -> None:
        stage1_content = {"order": "stage1"}
        stage2_content = {"order": "stage2"}
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_two_candidates.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Stage 2 answer.",
                    "route": "used_selected_memory",
                    "selected_memory_ids": ["mem_1"],
                    "used_memory_ids": ["mem_1"],
                    "stage1_decision": {
                        "decision_type": "request_memory",
                        "selected_memory_ids": ["mem_1"],
                        "draft_answer": None,
                        "extracted_facts": [],
                        "memory_candidates": [
                            {
                                "candidate_type": "fact",
                                "content": stage1_content,
                                "recommended_action": "save_immediately",
                                "confidence": 0.8,
                            }
                        ],
                        "rationale": "needs memory",
                    },
                    "stage2_decision": {
                        "final_answer": "Stage 2 answer.",
                        "extracted_facts": [],
                        "memory_candidates": [
                            {
                                "candidate_type": "fact",
                                "content": stage2_content,
                                "recommended_action": "unsupported",
                                "confidence": 0.7,
                            }
                        ],
                        "used_memory_ids": ["mem_1"],
                        "rationale": "used memory",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
        self.assertEqual([stage1_content, stage2_content], [candidate["content"] for candidate in candidates])
        self.assertEqual(["save_immediately", "stage"], [candidate["recommended_action"] for candidate in candidates])

    def test_duplicate_candidates_in_same_llm_result_are_persisted_once(self) -> None:
        duplicate_candidate = {
            "candidate_type": "name_alias",
            "content": {"raw_name": "L."},
            "recommended_action": "stage",
            "confidence": 0.8,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_duplicate_same_result.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Single semantic candidate kept.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Single semantic candidate kept.",
                        "extracted_facts": [],
                        "memory_candidates": [duplicate_candidate, duplicate_candidate],
                        "rationale": "duplicate candidate",
                    },
                }
            ):
                exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
            memory_item_count = self._count_rows(db_path, "memory_items")
            memory_staging_count = self._count_rows(db_path, "memory_staging")
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Single semantic candidate kept.", rendered)
        self.assertEqual(1, len(candidates))
        self.assertEqual(2, len(turns))
        self.assertEqual(1, len(analysis_events))
        self.assertEqual(0, memory_item_count)
        self.assertEqual(0, memory_staging_count)

    def test_duplicate_candidates_across_llm_runs_for_same_thread_are_skipped(self) -> None:
        duplicate_candidate = {
            "candidate_type": "name_alias",
            "content": {"raw_name": "L."},
            "recommended_action": "stage",
            "confidence": 0.8,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_duplicate_same_track.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Track answer.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Track answer.",
                        "extracted_facts": [],
                        "memory_candidates": [duplicate_candidate],
                        "rationale": "duplicate across runs",
                    },
                }
            ):
                self._run_cli_args_with_env(
                    ["--thread-id", "same-track", "First message"],
                    self._llm_env(db_path),
                )
                self._run_cli_args_with_env(
                    ["--thread-id", "same-track", "Second message"],
                    self._llm_env(db_path),
                )
            candidates = self._list_memory_candidates(db_path)
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
            memory_item_count = self._count_rows(db_path, "memory_items")
            memory_staging_count = self._count_rows(db_path, "memory_staging")
        self.assertEqual(1, len(candidates))
        self.assertEqual(2, len([turn for turn in turns if turn["role"] == "assistant"]))
        self.assertEqual(2, len(analysis_events))
        self.assertEqual(0, memory_item_count)
        self.assertEqual(0, memory_staging_count)

    def test_distinct_candidates_are_still_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_distinct_candidates.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Distinct candidates kept.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Distinct candidates kept.",
                        "extracted_facts": [],
                        "memory_candidates": [
                            {
                                "candidate_type": "relation",
                                "content": {"subject": "user", "relation": "friend", "object": "L."},
                                "recommended_action": "stage",
                                "confidence": 0.8,
                            },
                            {
                                "candidate_type": "relation",
                                "content": {"subject": "user", "relation": "colleague", "object": "L."},
                                "recommended_action": "stage",
                                "confidence": 0.8,
                            },
                        ],
                        "rationale": "distinct relations",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
        self.assertEqual(2, len(candidates))
        self.assertEqual(
            [
                {"subject": "user", "relation": "friend", "object": "L."},
                {"subject": "user", "relation": "colleague", "object": "L."},
            ],
            [candidate["content"] for candidate in candidates],
        )

    def test_duplicate_candidate_skip_is_best_effort_and_does_not_fail_cli(self) -> None:
        duplicate_candidate = {
            "candidate_type": "person",
            "content": {"display_name": "L."},
            "recommended_action": "stage",
            "confidence": 0.8,
        }
        distinct_candidate = {
            "candidate_type": "relation",
            "content": {"subject": "user", "relation": "acquaintance", "object": "L."},
            "recommended_action": "stage",
            "confidence": 0.8,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_duplicate_best_effort.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Duplicates skipped cleanly.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Duplicates skipped cleanly.",
                        "extracted_facts": [],
                        "memory_candidates": [duplicate_candidate, duplicate_candidate, distinct_candidate],
                        "rationale": "best effort duplicate skip",
                    },
                }
            ):
                exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Duplicates skipped cleanly.", rendered)
        self.assertEqual(2, len(candidates))

    def test_invalid_raw_llm_candidates_are_skipped_but_remain_in_analysis_payload(self) -> None:
        invalid_candidates = [
            "not a dict",
            {"candidate_type": "", "content": {"bad": "type"}, "recommended_action": "stage", "confidence": 0.5},
            {"candidate_type": "fact", "content": "not a dict", "recommended_action": "stage", "confidence": 0.5},
            {"candidate_type": "fact", "content": {"bad": "confidence"}, "recommended_action": "stage"},
            {"candidate_type": "fact", "content": {"bad": "range"}, "recommended_action": "stage", "confidence": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_invalid_candidates.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Invalid candidates ignored.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Invalid candidates ignored.",
                        "extracted_facts": [],
                        "memory_candidates": invalid_candidates,
                        "rationale": "invalid candidates",
                    },
                }
            ):
                exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            candidates = self._list_memory_candidates(db_path)
            payload = self._list_track_analysis_events(db_path)[0]["payload"]
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Invalid candidates ignored.", rendered)
        self.assertEqual([], candidates)
        self.assertEqual(invalid_candidates, payload["memory_candidates"])

    def test_assistant_turn_is_not_persisted_if_orchestrator_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_orchestrator_failure.sqlite3")
            with self._patched_llm_path(error=ProviderResponseError("orchestrator failed")):
                exit_code, rendered = self._run_cli_with_env("User message.", self._llm_env(db_path))
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
            candidates = self._list_memory_candidates(db_path)
        self.assertEqual(1, exit_code)
        self.assertIn("LLM failed", rendered)
        self.assertIn("user turn saved", rendered)
        self.assertIn("assistant turn not saved", rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertEqual([("user", "user", "User message.")], [
            (turn["input_source"], turn["role"], turn["content_text"]) for turn in turns
        ])
        self.assertEqual([], analysis_events)
        self.assertEqual([], candidates)

    def test_provider_orchestrator_is_not_called_inside_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_transaction.sqlite3")
            with self._patched_llm_path({"answer": "Assistant answer."}) as calls:
                self._run_cli_with_env("User message.", self._llm_env(db_path))
        self.assertEqual([False], calls["in_transaction_at_run"])

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
                exit_code, rendered = self._run_cli_with_env("hello", self._llm_env(db_path))
            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
            candidates = self._list_memory_candidates(db_path)
        self.assertEqual(1, exit_code)
        self.assertIn("LLM failed", rendered)
        self.assertIn("configured provider failed", rendered)
        self.assertIn("user turn saved", rendered)
        self.assertIn("assistant turn not saved", rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertEqual([("user", "user", "hello")], [
            (turn["input_source"], turn["role"], turn["content_text"]) for turn in turns
        ])
        self.assertEqual([], analysis_events)
        self.assertEqual([], candidates)

    def test_second_llm_turn_sees_previous_analysis_from_first_successful_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_previous_analysis.sqlite3")
            first_result = {
                "answer": "First answer.",
                "stage1_decision": {
                    "decision_type": "answer_directly",
                    "selected_memory_ids": [],
                    "draft_answer": "First answer.",
                    "extracted_facts": [{"fact": "first"}],
                    "memory_candidates": [{"candidate": "first"}],
                    "rationale": "first turn",
                },
            }
            with self._patched_llm_path(first_result) as calls:
                self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "First message"],
                    self._llm_env(db_path),
                )
                self._run_cli_args_with_env(
                    ["--thread-id", "pav-main", "Second message"],
                    self._llm_env(db_path),
                )
        self.assertIsNone(calls["previous_analysis_at_run"][0])
        self.assertIsNotNone(calls["previous_analysis_at_run"][1])
        self.assertEqual([{"fact": "first"}], calls["previous_analysis_at_run"][1]["extracted_facts"])
        self.assertEqual([{"candidate": "first"}], calls["previous_analysis_at_run"][1]["memory_candidates"])

    def test_llm_candidate_persistence_does_not_write_memory_items_or_staging_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_no_memory_writes.sqlite3")
            with self._patched_llm_path(
                {
                    "answer": "Assistant answer.",
                    "stage1_decision": {
                        "decision_type": "answer_directly",
                        "selected_memory_ids": [],
                        "draft_answer": "Assistant answer.",
                        "extracted_facts": [{"fact": "observed"}],
                        "memory_candidates": [
                            {
                                "candidate_type": "fact",
                                "content": {"candidate": "audit only"},
                                "recommended_action": "stage",
                                "confidence": 0.6,
                            }
                        ],
                        "rationale": "audit only",
                    },
                }
            ):
                self._run_cli_with_env("User message.", self._llm_env(db_path))
            self.assertEqual(1, self._count_rows(db_path, "memory_candidates"))
            self.assertEqual(0, self._count_rows(db_path, "memory_items"))
            self.assertEqual(0, self._count_rows(db_path, "memory_staging"))

    def test_five_turn_memory_candidate_regression_is_deterministic(self) -> None:
        scenarios = {
            "ты знаешь алену?": {
                "answer": "I do not know Alena yet. Who is she to you?",
                "stage1_candidates": [],
                "stage1_facts": [],
                "stage1_reason": "No durable update yet; this is a request for context.",
            },
            "я тебе про нее не рассказывал, вспоминай": {
                "answer": "You are referring to the previously mentioned person, but I still need the context of who she is.",
                "stage1_candidates": [],
                "stage1_facts": [],
                "stage1_reason": "No durable update yet; this is a request to recall context.",
            },
            "а что если я тебе скажу что ее рабочее имя Екатерина?": {
                "answer": "Got it. You are saying her working name is Ekaterina. Is that the same person you mean?",
                "stage1_candidates": [],
                "stage1_facts": [],
                "stage1_reason": "No durable update; this is a clarification about the previously mentioned person.",
            },
            "она моя любовница, но когда была проституткой ее рабочее имя было Екатерина": {
                "answer": "Understood. You are saying the previously mentioned person is your lover, and that her working name was Ekaterina when she did sex work.",
                "stage1_candidates": [
                    {
                        "candidate_type": "relation",
                        "content": {
                            "subject": "user",
                            "relation": "lover",
                            "object": "previously mentioned person",
                            "subject_ref": "same_person",
                        },
                        "recommended_action": "stage",
                        "confidence": 0.8,
                    },
                    {
                        "candidate_type": "fact",
                        "content": {
                            "text": "The previously mentioned person did sex work.",
                            "subject_ref": "same_person",
                            "claim_status": "user_reported",
                            "sensitivity": "high",
                            "context_type": "biographical_context",
                        },
                        "recommended_action": "stage",
                        "confidence": 0.6,
                    },
                    {
                        "candidate_type": "name_alias",
                        "content": {
                            "raw_name": "Екатерина",
                            "subject_ref": "same_person",
                        },
                        "recommended_action": "stage",
                        "confidence": 0.8,
                    },
                ],
                "stage1_facts": [
                    {"fact": "The previously mentioned person is the user's lover."},
                    {"fact": "The previously mentioned person did sex work."},
                    {"fact": "The previously mentioned person used the working name Ekaterina."},
                ],
                "stage1_reason": "Extracted relationship, sensitive biographical context, and alias information for the same person.",
            },
            "я тебе уже ответил, посмотри": {
                "answer": "Right, you already answered. I will use the previous context you gave me.",
                "stage1_candidates": [],
                "stage1_facts": [],
                "stage1_reason": "No new durable update; this is a reminder to use prior context.",
            },
        }

        class RegressionTransport:
            """Deterministic transport that repairs Stage 0 and serves scenario-specific Stage 1 answers."""

            def __init__(self) -> None:
                self.requests: list[dict] = []
                self.stage1_contexts: list[dict] = []

            def post_json(
                self,
                *,
                url: str,
                headers: dict[str, str],
                payload: dict[str, object],
                timeout_seconds: float,
            ) -> dict[str, object]:
                del url, headers, timeout_seconds
                self.requests.append(payload)
                system_prompt = str(payload["messages"][0]["content"])
                user_context = json.loads(str(payload["messages"][1]["content"]))
                current_user_message = user_context["current_user_message"]
                if "Stage 1" in system_prompt:
                    stage0_frame = user_context["stage0_nlu_frame"]
                    if stage0_frame["dialogue_acts"] != ["other"]:
                        raise AssertionError(f"Stage 0 dialogue acts were not repaired: {stage0_frame['dialogue_acts']}")
                    self.stage1_contexts.append(user_context)
                    scenario = scenarios[current_user_message]
                    memory_candidates = scenario["stage1_candidates"]
                    has_candidates = bool(memory_candidates)
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "schema_version": "0.4.3",
                                            "decision_type": "answer_directly",
                                            "selected_memory_ids": [],
                                            "draft_answer": scenario["answer"],
                                            "extracted_facts": scenario["stage1_facts"],
                                            "memory_candidates": memory_candidates,
                                            "memory_update_extraction": {
                                                "status": "ok" if has_candidates else "fail",
                                                "reason": scenario["stage1_reason"],
                                            },
                                            "rationale": "Deterministic regression scenario.",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }
                if "Stage 0" in system_prompt:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "schema_version": "stage0_nlu_frame.v1",
                                            "normalized_intent": f"Normalized: {current_user_message}",
                                            "dialogue_acts": ["statement"],
                                            "entities": [],
                                            "current_signal": {
                                                "status": "clear",
                                                "kind": "other",
                                                "summary": "Context is present in the current message.",
                                                "needs_confirmation": False,
                                            },
                                            "clarification": {
                                                "needed": False,
                                                "question": "",
                                            },
                                            "memory_selection_hint": {
                                                "needed": False,
                                                "reason": "",
                                                "query_terms": [],
                                            },
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }
                raise AssertionError("Unexpected prompt stage")

        transport = RegressionTransport()
        provider = OpenAICompatibleLLMProvider(
            base_url="https://llm.example.test/v1",
            api_key="test_api_key",
            model="test_model",
            transport=transport,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli_five_turn_regression.sqlite3")
            with self._working_directory(temp_dir):
                with patch(
                    "mnemosyne_brain.app.cli.OpenAICompatibleLLMProvider.from_env",
                    return_value=provider,
                ):
                    rendered_turns: list[str] = []
                    for message in list(scenarios):
                        exit_code, rendered = self._run_cli_args_with_env(
                            ["--thread-id", "five-turn-regression", message],
                            self._llm_env(db_path),
                        )
                        rendered_turns.append(rendered)
                        self.assertEqual(0, exit_code, rendered)

            turns = self._list_turns(db_path)
            analysis_events = self._list_track_analysis_events(db_path)
            candidates = self._list_memory_candidates(db_path)
            memory_item_count = self._count_rows(db_path, "memory_items")
            memory_staging_count = self._count_rows(db_path, "memory_staging")

        self.assertEqual(10, len(turns))
        self.assertEqual(5, len(analysis_events))
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual(0, memory_item_count)
        self.assertEqual(0, memory_staging_count)
        self.assertTrue(any(candidate["candidate_type"] == "relation" for candidate in candidates))
        self.assertTrue(any(candidate["candidate_type"] == "name_alias" for candidate in candidates))
        self.assertTrue(
            any(
                candidate["candidate_type"] == "fact"
                and candidate["content"].get("context_type") == "biographical_context"
                and candidate["content"].get("subject_ref") == "same_person"
                for candidate in candidates
            )
        )
        turn4_candidates = [
            candidate for candidate in candidates if candidate["turn_id"] == turns[6]["turn_id"]
        ]
        self.assertEqual(3, len(turn4_candidates))
        self.assertTrue(
            all(candidate["content"].get("subject_ref") == "same_person" for candidate in turn4_candidates)
        )
        self.assertIn("working name is Ekaterina", rendered_turns[2])
        self.assertNotIn("who is she?", rendered_turns[2].lower())
        self.assertIn("your lover", rendered_turns[3])
        self.assertIn("working name was Ekaterina", rendered_turns[3])
        self.assertNotIn("who is she?", rendered_turns[3].lower())
        self.assertIn("already answered", rendered_turns[4].lower())
        for rendered in rendered_turns:
            self.assertNotIn("saved", rendered.lower())
            self.assertNotIn("stored", rendered.lower())
            self.assertNotIn("remembered", rendered.lower())
            self.assertNotIn("permanently applied", rendered.lower())

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

    def _track_from_output(self, rendered: str) -> str:
        for line in rendered.splitlines():
            if line.startswith("Track: "):
                return line.removeprefix("Track: ")
        raise AssertionError(f"No Track line found in output: {rendered}")

    def _list_turns(self, db_path: str) -> list[sqlite3.Row]:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(
                """
                SELECT turn_id, input_source, role, content_text, track_id
                FROM dialogue_turns
                ORDER BY created_at, turn_id
                """
            ).fetchall()
        finally:
            connection.close()

    def _list_track_analysis_events(self, db_path: str) -> list[dict]:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT event_type, actor_type, target_type, target_id, track_id, payload_json
                FROM audit_events
                WHERE event_type = 'track_analysis_saved'
                ORDER BY created_at, audit_event_id
                """
            ).fetchall()
            return [
                {
                    "event_type": row["event_type"],
                    "actor_type": row["actor_type"],
                    "target_type": row["target_type"],
                    "target_id": row["target_id"],
                    "track_id": row["track_id"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]
        finally:
            connection.close()

    def _list_memory_candidates(self, db_path: str) -> list[dict]:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT candidate_type, recommended_action, confidence, content_json, provenance_json, track_id, turn_id
                FROM memory_candidates
                ORDER BY rowid
                """
            ).fetchall()
            return [
                {
                    "candidate_type": row["candidate_type"],
                    "recommended_action": row["recommended_action"],
                    "confidence": row["confidence"],
                    "content": json.loads(row["content_json"]),
                    "provenance": json.loads(row["provenance_json"]),
                    "track_id": row["track_id"],
                    "turn_id": row["turn_id"],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def _count_rows(self, db_path: str, table_name: str) -> int:
        connection = sqlite3.connect(db_path)
        try:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        finally:
            connection.close()

    def _default_llm_result(self, answer: str = "ok") -> dict:
        return {
            "route": "answer_directly",
            "answer": answer,
            "selected_memory_ids": [],
            "used_memory_ids": [],
            "stage1_decision": {
                "decision_type": "answer_directly",
                "selected_memory_ids": [],
                "draft_answer": answer,
                "extracted_facts": [],
                "memory_candidates": [],
                "memory_update_extraction": {
                    "status": "fail",
                    "reason": "No durable information extracted.",
                },
                "rationale": None,
            },
            "stage2_decision": None,
        }

    def _memory_update_extraction_for(self, memory_candidates: list | None) -> dict:
        candidates = memory_candidates or []
        return {
            "status": "ok" if candidates else "fail",
            "reason": "Durable information extracted." if candidates else "No durable information extracted.",
        }

    def _with_memory_update_extraction(self, decision: dict | None) -> dict | None:
        if decision is None:
            return None
        return {
            **decision,
            "memory_update_extraction": decision.get(
                "memory_update_extraction",
                self._memory_update_extraction_for(decision.get("memory_candidates")),
            ),
        }

    def _llm_result(self, overrides: dict | None) -> dict:
        result = self._default_llm_result()
        if overrides:
            result.update(overrides)
            if "answer" in overrides and "stage1_decision" not in overrides:
                result["stage1_decision"] = {
                    **result["stage1_decision"],
                    "draft_answer": overrides["answer"],
                }
        result["stage1_decision"] = self._with_memory_update_extraction(result["stage1_decision"])
        result["stage2_decision"] = self._with_memory_update_extraction(result["stage2_decision"])
        return result

    @contextmanager
    def _patched_llm_path(
        self,
        result: dict | None = None,
        *,
        error: Exception | None = None,
    ) -> Iterator[dict]:
        calls = {
            "provider_from_env": 0,
            "orchestrator_runs": 0,
            "messages": [],
            "turn_counts_at_run": [],
            "in_transaction_at_run": [],
            "recent_texts_at_run": [],
            "previous_analysis_at_run": [],
        }

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

            def run_turn(
                self,
                track_id: str,
                current_user_message: str,
                *,
                exclude_turn_id: str | None = None,
            ) -> dict:
                calls["orchestrator_runs"] += 1
                calls["messages"].append(current_user_message)
                calls["turn_counts_at_run"].append(self._repository.count_rows("dialogue_turns"))
                calls["in_transaction_at_run"].append(self._repository.connection.in_transaction)
                recent_turns = self._repository.list_recent_turns_for_active_track(
                    track_id,
                    limit=12,
                    exclude_turn_id=exclude_turn_id,
                )
                calls["recent_texts_at_run"].append([turn.content_text for turn in recent_turns])
                calls["previous_analysis_at_run"].append(self._repository.get_latest_track_analysis(track_id))
                if error is not None:
                    raise error
                return self_outer._llm_result(result)

        self_outer = self
        with patch.multiple(
            "mnemosyne_brain.app.cli",
            OpenAICompatibleLLMProvider=FakeProviderFactory,
            DeterministicLLMOrchestrator=FakeOrchestrator,
        ):
            yield calls
