"""SQLite repository for all durable business truth."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..contracts.analysis import ConflictDecision
from ..contracts.base import SCHEMA_VERSION, new_id, server_now
from ..contracts.executor import ExecutorEvent, ExecutorTaskCapsule
from ..contracts.identity import IdentifierAssignment
from ..contracts.memory import DialogueTrack, DialogueTurn, MemoryCandidate, TrackStatus
from ..contracts.provenance import Provenance
from .transactions import SqliteUnitOfWork


class RepositoryError(RuntimeError):
    """Base repository exception for explicit application errors."""


class SqliteRepository:
    """Narrow SQLite adapter used by services and graph nodes."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.isolation_level = None
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def connect(cls, db_path: str | Path) -> "SqliteRepository":
        """Create a repository connected to a SQLite file path."""

        return cls(sqlite3.connect(str(db_path)))

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the connection for migrations and test assertions."""

        return self._connection

    def transaction(self) -> SqliteUnitOfWork:
        """Create an explicit transaction boundary."""

        return SqliteUnitOfWork(self._connection)

    def to_json(self, value: Any) -> str:
        """Serialize JSON columns once using Pydantic JSON mode when available."""

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def from_json(self, value: str | None) -> Any:
        """Deserialize a JSON column."""

        return None if value is None else json.loads(value)

    def stable_key(self, *parts: Any) -> str:
        """Build a deterministic key from JSON-serializable parts."""

        raw = self.to_json(list(parts))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def bootstrap_or_load_track(
        self,
        *,
        dialogue_id: str,
        thread_id: str,
        owner_user_id: str,
    ) -> DialogueTrack:
        """Load a track for a thread or create one atomically."""

        row = self._connection.execute(
            "SELECT * FROM dialogue_tracks_temp WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is not None:
            return self._track_from_row(row)

        now = server_now()
        track_id = new_id("trk")
        track_json = {"dialogue_id": dialogue_id, "thread_id": thread_id}
        self._connection.execute(
            """
            INSERT INTO dialogue_tracks_temp(
              track_id, schema_version, dialogue_id, thread_id, owner_user_id, status,
              summary, track_json, created_at, updated_at, last_turn_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track_id,
                SCHEMA_VERSION,
                dialogue_id,
                thread_id,
                owner_user_id,
                TrackStatus.ACTIVE.value,
                None,
                self.to_json(track_json),
                now,
                now,
                None,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO dialogue_threads(dialogue_id, track_id, thread_id, status, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (dialogue_id, track_id, thread_id, TrackStatus.ACTIVE.value, now, now),
        )
        return self.get_track(track_id)

    def get_track(self, track_id: str) -> DialogueTrack:
        """Load a dialogue track by id."""

        row = self._connection.execute(
            "SELECT * FROM dialogue_tracks_temp WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown track_id: {track_id}")
        return self._track_from_row(row)

    def load_track_by_capsule(self, capsule_id: str) -> DialogueTrack:
        """Load a track using an executor capsule reference."""

        task = self.get_executor_task(capsule_id)
        return self.get_track(task.source_track_id)

    def update_track_status(
        self, track_id: str, status: TrackStatus, last_turn_id: str | None = None
    ) -> DialogueTrack:
        """Update durable track status."""

        now = server_now()
        self._connection.execute(
            """
            UPDATE dialogue_tracks_temp
            SET status = ?, updated_at = ?, last_turn_id = COALESCE(?, last_turn_id)
            WHERE track_id = ?
            """,
            (status.value, now, last_turn_id, track_id),
        )
        self._connection.execute(
            "UPDATE dialogue_threads SET status = ?, updated_at = ? WHERE track_id = ?",
            (status.value, now, track_id),
        )
        return self.get_track(track_id)

    def persist_dialogue_turn(
        self,
        *,
        dialogue_id: str,
        input_source: str,
        role: str,
        track_id: str | None = None,
        thread_id: str | None = None,
        external_message_id: str | None = None,
        content_text: str | None = None,
        content_json: Any = None,
    ) -> tuple[DialogueTurn, bool]:
        """Persist a dialogue turn idempotently by external_message_id."""

        if external_message_id is not None:
            existing = self._connection.execute(
                "SELECT * FROM dialogue_turns WHERE external_message_id = ?",
                (external_message_id,),
            ).fetchone()
            if existing is not None:
                return self._turn_from_row(existing), False

        now = server_now()
        turn_id = new_id("turn")
        self._connection.execute(
            """
            INSERT INTO dialogue_turns(
              turn_id, schema_version, dialogue_id, track_id, thread_id, input_source,
              role, external_message_id, content_text, content_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                SCHEMA_VERSION,
                dialogue_id,
                track_id,
                thread_id,
                input_source,
                role,
                external_message_id,
                content_text,
                self.to_json(content_json) if content_json is not None else None,
                now,
            ),
        )
        if track_id is not None:
            self.update_track_status(track_id, TrackStatus.ACTIVE, last_turn_id=turn_id)
        return self.get_turn(turn_id), True

    def get_turn(self, turn_id: str) -> DialogueTurn:
        """Load a dialogue turn."""

        row = self._connection.execute(
            "SELECT * FROM dialogue_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown turn_id: {turn_id}")
        return self._turn_from_row(row)

    def attach_turn_to_track(self, turn_id: str, track_id: str, thread_id: str) -> DialogueTurn:
        """Attach a previously persisted turn to a bootstrapped track."""

        self._connection.execute(
            "UPDATE dialogue_turns SET track_id = ?, thread_id = ? WHERE turn_id = ?",
            (track_id, thread_id, turn_id),
        )
        return self.get_turn(turn_id)

    def persist_memory_candidate(self, candidate: MemoryCandidate) -> tuple[MemoryCandidate, bool]:
        """Persist a memory candidate idempotently by idempotency_key."""

        existing = self._connection.execute(
            "SELECT * FROM memory_candidates WHERE idempotency_key = ?",
            (candidate.idempotency_key,),
        ).fetchone()
        if existing is not None:
            return self._candidate_from_row(existing), False
        self._connection.execute(
            """
            INSERT INTO memory_candidates(
              candidate_id, schema_version, dialogue_id, track_id, turn_id, candidate_type,
              recommended_action, confidence, dedupe_key, idempotency_key, content_json,
              provenance_json, merge_target_memory_id, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.candidate_id,
                candidate.schema_version,
                candidate.dialogue_id,
                candidate.track_id,
                candidate.turn_id,
                candidate.candidate_type,
                candidate.recommended_action,
                candidate.confidence,
                candidate.dedupe_key,
                candidate.idempotency_key,
                self.to_json(candidate.content_json),
                self.to_json(candidate.provenance_json),
                candidate.merge_target_memory_id,
                candidate.created_at,
                candidate.updated_at,
            ),
        )
        return candidate, True

    def get_memory_candidate(self, candidate_id: str) -> MemoryCandidate:
        """Load a memory candidate."""

        row = self._connection.execute(
            "SELECT * FROM memory_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return self._candidate_from_row(row)

    def find_active_memory_by_dedupe_key(self, dedupe_key: str) -> list[str]:
        """Return active memory ids with the same dedupe key."""

        rows = self._connection.execute(
            "SELECT memory_id FROM memory_items WHERE dedupe_key = ? AND status IN ('active', 'needs_confirmation')",
            (dedupe_key,),
        ).fetchall()
        return [row["memory_id"] for row in rows]

    def insert_memory_item(self, candidate: MemoryCandidate) -> str:
        """Insert a durable memory item from a candidate."""

        now = server_now()
        memory_id = new_id("mem")
        self._connection.execute(
            """
            INSERT INTO memory_items(
              memory_id, schema_version, memory_type, status, stability, content_json,
              intent_tags, entity_keys, provenance_json, dedupe_key, source_track_id,
              source_turn_id, valid_from, valid_to, observed_at, confidence, privacy_level,
              created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                SCHEMA_VERSION,
                candidate.candidate_type,
                "active",
                "stable",
                self.to_json(candidate.content_json),
                self.to_json([]),
                self.to_json([]),
                self.to_json(candidate.provenance_json),
                candidate.dedupe_key,
                candidate.track_id,
                candidate.turn_id,
                None,
                None,
                now,
                candidate.confidence,
                "normal",
                now,
                now,
            ),
        )
        return memory_id

    def insert_memory_staging(self, candidate: MemoryCandidate, decision: ConflictDecision) -> str:
        """Insert a staged memory review record."""

        now = server_now()
        staging_id = new_id("stage")
        self._connection.execute(
            """
            INSERT INTO memory_staging(
              staging_id, schema_version, candidate_id, candidate_type, status, recommended_action,
              confidence, dedupe_key, idempotency_key, merge_target_memory_id, conflict_memory_ids,
              content_json, provenance_json, review_reason, reviewed_by, reviewed_at, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                staging_id,
                SCHEMA_VERSION,
                candidate.candidate_id,
                candidate.candidate_type,
                "pending_review",
                candidate.recommended_action,
                candidate.confidence,
                candidate.dedupe_key,
                candidate.idempotency_key,
                candidate.merge_target_memory_id,
                self.to_json(decision.conflict_memory_ids),
                self.to_json(candidate.content_json),
                self.to_json(candidate.provenance_json),
                decision.reason,
                None,
                None,
                now,
                now,
            ),
        )
        return staging_id

    def insert_audit_event(
        self,
        *,
        event_type: str,
        actor_type: str,
        target_type: str,
        target_id: str,
        payload: Any,
        actor_id: str | None = None,
        dialogue_id: str | None = None,
        track_id: str | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Insert an audit event."""

        audit_event_id = new_id("audit")
        self._connection.execute(
            """
            INSERT INTO audit_events(
              audit_event_id, schema_version, event_type, actor_type, actor_id, dialogue_id,
              track_id, turn_id, target_type, target_id, payload_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_event_id,
                SCHEMA_VERSION,
                event_type,
                actor_type,
                actor_id,
                dialogue_id,
                track_id,
                turn_id,
                target_type,
                target_id,
                self.to_json(payload),
                server_now(),
            ),
        )
        return audit_event_id

    def insert_identifier_assignment(self, assignment: IdentifierAssignment) -> str:
        """Insert an identifier assignment after Pydantic validation."""

        self._connection.execute(
            """
            INSERT INTO identifier_assignments(
              assignment_id, schema_version, identifier_key, person_id, persona_id,
              resolution_status, candidate_person_ids, assignment_scope, status, valid_from,
              valid_to, confidence, provenance_json, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assignment.assignment_id,
                assignment.schema_version,
                assignment.identifier_key,
                assignment.person_id,
                assignment.persona_id,
                assignment.resolution_status,
                self.to_json(assignment.candidate_person_ids),
                assignment.assignment_scope,
                assignment.status,
                assignment.valid_from,
                assignment.valid_to,
                assignment.confidence,
                self.to_json(assignment.provenance_json),
                assignment.created_at,
                assignment.updated_at,
            ),
        )
        return assignment.assignment_id

    def update_identifier_assignment(self, assignment_id: str, *, status: str) -> None:
        """Update identifier assignment status."""

        self._connection.execute(
            "UPDATE identifier_assignments SET status = ?, updated_at = ? WHERE assignment_id = ?",
            (status, server_now(), assignment_id),
        )

    def insert_executor_task(self, capsule: ExecutorTaskCapsule) -> tuple[ExecutorTaskCapsule, bool]:
        """Insert an executor task idempotently by idempotency key."""

        existing = self._connection.execute(
            "SELECT * FROM executor_tasks WHERE idempotency_key = ?",
            (capsule.idempotency_key,),
        ).fetchone()
        if existing is not None:
            return self._task_from_row(existing), False
        self._connection.execute(
            """
            INSERT INTO executor_tasks(
              capsule_id, schema_version, source_track_id, thread_id, executor, status,
              idempotency_key, attempt_count, locked_by, locked_until, capsule_json,
              result_json, last_error_json, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capsule.capsule_id,
                capsule.schema_version,
                capsule.source_track_id,
                capsule.thread_id,
                capsule.executor,
                capsule.status,
                capsule.idempotency_key,
                capsule.attempt_count,
                None,
                None,
                self.to_json(capsule.capsule_json),
                self.to_json(capsule.result_json) if capsule.result_json is not None else None,
                self.to_json(capsule.last_error_json) if capsule.last_error_json is not None else None,
                capsule.created_at,
                capsule.updated_at,
            ),
        )
        return capsule, True

    def get_executor_task(self, capsule_id: str) -> ExecutorTaskCapsule:
        """Load an executor task."""

        row = self._connection.execute(
            "SELECT * FROM executor_tasks WHERE capsule_id = ?",
            (capsule_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown capsule_id: {capsule_id}")
        return self._task_from_row(row)

    def update_executor_task_status(self, capsule_id: str, status: str, *, final: bool = False) -> None:
        """Update executor task status and attempt count."""

        task = self.get_executor_task(capsule_id)
        attempt_count = task.attempt_count + 1 if final else task.attempt_count
        self._connection.execute(
            "UPDATE executor_tasks SET status = ?, attempt_count = ?, updated_at = ? WHERE capsule_id = ?",
            (status, attempt_count, server_now(), capsule_id),
        )

    def get_executor_event(self, event_id: str) -> ExecutorEvent:
        """Load a persisted executor event."""

        row = self._connection.execute(
            "SELECT * FROM executor_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._event_from_row(row)

    def find_executor_event(self, event_id: str) -> ExecutorEvent | None:
        """Return an executor event if it already exists."""

        row = self._connection.execute(
            "SELECT * FROM executor_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return None if row is None else self._event_from_row(row)

    def insert_executor_event(self, event: ExecutorEvent) -> None:
        """Persist an executor event before graph invocation."""

        self._connection.execute(
            """
            INSERT INTO executor_events(
              event_id, schema_version, capsule_id, correlation_id, executor, status,
              attempt, is_final, applied, stale, applied_at, payload_json, error_json,
              artifacts_json, created_at, received_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.schema_version,
                event.capsule_id,
                event.correlation_id,
                event.executor,
                event.status,
                event.attempt,
                int(event.is_final),
                int(event.applied),
                int(event.stale),
                event.applied_at,
                self.to_json(event.payload),
                self.to_json(event.error) if event.error is not None else None,
                self.to_json(event.artifacts),
                event.created_at,
                event.received_at,
            ),
        )

    def mark_executor_event_applied(self, event_id: str) -> None:
        """Mark an executor event as applied after graph success."""

        self._connection.execute(
            "UPDATE executor_events SET applied = 1, applied_at = ? WHERE event_id = ?",
            (server_now(), event_id),
        )

    def resolve_by_phone(self, raw_phone: str) -> list[IdentifierAssignment]:
        """Resolve active identity assignments by normalized phone."""

        identifier_key = f"phone:{self._normalize_phone(raw_phone)}"
        rows = self._connection.execute(
            """
            SELECT * FROM identifier_assignments
            WHERE identifier_key = ? AND status = 'active'
            """,
            (identifier_key,),
        ).fetchall()
        return [self._assignment_from_row(row) for row in rows]

    def get_current_persona_phone(self, person_id: str, persona_id: str | None = None) -> str | None:
        """Return the current active phone identifier for a person/persona."""

        params: list[Any] = [person_id]
        persona_clause = "persona_id IS NULL"
        if persona_id is not None:
            persona_clause = "persona_id = ?"
            params.append(persona_id)
        rows = self._connection.execute(
            f"""
            SELECT identifiers.raw_value
            FROM identifier_assignments
            JOIN identifiers ON identifiers.identifier_key = identifier_assignments.identifier_key
            WHERE identifier_assignments.person_id = ?
              AND {persona_clause}
              AND identifier_assignments.status = 'active'
              AND identifier_assignments.valid_to IS NULL
              AND identifiers.identifier_type = 'phone'
            ORDER BY identifier_assignments.created_at DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return None if rows is None else rows["raw_value"]

    def insert_identifier(self, *, identifier_type: str, raw_value: str) -> str:
        """Insert an identifier using normalized reverse lookup key."""

        normalized = self._normalize_phone(raw_value) if identifier_type == "phone" else raw_value.strip().lower()
        identifier_key = f"{identifier_type}:{normalized}"
        now = server_now()
        identifier_id = new_id("ident")
        self._connection.execute(
            """
            INSERT OR IGNORE INTO identifiers(
              identifier_id, schema_version, identifier_type, raw_value, normalized_value,
              identifier_key, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (identifier_id, SCHEMA_VERSION, identifier_type, raw_value, normalized, identifier_key, now, now),
        )
        return identifier_key

    def insert_person(self, display_name: str) -> str:
        """Insert a person record."""

        person_id = new_id("person")
        now = server_now()
        self._connection.execute(
            """
            INSERT INTO persons(person_id, schema_version, display_name, status, merged_into, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (person_id, SCHEMA_VERSION, display_name, "active", None, now, now),
        )
        return person_id

    def count_rows(self, table_name: str) -> int:
        """Return a table row count for tests and demo summaries."""

        allowed_tables = {
            "dialogue_threads",
            "dialogue_turns",
            "dialogue_tracks_temp",
            "memory_candidates",
            "memory_staging",
            "memory_items",
            "persons",
            "personas",
            "identifiers",
            "identifier_assignments",
            "name_aliases",
            "executor_tasks",
            "executor_events",
            "audit_events",
        }
        if table_name not in allowed_tables:
            raise ValueError(f"Unsupported table for counting: {table_name}")
        return self._connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    def list_recent_turns_for_active_track(
        self,
        track_id: str,
        *,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[DialogueTurn]:
        """Return recent dialogue turns for an active working track only."""

        track = self.get_track(track_id)
        if track.status is TrackStatus.CLOSED:
            return []
        params: list[Any] = [track_id]
        exclude_clause = ""
        if exclude_turn_id is not None:
            exclude_clause = "AND turn_id != ?"
            params.append(exclude_turn_id)
        params.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT *
            FROM dialogue_turns
            WHERE track_id = ?
              {exclude_clause}
            ORDER BY created_at DESC, turn_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [self._turn_from_row(row) for row in reversed(rows)]

    def get_latest_track_analysis(self, track_id: str) -> dict[str, Any] | None:
        """Return the latest saved analysis payload for a track."""

        row = self._connection.execute(
            """
            SELECT payload_json
            FROM audit_events
            WHERE event_type = 'track_analysis_saved'
              AND target_type = 'dialogue_track'
              AND target_id = ?
            ORDER BY created_at DESC, audit_event_id DESC
            LIMIT 1
            """,
            (track_id,),
        ).fetchone()
        return None if row is None else self.from_json(row["payload_json"])

    def list_pinned_exact_messages(self, track_id: str, *, limit: int) -> list[dict[str, Any]]:
        """Return pinned exact messages or quotes for a track."""

        rows = self._connection.execute(
            """
            SELECT audit_event_id, payload_json, created_at
            FROM audit_events
            WHERE event_type = 'pinned_exact_message'
              AND target_type = 'dialogue_track'
              AND target_id = ?
            ORDER BY created_at DESC, audit_event_id DESC
            LIMIT ?
            """,
            (track_id, limit),
        ).fetchall()
        pins: list[dict[str, Any]] = []
        for row in reversed(rows):
            payload = self.from_json(row["payload_json"])
            pins.append(
                {
                    "pin_id": row["audit_event_id"],
                    "text": payload.get("text", ""),
                    "created_at": row["created_at"],
                }
            )
        return pins

    def list_memory_manifest_items(self, *, limit: int) -> list[dict[str, Any]]:
        """Return bounded memory metadata without full memory content."""

        rows = self._connection.execute(
            """
            SELECT memory_id, memory_type, content_json, confidence, created_at, updated_at
            FROM memory_items
            WHERE status IN ('active', 'needs_confirmation')
            ORDER BY updated_at DESC, memory_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._memory_manifest_from_row(row) for row in rows]

    def get_memory_context_items(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return full memory content for selected memory ids only."""

        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._connection.execute(
            f"""
            SELECT memory_id, memory_type, content_json, confidence, created_at, updated_at
            FROM memory_items
            WHERE memory_id IN ({placeholders})
              AND status IN ('active', 'needs_confirmation')
            """,
            tuple(memory_ids),
        ).fetchall()
        return {
            row["memory_id"]: {
                "memory_id": row["memory_id"],
                "kind": row["memory_type"],
                "content": self.from_json(row["content_json"]),
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def _normalize_phone(self, raw_phone: str) -> str:
        digits = "".join(character for character in raw_phone if character.isdigit())
        if not digits:
            raise ValueError("phone must contain digits")
        return digits

    def _track_from_row(self, row: sqlite3.Row) -> DialogueTrack:
        payload = dict(row)
        payload["track_json"] = self.from_json(payload["track_json"])
        payload["status"] = TrackStatus(payload["status"])
        return DialogueTrack(**payload)

    def _turn_from_row(self, row: sqlite3.Row) -> DialogueTurn:
        payload = dict(row)
        payload["content_json"] = self.from_json(payload["content_json"])
        return DialogueTurn(**payload)

    def _candidate_from_row(self, row: sqlite3.Row) -> MemoryCandidate:
        payload = dict(row)
        payload["content_json"] = self.from_json(payload["content_json"])
        payload["provenance_json"] = Provenance(**self.from_json(payload["provenance_json"]))
        return MemoryCandidate(**payload)

    def _task_from_row(self, row: sqlite3.Row) -> ExecutorTaskCapsule:
        payload = dict(row)
        payload["capsule_json"] = self.from_json(payload["capsule_json"])
        payload["result_json"] = self.from_json(payload["result_json"])
        payload["last_error_json"] = self.from_json(payload["last_error_json"])
        payload.pop("locked_by", None)
        payload.pop("locked_until", None)
        return ExecutorTaskCapsule(**payload)

    def _event_from_row(self, row: sqlite3.Row) -> ExecutorEvent:
        payload = dict(row)
        payload["is_final"] = bool(payload["is_final"])
        payload["applied"] = bool(payload["applied"])
        payload["stale"] = bool(payload["stale"])
        payload["payload"] = self.from_json(payload.pop("payload_json"))
        error_payload = self.from_json(payload.pop("error_json"))
        payload["error"] = error_payload
        payload["artifacts"] = self.from_json(payload.pop("artifacts_json"))
        return ExecutorEvent(**payload)

    def _assignment_from_row(self, row: sqlite3.Row) -> IdentifierAssignment:
        payload = dict(row)
        payload["candidate_person_ids"] = self.from_json(payload["candidate_person_ids"])
        payload["provenance_json"] = Provenance(**self.from_json(payload["provenance_json"]))
        return IdentifierAssignment(**payload)

    def _memory_manifest_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        content = self.from_json(row["content_json"])
        text = ""
        if isinstance(content, dict):
            text = str(content.get("preview") or content.get("title") or content.get("text") or "")
        preview = text[:120]
        if len(text) > 120:
            preview = preview.rstrip() + "..."
        return {
            "memory_id": row["memory_id"],
            "kind": row["memory_type"],
            "entity_type": content.get("entity_type") if isinstance(content, dict) else None,
            "entity_label": content.get("entity_label") if isinstance(content, dict) else None,
            "title": content.get("title") if isinstance(content, dict) else None,
            "short_preview": preview,
            "importance": content.get("importance") if isinstance(content, dict) else None,
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
