"""Contract tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mnemosyne_brain.app.contracts.base import SCHEMA_VERSION
from mnemosyne_brain.app.contracts.identity import IdentifierAssignment
from mnemosyne_brain.app.contracts.provenance import Provenance


class ContractsTestCase(unittest.TestCase):
    """Verifies required Pydantic validation rules."""

    def _assignment_payload(self) -> dict:
        return {
            "assignment_id": "assign_1",
            "identifier_key": "phone:123",
            "person_id": "person_1",
            "persona_id": None,
            "resolution_status": "resolved",
            "candidate_person_ids": [],
            "status": "active",
            "confidence": 0.9,
            "provenance_json": Provenance(source="test"),
            "created_at": "2026-07-05T12:00:00Z",
            "updated_at": "2026-07-05T12:00:00Z",
        }

    def test_persisted_models_use_schema_version(self) -> None:
        assignment = IdentifierAssignment(**self._assignment_payload())
        self.assertEqual(SCHEMA_VERSION, assignment.schema_version)

    def test_resolved_assignment_without_person_fails_validation(self) -> None:
        payload = self._assignment_payload()
        payload["person_id"] = None
        with self.assertRaises(ValidationError):
            IdentifierAssignment(**payload)

    def test_ambiguous_assignment_without_candidates_fails_validation(self) -> None:
        payload = self._assignment_payload()
        payload["person_id"] = None
        payload["resolution_status"] = "ambiguous"
        payload["candidate_person_ids"] = []
        with self.assertRaises(ValidationError):
            IdentifierAssignment(**payload)
