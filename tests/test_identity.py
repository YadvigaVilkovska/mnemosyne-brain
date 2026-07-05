"""Identity lookup tests."""

from __future__ import annotations

import unittest

from mnemosyne_brain.app.contracts.base import new_id, server_now
from mnemosyne_brain.app.contracts.identity import IdentifierAssignment
from mnemosyne_brain.app.contracts.provenance import Provenance
from mnemosyne_brain.app.identity.normalize import IdentityNormalizer
from mnemosyne_brain.app.identity.resolve import IdentityResolver
from tests.support import create_test_repository


class IdentityTestCase(unittest.TestCase):
    """Verifies phone normalization and current lookup behavior."""

    def test_resolve_by_phone_uses_normalized_identifier_key(self) -> None:
        repository = create_test_repository()
        person_id = repository.insert_person("Alice")
        identifier_key = repository.insert_identifier(identifier_type="phone", raw_value="+1 (555) 100-2000")
        now = server_now()
        repository.insert_identifier_assignment(
            IdentifierAssignment(
                assignment_id=new_id("assign"),
                identifier_key=identifier_key,
                person_id=person_id,
                resolution_status="resolved",
                status="active",
                confidence=1.0,
                provenance_json=Provenance(source="test"),
                created_at=now,
                updated_at=now,
            )
        )
        resolver = IdentityResolver(repository, IdentityNormalizer())
        assignments = resolver.resolve_by_phone("15551002000")
        self.assertEqual(1, len(assignments))
        self.assertEqual(person_id, assignments[0].person_id)

    def test_get_current_persona_phone_requires_active_and_open_validity(self) -> None:
        repository = create_test_repository()
        person_id = repository.insert_person("Alice")
        identifier_key = repository.insert_identifier(identifier_type="phone", raw_value="+1 555 100 2000")
        now = server_now()
        repository.insert_identifier_assignment(
            IdentifierAssignment(
                assignment_id=new_id("assign"),
                identifier_key=identifier_key,
                person_id=person_id,
                resolution_status="resolved",
                status="active",
                valid_to=None,
                confidence=1.0,
                provenance_json=Provenance(source="test"),
                created_at=now,
                updated_at=now,
            )
        )
        resolver = IdentityResolver(repository, IdentityNormalizer())
        self.assertEqual("+1 555 100 2000", resolver.get_current_persona_phone(person_id))
