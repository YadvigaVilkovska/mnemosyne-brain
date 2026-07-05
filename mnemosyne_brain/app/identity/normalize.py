"""Identity normalization services."""

from __future__ import annotations

from .validators import normalize_phone


class IdentityNormalizer:
    """Normalizes identity values by identifier type."""

    def normalize(self, identifier_type: str, raw_value: str) -> str:
        """Return the canonical identifier value."""

        if identifier_type == "phone":
            return normalize_phone(raw_value)
        normalized = raw_value.strip().lower()
        if not normalized:
            raise ValueError("identifier value must not be empty")
        return normalized
