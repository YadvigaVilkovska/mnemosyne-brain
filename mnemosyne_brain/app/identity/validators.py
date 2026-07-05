"""Identity validation helpers."""

from __future__ import annotations


def normalize_phone(raw_phone: str) -> str:
    """Normalize a phone number to digits for reverse lookup."""

    digits = "".join(character for character in raw_phone if character.isdigit())
    if not digits:
        raise ValueError("phone must contain digits")
    return digits
