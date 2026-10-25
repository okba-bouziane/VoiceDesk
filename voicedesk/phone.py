"""Phone number normalisation shared by the conversation schemas and the booking service."""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[\s\-().]")
_VALID = re.compile(r"^\+?\d{8,15}$")


def normalise_phone(value: str) -> str:
    """Strip spaces, dashes, dots and parentheses."""
    return _SEPARATORS.sub("", value or "")


def is_valid_phone(normalised: str) -> bool:
    """8 to 15 digits, optionally starting with +."""
    return _VALID.match(normalised) is not None
