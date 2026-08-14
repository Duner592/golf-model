"""Safe error handling for HTTP requests that carry API credentials."""

from __future__ import annotations

import re

import requests


_SENSITIVE_QUERY_VALUE = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|token|access[_-]?token|authorization)=)([^&#\s]+)"
)
_SENSITIVE_JSON_VALUE = re.compile(
    r'(?i)(["\'](?:key|api[_-]?key|token|access[_-]?token|authorization)["\']\s*:\s*["\'])([^"\']+)(["\'])'
)


def redact_sensitive_text(value: object) -> str:
    """Return text with common API credential values removed."""
    text = str(value)
    text = _SENSITIVE_QUERY_VALUE.sub(r"\1[REDACTED]", text)
    return _SENSITIVE_JSON_VALUE.sub(r"\1[REDACTED]\3", text)


def raise_for_status_safely(response) -> None:
    """Raise a redacted error rather than exposing credential-bearing URLs."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(redact_sensitive_text(exc), response=exc.response) from None
