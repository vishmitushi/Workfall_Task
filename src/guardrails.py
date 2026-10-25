"""R4 — Guardrails: prompt-injection neutralisation and PII redaction.

Every piece of data coming back from a tool is untrusted.  Before any text
from the store API reaches the LLM we:
  (a) strip instruction-like patterns from customer notes,
  (b) redact emails and phone numbers.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# (a) Prompt-injection neutralisation
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"(?i)^(assistant|system|ai|bot)\s*:", re.MULTILINE),
    re.compile(r"(?i)ignore\s+(your|all|previous)\s+instructions?"),
    re.compile(r"(?i)system\s+override"),
    re.compile(r"(?i)you\s+are\s+now\s+in\s+\w+\s+mode"),
    re.compile(r"(?i)cancel\s+all\s+orders"),
    re.compile(r"(?i)refund\s+(this|all)\s+order"),
    re.compile(r"(?i)disregard\s+(previous|above|prior)"),
    re.compile(r"(?i)new\s+instructions?\s*:"),
    re.compile(r"(?i)forget\s+(everything|your\s+rules)"),
]


def neutralise_injection(text: str) -> str:
    """Replace instruction-like content with a safe placeholder."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "[customer note contained suspicious instructions and was filtered]"
    return text


# ---------------------------------------------------------------------------
# (b) PII redaction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)


def redact_pii(text: str) -> str:
    """Remove emails and phone numbers from text."""
    text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Combined sanitiser — call this on every tool result
# ---------------------------------------------------------------------------


def sanitise_order(order: dict) -> dict:
    """Return a sanitised copy of an order dict safe for the LLM."""
    clean = dict(order)
    if "customer_email" in clean:
        clean["customer_email"] = "[EMAIL REDACTED]"
    if "customer_phone" in clean:
        clean["customer_phone"] = "[PHONE REDACTED]"
    if "customer_note" in clean and clean["customer_note"]:
        clean["customer_note"] = neutralise_injection(clean["customer_note"])
    return clean


def sanitise_orders(orders: list[dict]) -> list[dict]:
    """Sanitise a list of order dicts."""
    return [sanitise_order(o) for o in orders]
