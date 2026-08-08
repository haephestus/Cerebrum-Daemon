"""
api.auth.email_inator
=====================
Minimal email-sending seam. Selected by config:

  * no EMAIL_API_KEY   → DevLogEmailSender: logs the message (incl. the reset
    code) so the whole flow is usable/testable locally with no provider.
  * EMAIL_API_KEY set  → ResendEmailSender: POSTs to the Resend HTTP API via
    httpx (already a dependency). Swap the provider by editing one class.

Env:
  EMAIL_API_KEY   provider key (absent → dev-log mode)
  EMAIL_FROM      From address (default a placeholder)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> bool: ...


class DevLogEmailSender:
    """Logs the email instead of sending — for local/dev with no provider."""

    def send(self, to: str, subject: str, body: str) -> bool:
        logger.warning(
            "[DEV EMAIL — not actually sent]\n  to: %s\n  subject: %s\n  %s",
            to,
            subject,
            body.replace("\n", "\n  "),
        )
        return True


class ResendEmailSender:
    def __init__(self, api_key: str, from_addr: str):
        self._key = api_key
        self._from = from_addr

    def send(self, to: str, subject: str, body: str) -> bool:
        try:
            import httpx

            r = httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"from": self._from, "to": [to], "subject": subject, "text": body},
                timeout=15.0,
            )
            r.raise_for_status()
            return True
        except Exception as e:
            logger.error("email send to %s failed: %s", to, e)
            return False


def get_email_sender() -> EmailSender:
    key = os.getenv("EMAIL_API_KEY")
    from_addr = os.getenv("EMAIL_FROM", "Cerebrum <noreply@cerebrum.local>")
    if key:
        return ResendEmailSender(key, from_addr)
    return DevLogEmailSender()
