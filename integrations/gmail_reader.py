"""
MYCONEX Gmail Reader
---------------------
IMAP client for Gmail using an App Password.

Setup:
  1. Enable 2FA on the Gmail account.
  2. Go to: https://myaccount.google.com/apppasswords
  3. Create an app password (name it "myconex" or "buzlock").
  4. Add to .env:
       GMAIL_ADDRESS=you@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

Usage:
    from integrations.gmail_reader import GmailReader
    reader = GmailReader()
    emails = reader.search(query="from:newsletter", limit=5)
"""

from __future__ import annotations

import email
import imaplib
import logging
import os
import re
import textwrap
from datetime import datetime, timezone
from email.header import decode_header
from typing import Any

logger = logging.getLogger(__name__)

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993


def _decode_mime_words(value: str) -> str:
    """Decode RFC 2047 encoded header words."""
    parts = []
    for raw, charset in decode_header(value):
        if isinstance(raw, bytes):
            parts.append(raw.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(raw)
    return "".join(parts)


def _extract_text(msg: email.message.Message) -> str:
    """Pull plain-text body from a MIME message, falling back to HTML→text."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            charset = part.get_content_charset() or "utf-8"
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            text = payload.decode(charset, errors="replace")
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts)

    # Minimal HTML→text: strip tags, collapse whitespace
    if html_parts:
        raw = "\n".join(html_parts)
        raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<p[^>]*>", "\n", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", "", raw)
        raw = re.sub(r"&nbsp;", " ", raw)
        raw = re.sub(r"&amp;", "&", raw)
        raw = re.sub(r"&lt;", "<", raw)
        raw = re.sub(r"&gt;", ">", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()

    return "(no body)"


def _format_email(uid: str, msg: email.message.Message, body_limit: int = 2000) -> dict[str, Any]:
    subject = _decode_mime_words(msg.get("Subject", "(no subject)"))
    sender  = _decode_mime_words(msg.get("From", ""))
    to      = _decode_mime_words(msg.get("To", ""))
    date    = msg.get("Date", "")
    body    = _extract_text(msg)
    if len(body) > body_limit:
        body = body[:body_limit] + f"\n... [{len(body) - body_limit} chars truncated]"
    return {
        "uid": uid,
        "subject": subject,
        "from": sender,
        "to": to,
        "date": date,
        "body": body,
    }


class GmailReader:
    """Thin wrapper around imaplib for Gmail."""

    def __init__(
        self,
        address: str | None = None,
        app_password: str | None = None,
    ) -> None:
        self.address = address or os.getenv("GMAIL_ADDRESS", "")
        self.app_password = app_password or os.getenv("GMAIL_APP_PASSWORD", "")
        if not self.address or not self.app_password:
            raise ValueError(
                "Gmail credentials missing. Set GMAIL_ADDRESS and "
                "GMAIL_APP_PASSWORD in .env"
            )
        # Normalise: remove spaces from app password
        self.app_password = self.app_password.replace(" ", "")

    def _connect(self) -> imaplib.IMAP4_SSL:
        conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        conn.login(self.address, self.app_password)
        return conn

    # ── Public API ────────────────────────────────────────────────────────────

    def list_folders(self) -> list[str]:
        """Return available mailbox folder names."""
        conn = self._connect()
        try:
            _, data = conn.list()
            folders = []
            for item in data:
                if isinstance(item, bytes):
                    # e.g. b'(\\HasNoChildren) "/" "INBOX"'
                    parts = item.decode().split('"')
                    name = parts[-2] if len(parts) >= 2 else item.decode()
                    folders.append(name)
            return folders
        finally:
            conn.logout()

    def search(
        self,
        query: str = "ALL",
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
        body_limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """
        Search emails and return a list of dicts.

        query     — IMAP search string or friendly shorthand:
                    "unread"            → UNSEEN
                    "from:foo@bar.com"  → FROM "foo@bar.com"
                    "subject:invoice"   → SUBJECT "invoice"
                    "ALL"               → all messages
        folder    — mailbox to search (default: INBOX)
        limit     — max number of emails to return (newest first)
        unread_only — if True, adds UNSEEN to the search criteria
        """
        criteria = self._build_criteria(query, unread_only)
        conn = self._connect()
        try:
            conn.select(f'"{folder}"', readonly=True)
            _, data = conn.search(None, criteria)
            uids = data[0].split() if data[0] else []
            # Newest first, cap at limit
            uids = uids[-limit:][::-1]
            if not uids:
                return []
            results = []
            for uid in uids:
                try:
                    _, msg_data = conn.fetch(uid, "(RFC822)")
                    if msg_data and msg_data[0]:
                        raw = msg_data[0][1]
                        if isinstance(raw, bytes):
                            msg = email.message_from_bytes(raw)
                            results.append(_format_email(uid.decode(), msg, body_limit))
                except Exception as exc:
                    logger.warning("Failed to fetch email uid=%s: %s", uid, exc)
            return results
        finally:
            conn.logout()

    def read(self, uid: str, folder: str = "INBOX", body_limit: int = 4000) -> dict[str, Any] | None:
        """Fetch a single email by UID."""
        conn = self._connect()
        try:
            conn.select(f'"{folder}"', readonly=True)
            _, msg_data = conn.fetch(uid.encode(), "(RFC822)")
            if not msg_data or not msg_data[0]:
                return None
            raw = msg_data[0][1]
            if not isinstance(raw, bytes):
                return None
            msg = email.message_from_bytes(raw)
            return _format_email(uid, msg, body_limit)
        finally:
            conn.logout()

    def get_unread_count(self, folder: str = "INBOX") -> int:
        """Return the number of unread messages in a folder."""
        conn = self._connect()
        try:
            conn.select(f'"{folder}"', readonly=True)
            _, data = conn.search(None, "UNSEEN")
            return len(data[0].split()) if data[0] else 0
        finally:
            conn.logout()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_criteria(query: str, unread_only: bool) -> str:
        q = query.strip()
        # Friendly shorthands
        if q.lower() == "unread":
            criteria = "UNSEEN"
        elif q.lower().startswith("from:"):
            addr = q[5:].strip()
            criteria = f'FROM "{addr}"'
        elif q.lower().startswith("subject:"):
            subj = q[8:].strip()
            criteria = f'SUBJECT "{subj}"'
        elif q.lower().startswith("body:"):
            text = q[5:].strip()
            criteria = f'BODY "{text}"'
        elif q.lower().startswith("to:"):
            addr = q[3:].strip()
            criteria = f'TO "{addr}"'
        else:
            criteria = q if q else "ALL"

        if unread_only and "UNSEEN" not in criteria:
            criteria = f"UNSEEN {criteria}"
        return criteria

    @staticmethod
    def format_for_llm(emails: list[dict[str, Any]]) -> str:
        """Render a list of email dicts as readable text for an LLM."""
        if not emails:
            return "No emails found."
        lines = [f"Found {len(emails)} email(s):\n"]
        for i, e in enumerate(emails, 1):
            lines.append(f"{'─' * 60}")
            lines.append(f"[{i}] UID: {e['uid']}")
            lines.append(f"    From:    {e['from']}")
            lines.append(f"    Subject: {e['subject']}")
            lines.append(f"    Date:    {e['date']}")
            body = textwrap.indent(e["body"].strip(), "    ")
            lines.append(f"\n{body}\n")
        return "\n".join(lines)
