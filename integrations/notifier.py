"""
MYCONEX Push Notifier
----------------------
Sends critical alerts to ntfy.sh and/or Pushover when the bot needs to reach
you outside of Discord/Telegram (crashes, disk full, high-severity signals).

Env vars:
  NTFY_URL          — ntfy.sh topic URL, e.g. https://ntfy.sh/my-secret-topic
  NTFY_TOKEN        — optional Bearer token for private ntfy servers
  PUSHOVER_TOKEN    — Pushover application token
  PUSHOVER_USER     — Pushover user/group key
  NOTIFY_MIN_LEVEL  — minimum level to send: "critical" | "warning" | "info" (default: warning)

Usage:
    from integrations.notifier import notify
    await notify("Bot crashed", level="critical", title="MYCONEX Alert")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
import urllib.parse
from typing import Literal

logger = logging.getLogger(__name__)

_NTFY_URL    = os.getenv("NTFY_URL", "")
_NTFY_TOKEN  = os.getenv("NTFY_TOKEN", "")
_PO_TOKEN    = os.getenv("PUSHOVER_TOKEN", "")
_PO_USER     = os.getenv("PUSHOVER_USER", "")
_MIN_LEVEL   = os.getenv("NOTIFY_MIN_LEVEL", "warning")

_LEVELS = {"info": 0, "warning": 1, "critical": 2}

_NTFY_PRIORITY = {"info": "default", "warning": "high", "critical": "urgent"}
_PO_PRIORITY   = {"info": 0,         "warning": 1,      "critical": 2}


def _level_ok(level: str) -> bool:
    return _LEVELS.get(level, 1) >= _LEVELS.get(_MIN_LEVEL, 1)


async def notify(
    message: str,
    title: str = "MYCONEX",
    level: Literal["info", "warning", "critical"] = "warning",
    tags: list[str] | None = None,
) -> None:
    """Send a push notification via ntfy.sh and/or Pushover."""
    if not _level_ok(level):
        return

    tasks = []
    if _NTFY_URL:
        tasks.append(_ntfy(message, title, level, tags or []))
    if _PO_TOKEN and _PO_USER:
        tasks.append(_pushover(message, title, level))

    if not tasks:
        logger.debug("[notifier] no push targets configured (set NTFY_URL or PUSHOVER_TOKEN)")
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.warning("[notifier] send error: %s", r)


async def notify_critical(message: str, title: str = "MYCONEX CRITICAL") -> None:
    await notify(message, title=title, level="critical", tags=["rotating_light"])


# ── ntfy.sh ───────────────────────────────────────────────────────────────────

async def _ntfy(message: str, title: str, level: str, tags: list[str]) -> None:
    def _send():
        headers: dict[str, str] = {
            "Title":    title,
            "Priority": _NTFY_PRIORITY.get(level, "default"),
            "Tags":     ",".join(tags) if tags else level,
            "Content-Type": "text/plain",
        }
        if _NTFY_TOKEN:
            headers["Authorization"] = f"Bearer {_NTFY_TOKEN}"
        req = urllib.request.Request(
            _NTFY_URL,
            data=message.encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8):
            pass

    try:
        await asyncio.get_event_loop().run_in_executor(None, _send)
        logger.info("[notifier] ntfy sent: %s", title)
    except Exception as exc:
        raise RuntimeError(f"ntfy failed: {exc}") from exc


# ── Pushover ──────────────────────────────────────────────────────────────────

async def _pushover(message: str, title: str, level: str) -> None:
    def _send():
        payload = urllib.parse.urlencode({
            "token":    _PO_TOKEN,
            "user":     _PO_USER,
            "title":    title,
            "message":  message,
            "priority": _PO_PRIORITY.get(level, 0),
            # retry+expire required for priority=2
            "retry":    "60",
            "expire":   "3600",
        }).encode()
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=payload,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            result = json.loads(r.read())
            if result.get("status") != 1:
                raise RuntimeError(f"Pushover error: {result}")

    try:
        await asyncio.get_event_loop().run_in_executor(None, _send)
        logger.info("[notifier] pushover sent: %s", title)
    except Exception as exc:
        raise RuntimeError(f"pushover failed: {exc}") from exc


# ── Health watchdog ────────────────────────────────────────────────────────────

async def run_health_watchdog(interval_minutes: int = 15) -> None:
    """
    Periodically checks disk usage and sends a push alert if thresholds are exceeded.
    Runs as a background task.
    """
    import time
    last_disk_alert: float = 0.0

    logger.info("[notifier] health watchdog started — interval=%dm", interval_minutes)
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            try:
                import psutil
                disk = psutil.disk_usage("/")
                if disk.percent >= 90 and time.time() - last_disk_alert > 3600:
                    await notify(
                        f"Disk usage at {disk.percent:.0f}% — {disk.free // (1<<30)}GB free",
                        title="MYCONEX: Disk Warning",
                        level="critical",
                        tags=["warning", "disk"],
                    )
                    last_disk_alert = time.time()
            except ImportError:
                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[notifier] watchdog check error: %s", exc)
