"""
MYCONEX Notification Bus
-------------------------
Simple asyncio queue that background tasks (email ingester, YouTube ingester,
RSS monitor, etc.) push digest messages to, and the Discord gateway drains.

Usage — push a notification:
    from core.notifications import notify
    await notify("📧 Processed 5 new emails — 3 project ideas extracted.")

Usage — drain in gateway:
    from core.notifications import drain
    messages = await drain()
    for msg in messages:
        await channel.send(msg)

Usage — subscribe for SSE (dashboard):
    from core.notifications import subscribe, unsubscribe, get_recent
    q = subscribe()
    try:
        msg = await asyncio.wait_for(q.get(), timeout=15.0)
    finally:
        unsubscribe(q)
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

# SSE broadcast: each subscriber gets a copy of every notification
_subscribers: list[asyncio.Queue] = []

# Ring buffer for history — new SSE clients receive the last N entries on connect
_activity_log: deque = deque(maxlen=200)


async def notify(message: str) -> None:
    """Push a notification message (non-blocking; drops if queue full)."""
    _activity_log.append(message)

    try:
        _queue.put_nowait(message)
    except asyncio.QueueFull:
        pass  # Don't block background tasks on a full queue

    # Fan out to SSE subscribers; prune dead/full queues
    dead: list[asyncio.Queue] = []
    for q in _subscribers:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


async def drain() -> list[str]:
    """Pull all pending notifications without blocking."""
    messages: list[str] = []
    while not _queue.empty():
        try:
            messages.append(_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return messages


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Returns a queue that receives all future notifications."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a subscriber queue (call in finally block after client disconnect)."""
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


def get_recent(n: int = 50) -> list[str]:
    """Return the last n notifications from the ring buffer (oldest first)."""
    log = list(_activity_log)
    return log[-n:] if len(log) > n else log
