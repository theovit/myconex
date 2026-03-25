"""
MYCONEX Cross-Source Signal Detector
--------------------------------------
Detects when the same concept surfaces in 2+ different source types (email,
YouTube, RSS, podcast) within a rolling 7-day window.  When a signal is found
it pushes a notification to the Discord bus so Buzlock can surface it.

A "signal" is a Qdrant result cluster where:
  • The same query (topic from the interest profile) returns hits from ≥2 sources
  • Each hit has a relevance score ≥ SIGNAL_THRESHOLD

Runs as a lightweight periodic check (default every 6h) — cheap because it
reuses the existing Qdrant collection rather than embedding new data.

Usage (wired in buzlock_bot.py):
    from integrations.signal_detector import SignalDetector
    detector = SignalDetector()
    asyncio.create_task(detector.run_forever())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE          = Path.home() / ".myconex"
_PROFILE_FILE  = _BASE / "interest_profile.json"
_SIGNALS_FILE  = _BASE / "signals_log.json"

SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "0.72"))
SIGNAL_INTERVAL  = int(os.getenv("SIGNAL_INTERVAL_HOURS", "6"))
SIGNAL_LOOKBACK  = int(os.getenv("SIGNAL_LOOKBACK_DAYS", "7"))


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save(path: Path, data: Any) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class SignalDetector:
    """
    Periodically queries the Qdrant knowledge base for semantic clusters
    spanning multiple source types and pushes notifications for new signals.
    """

    def __init__(self, interval_hours: int = SIGNAL_INTERVAL) -> None:
        self.interval = interval_hours * 3600
        self._running = False

    async def run_forever(self) -> None:
        self._running = True
        logger.info("[signals] detector started — interval=%dh", self.interval // 3600)
        # Offset first run by 30 min so ingesters have time to populate on startup
        await asyncio.sleep(1800)
        while self._running:
            try:
                signals = await self.detect()
                if signals:
                    logger.info("[signals] %d new signal(s) detected", len(signals))
            except Exception as exc:
                logger.warning("[signals] detection error: %s", exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    async def detect(self) -> list[dict[str, Any]]:
        """
        Run a detection pass.  Returns list of new signals (also pushed to
        the notification bus and appended to signals_log.json).
        """
        try:
            from integrations.knowledge_store import search
        except ImportError:
            return []

        profile = _load(_PROFILE_FILE, {})
        topics = sorted(
            profile.get("topics", {}).items(),
            key=lambda x: -x[1],
        )[:20]  # top 20 topics as probe queries

        if not topics:
            return []

        seen_signals = {
            s["topic"] for s in _load(_SIGNALS_FILE, [])
            if _is_recent(s.get("detected_at", ""), days=SIGNAL_LOOKBACK)
        }

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=SIGNAL_LOOKBACK)
        ).isoformat()

        new_signals: list[dict[str, Any]] = []

        for topic, _ in topics:
            if topic in seen_signals:
                continue

            results = await search(
                query=topic,
                limit=20,
                score_threshold=SIGNAL_THRESHOLD,
            )

            # Filter to entries within the lookback window
            recent = [
                r for r in results
                if (r.get("metadata", {}).get("stored_at", "") >= cutoff
                    or r.get("metadata", {}).get("published", "") >= cutoff[:10])
            ]

            # Group by source
            by_source: dict[str, list[dict]] = {}
            for r in recent:
                src = r.get("source", "unknown")
                by_source.setdefault(src, []).append(r)

            # Signal fires when 2+ distinct sources mention this topic
            if len(by_source) >= 2:
                sources = sorted(by_source.keys())
                signal = {
                    "topic":       topic,
                    "sources":     sources,
                    "hit_count":   len(recent),
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "top_hit":     recent[0].get("content", "")[:200],
                }
                new_signals.append(signal)

                # Push notification
                try:
                    from core.notifications import notify
                    source_str = " + ".join(
                        _SOURCE_EMOJI.get(s, "📌") + s for s in sources[:3]
                    )
                    await notify(
                        f"🔔 **Signal detected**: `{topic}`\n"
                        f"Mentioned across {source_str} in the last {SIGNAL_LOOKBACK} days\n"
                        f"_{recent[0].get('content', '')[:120]}_"
                    )
                except Exception:
                    pass

        if new_signals:
            existing = _load(_SIGNALS_FILE, [])
            existing.extend(new_signals)
            _save(_SIGNALS_FILE, existing[-500:])

        return new_signals


_SOURCE_EMOJI = {
    "email":   "📧 ",
    "youtube": "📺 ",
    "rss":     "📰 ",
    "podcast": "🎙️ ",
}


def _is_recent(ts: str, days: int) -> bool:
    if not ts:
        return False
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return ts >= cutoff
