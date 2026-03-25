"""
MYCONEX YouTube Ingester
-------------------------
Processes YouTube videos through Fabric patterns and stores extracted wisdom
alongside the email pipeline.

Two ingestion modes:

  1. Watch history  — reads a Google Takeout watch-history.json file and
                      processes any videos not yet seen.
                      Export at: https://takeout.google.com
                      (select "YouTube and YouTube Music" → "watch-history.json")

  2. On-demand URL  — processes a single YouTube URL immediately.

Outputs go to the shared wisdom store (~/.myconex/wisdom_store.json) and
interest profile (~/.myconex/interest_profile.json) so Buzlock builds one
unified knowledge base from both emails and videos.

.env keys:
  YOUTUBE_WATCH_HISTORY_PATH  — path to watch-history.json (enables auto-polling)
  YOUTUBE_INGEST_INTERVAL     — poll interval in minutes (default 60)
  YOUTUBE_INGEST_BATCH        — videos to process per poll pass (default 10)
  YOUTUBE_FABRIC_PATTERNS     — comma-separated patterns (default: extract_wisdom,summarize)
  YOUTUBE_FABRIC_ENABLED      — set false to skip Fabric (default: true)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR      = Path.home() / ".myconex"
_SEEN_FILE     = _BASE_DIR / "youtube_seen_ids.json"
_INSIGHTS_FILE = _BASE_DIR / "youtube_insights.json"
_WISDOM_FILE   = _BASE_DIR / "wisdom_store.json"
_PROFILE_FILE  = _BASE_DIR / "interest_profile.json"
_MEMORY_FILE   = _BASE_DIR / "memory.json"

# ── Config ─────────────────────────────────────────────────────────────────────
_HISTORY_PATH       = os.getenv("YOUTUBE_WATCH_HISTORY_PATH", "")
_WATCH_LATER_PATH   = os.getenv("YOUTUBE_WATCH_LATER_PATH", "")
_INGEST_INTERVAL    = int(os.getenv("YOUTUBE_INGEST_INTERVAL", "60"))
_BATCH_SIZE         = int(os.getenv("YOUTUBE_INGEST_BATCH", "10"))
_FABRIC_PATTERNS = [
    p.strip()
    for p in os.getenv("YOUTUBE_FABRIC_PATTERNS", "extract_wisdom,summarize").split(",")
    if p.strip()
]
_FABRIC_ENABLED  = os.getenv("YOUTUBE_FABRIC_ENABLED", "true").lower() != "false"


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save_json(path: Path, data: Any) -> None:
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Video ID extraction ────────────────────────────────────────────────────────

def _video_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# ── Google Takeout watch history parser ───────────────────────────────────────

def load_watch_history(path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a Google Takeout watch-history.json file.

    Returns a list of dicts with keys: video_id, url, title, watched_at.
    Filters out ads, shorts without IDs, and non-YouTube entries.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Watch history not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    entries: list[dict[str, Any]] = []

    for item in raw:
        title_url = item.get("titleUrl", "")
        if not title_url:
            continue
        vid_id = _video_id(title_url)
        if not vid_id:
            continue
        title = item.get("title", "")
        # Strip "Watched " prefix that Takeout adds
        if title.startswith("Watched "):
            title = title[8:]
        entries.append({
            "video_id": vid_id,
            "url": _canonical_url(vid_id),
            "title": title,
            "watched_at": item.get("time", ""),
        })

    # Deduplicate by video_id, keep most recent watch
    seen: dict[str, dict] = {}
    for e in entries:
        vid = e["video_id"]
        if vid not in seen or e["watched_at"] > seen[vid]["watched_at"]:
            seen[vid] = e
    return list(seen.values())


def load_watch_later(path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a Google Takeout Watch Later CSV.

    Takeout path: Takeout/YouTube and YouTube Music/playlists/Watch later-videos.csv
    Columns (Takeout format): Video ID, Playlist Video Creation Timestamp, Video Title

    Returns list of dicts with: video_id, url, title, watched_at (set to added timestamp).
    Sorted oldest → newest by the timestamp the video was added to Watch Later.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Watch Later CSV not found: {p}")

    entries: list[dict[str, Any]] = []
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        # Takeout CSVs sometimes have comment lines at the top
        lines = [l for l in f.readlines() if not l.startswith("#")]

    reader = csv.DictReader(lines)
    for row in reader:
        # Column names vary slightly between Takeout exports
        vid_id = (
            row.get("Video ID")
            or row.get("video id")
            or row.get("videoId")
            or ""
        ).strip()
        if not vid_id or len(vid_id) != 11:
            continue
        title = (
            row.get("Video Title")
            or row.get("title")
            or row.get("video title")
            or ""
        ).strip()
        timestamp = (
            row.get("Playlist Video Creation Timestamp")
            or row.get("timestamp")
            or ""
        ).strip()
        entries.append({
            "video_id": vid_id,
            "url": _canonical_url(vid_id),
            "title": title,
            "watched_at": timestamp,
            "source": "watch_later",
        })

    # Oldest → newest
    entries.sort(key=lambda e: e.get("watched_at", ""))
    return entries


def _auto_find_watch_later(history_path: str) -> str:
    """
    Given the watch-history.json path, try to auto-locate Watch Later CSV
    in the same Takeout directory tree.
    """
    base = Path(history_path).expanduser().resolve().parent
    candidates = [
        base.parent / "playlists" / "Watch later-videos.csv",
        base / "playlists" / "Watch later-videos.csv",
        base.parent.parent / "playlists" / "Watch later-videos.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


# ── Fabric extraction ──────────────────────────────────────────────────────────

async def _fabric_process_video(
    video_id: str,
    url: str,
    title: str,
) -> dict[str, Any]:
    """Get transcript and run Fabric patterns. Returns result dict."""
    if not _FABRIC_ENABLED:
        return {}

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from integrations.fabric_client import youtube_transcript, apply_pattern, is_available
        if not is_available():
            logger.debug("[yt_ingester] Fabric not available")
            return {}
    except ImportError as exc:
        logger.warning("[yt_ingester] fabric_client import error: %s", exc)
        return {}

    # Get transcript
    try:
        transcript = await youtube_transcript(url)
        if not transcript or len(transcript.strip()) < 50:
            logger.info("[yt_ingester] no transcript for %s (%s)", video_id, title[:50])
            return {"error": "no_transcript"}
    except Exception as exc:
        logger.warning("[yt_ingester] transcript fetch failed for %s: %s", video_id, exc)
        return {"error": str(exc)}

    result: dict[str, Any] = {
        "patterns_run": [],
        "raw": {},
        "transcript_length": len(transcript),
    }

    for pattern in _FABRIC_PATTERNS:
        try:
            output = await apply_pattern(pattern, transcript)
            if output:
                result["patterns_run"].append(pattern)
                result["raw"][pattern] = output
                logger.debug("[yt_ingester] pattern '%s' → %d chars for %s", pattern, len(output), video_id)
        except Exception as exc:
            logger.warning("[yt_ingester] pattern '%s' failed for %s: %s", pattern, video_id, exc)

    return result


# ── Profile / wisdom update ────────────────────────────────────────────────────

def _parse_fabric_sections(text: str) -> dict[str, list[str]]:
    """Same section parser as email_ingester."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip().upper()
            sections.setdefault(current, [])
        elif current and stripped.startswith(("-", "*", "•")):
            item = stripped.lstrip("-*• ").strip()
            if item:
                sections[current].append(item)
        elif current and stripped and not stripped.startswith("#"):
            sections[current].append(stripped)
    return sections


def _update_profile_from_video(fabric_result: dict[str, Any], title: str) -> None:
    """Merge video wisdom into the shared interest profile."""
    profile = _load_json(_PROFILE_FILE, {
        "topics": {}, "project_ideas": [], "likes": {}, "dislikes": {},
        "people": {}, "keywords": {}, "wisdom_items": [],
        "email_count": 0, "video_count": 0, "last_updated": "",
    })

    section_map = {
        "IDEAS":           "project_ideas",
        "INSIGHTS":        "topics",
        "RECOMMENDATIONS": "project_ideas",
        "FACTS":           "keywords",
        "QUOTES":          "keywords",
        "HABITS":          "likes",
        "ONE-SENTENCE TAKEAWAY": "wisdom_items",
        "SUMMARY":         "wisdom_items",
    }

    def _bump(counter: dict, items: list) -> None:
        for item in items:
            item = item.strip()
            if item and len(item) < 200:
                counter[item] = counter.get(item, 0) + 1

    for pattern_output in fabric_result.get("raw", {}).values():
        sections = _parse_fabric_sections(pattern_output)
        for section, items in sections.items():
            field = section_map.get(section)
            if not field:
                continue
            if field in ("topics", "keywords", "likes"):
                _bump(profile[field], items)
            elif field == "project_ideas":
                for idea in items:
                    idea = idea.strip()
                    if idea and idea not in profile["project_ideas"] and len(idea) < 300:
                        profile["project_ideas"].append(idea)
            elif field == "wisdom_items":
                for item in items:
                    item = item.strip()
                    if item and item not in profile["wisdom_items"] and len(item) < 500:
                        profile["wisdom_items"].append(item)

    profile["project_ideas"] = profile["project_ideas"][-200:]
    profile["wisdom_items"]  = profile["wisdom_items"][-500:]
    profile["video_count"]   = profile.get("video_count", 0) + 1
    profile["last_updated"]  = datetime.now(timezone.utc).isoformat()
    _save_json(_PROFILE_FILE, profile)

    # Sync to shared memory
    memory = _load_json(_MEMORY_FILE, {})
    def _top(counter: dict, n: int = 10) -> str:
        return ", ".join(k for k, _ in sorted(counter.items(), key=lambda x: -x[1])[:n])

    memory["interests_topics"]    = _top(profile["topics"])
    memory["interests_keywords"]  = _top(profile["keywords"])
    memory["interests_wisdom"]    = " | ".join(profile.get("wisdom_items", [])[-10:])
    memory["interests_project_ideas"] = " | ".join(profile["project_ideas"][-20:])
    memory["interests_video_count"] = str(profile["video_count"])
    memory["interests_last_updated"] = profile["last_updated"]
    _save_json(_MEMORY_FILE, memory)


def _save_wisdom_entry(video: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    """Append to the shared wisdom store."""
    if not fabric_result.get("patterns_run"):
        return
    store = _load_json(_WISDOM_FILE, [])
    store.append({
        "source": "youtube",
        "video_id": video["video_id"],
        "url": video["url"],
        "title": video.get("title", ""),
        "watched_at": video.get("watched_at", ""),
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "patterns": fabric_result.get("patterns_run", []),
        "transcript_length": fabric_result.get("transcript_length", 0),
        "raw": fabric_result.get("raw", {}),
    })
    _save_json(_WISDOM_FILE, store[-500:])


def _save_insight_entry(video: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    """Append to the YouTube-specific insights log."""
    log = _load_json(_INSIGHTS_FILE, [])
    log.append({
        "video_id": video["video_id"],
        "url": video["url"],
        "title": video.get("title", ""),
        "watched_at": video.get("watched_at", ""),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "fabric_patterns": fabric_result.get("patterns_run", []),
        "transcript_length": fabric_result.get("transcript_length", 0),
        "summary": fabric_result.get("raw", {}).get("summarize", "")[:500],
        "error": fabric_result.get("error", ""),
    })
    _save_json(_INSIGHTS_FILE, log[-1000:])


# ── Main ingester class ────────────────────────────────────────────────────────

class YouTubeIngester:
    """
    Polls a Google Takeout watch-history.json file and processes new videos.

    Usage:
        ingester = YouTubeIngester()
        asyncio.create_task(ingester.run_forever())

    Or on-demand:
        ingester = YouTubeIngester()
        await ingester.process_url("https://youtube.com/watch?v=...")
    """

    def __init__(
        self,
        history_path: str = "",
        watch_later_path: str = "",
        interval_minutes: int = _INGEST_INTERVAL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self.history_path = history_path or _HISTORY_PATH
        # Auto-find Watch Later if not explicitly set
        wl = watch_later_path or _WATCH_LATER_PATH
        if not wl and self.history_path:
            wl = _auto_find_watch_later(self.history_path)
        self.watch_later_path = wl
        self.interval = interval_minutes * 60
        self.batch_size = batch_size
        self._running = False

    # ── Background task ───────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        if not self.history_path:
            logger.info("[yt_ingester] no YOUTUBE_WATCH_HISTORY_PATH set — background polling disabled")
            return
        self._running = True
        logger.info("[yt_ingester] starting — history=%s interval=%dm", self.history_path, self.interval // 60)
        while self._running:
            try:
                count = await self.ingest_history()
                if count:
                    logger.info("[yt_ingester] processed %d new video(s)", count)
            except Exception as exc:
                logger.warning("[yt_ingester] ingest error: %s", exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    # ── Batch history ingest ──────────────────────────────────────────────────

    async def ingest_history(self) -> int:
        """
        Process unseen videos from watch history and Watch Later list.
        Both sorted oldest → newest. Returns count of videos processed.
        """
        all_videos: list[dict[str, Any]] = []

        if self.history_path:
            try:
                all_videos.extend(load_watch_history(self.history_path))
            except Exception as exc:
                logger.warning("[yt_ingester] could not load watch history: %s", exc)

        if self.watch_later_path:
            try:
                wl = load_watch_later(self.watch_later_path)
                logger.info("[yt_ingester] Watch Later: %d entries from %s", len(wl), self.watch_later_path)
                all_videos.extend(wl)
            except Exception as exc:
                logger.warning("[yt_ingester] could not load Watch Later: %s", exc)

        if not all_videos:
            return 0

        seen: set[str] = set(_load_json(_SEEN_FILE, []))

        # Deduplicate across both sources, keep watch history entry if duplicate
        by_id: dict[str, dict] = {}
        for v in all_videos:
            vid = v["video_id"]
            if vid not in by_id or v.get("source") != "watch_later":
                by_id[vid] = v

        new_videos = [v for v in by_id.values() if v["video_id"] not in seen]

        # Oldest → newest across both sources
        new_videos.sort(key=lambda v: v.get("watched_at", ""))
        batch = new_videos[: self.batch_size]

        processed = 0
        for video in batch:
            success = await self._process_video(video)
            if success:
                processed += 1
            seen.add(video["video_id"])

        if batch:
            _save_json(_SEEN_FILE, list(seen))

        if processed:
            try:
                from core.notifications import notify
                profile = _load_json(_PROFILE_FILE, {})
                top_topics = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])[:3]
                topic_str = ", ".join(t for t, _ in top_topics) if top_topics else "various topics"
                recent_ideas = profile.get("project_ideas", [])[-2:]
                idea_str = ""
                if recent_ideas:
                    idea_str = "\n💡 Recent project ideas: " + " | ".join(recent_ideas)
                await notify(
                    f"📺 **YouTube digest** — processed {processed} new video(s)\n"
                    f"🏷️ Top topics: {topic_str}{idea_str}"
                )
            except Exception:
                pass

        return processed

    # ── Single URL processing ─────────────────────────────────────────────────

    async def process_url(self, url: str, patterns: list[str] | None = None) -> str:
        """
        Process a single YouTube URL on demand.
        Returns a formatted summary of what was extracted.
        """
        vid_id = _video_id(url)
        if not vid_id:
            return f"Could not extract video ID from URL: {url}"

        video = {
            "video_id": vid_id,
            "url": _canonical_url(vid_id),
            "title": "",
            "watched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Temporarily override patterns if provided
        global _FABRIC_PATTERNS
        original = _FABRIC_PATTERNS[:]
        if patterns:
            _FABRIC_PATTERNS = patterns

        try:
            fabric_result = await _fabric_process_video(vid_id, video["url"], "")
        finally:
            _FABRIC_PATTERNS = original

        if fabric_result.get("error"):
            return f"Could not process video: {fabric_result['error']}"

        if not fabric_result.get("patterns_run"):
            return "No patterns produced output — Fabric may not be installed or the video has no transcript."

        seen: set[str] = set(_load_json(_SEEN_FILE, []))
        seen.add(vid_id)
        _save_json(_SEEN_FILE, list(seen))

        _save_wisdom_entry(video, fabric_result)
        _save_insight_entry(video, fabric_result)
        _update_profile_from_video(fabric_result, video.get("title", ""))

        # Return the best pattern output
        raw = fabric_result.get("raw", {})
        primary = raw.get("extract_wisdom") or raw.get("summarize") or next(iter(raw.values()), "")
        return primary or "Processed — no text output returned."

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _process_video(self, video: dict[str, Any]) -> bool:
        """Process one video entry. Returns True if Fabric ran successfully."""
        vid_id = video["video_id"]
        title  = video.get("title", "")[:80]
        logger.info("[yt_ingester] processing %s  %s", vid_id, title)

        fabric_result = await _fabric_process_video(vid_id, video["url"], title)

        _save_insight_entry(video, fabric_result)

        if fabric_result.get("patterns_run"):
            _save_wisdom_entry(video, fabric_result)
            _update_profile_from_video(fabric_result, title)

            # Embed wisdom into Qdrant knowledge base
            wisdom_text = (fabric_result.get("raw", {}).get("extract_wisdom")
                           or fabric_result.get("raw", {}).get("summarize", ""))
            if wisdom_text:
                try:
                    from integrations.knowledge_store import embed_and_store
                    await embed_and_store(
                        text=wisdom_text,
                        source="youtube",
                        metadata={
                            "title":      video.get("title", ""),
                            "url":        video.get("url", ""),
                            "watched_at": video.get("watched_at", ""),
                            "video_id":   vid_id,
                        },
                    )
                except Exception as exc:
                    logger.debug("[yt_ingester] Qdrant embed skipped: %s", exc)

            return True
        return False

    # ── Summaries ────────────────────────────────────────────────────────────

    @staticmethod
    def get_recent_insights(n: int = 10) -> str:
        log = _load_json(_INSIGHTS_FILE, [])
        if not log:
            return "No YouTube videos have been processed yet."
        recent = log[-n:][::-1]
        lines = [f"Last {len(recent)} processed video(s):\n"]
        for e in recent:
            lines.append(f"  {e.get('title', e.get('video_id', ''))}")
            lines.append(f"    URL:     {e.get('url', '')}")
            lines.append(f"    Watched: {e.get('watched_at', '')}")
            if e.get("summary"):
                lines.append(f"    Summary: {e['summary'][:200]}")
            if e.get("fabric_patterns"):
                lines.append(f"    Fabric:  {', '.join(e['fabric_patterns'])}")
            if e.get("error"):
                lines.append(f"    Error:   {e['error']}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def get_wisdom(n: int = 5) -> str:
        store = _load_json(_WISDOM_FILE, [])
        yt_entries = [e for e in store if e.get("source") == "youtube"]
        if not yt_entries:
            return "No YouTube wisdom extracted yet."
        recent = yt_entries[-n:][::-1]
        lines = [f"Fabric wisdom from {len(recent)} video(s):\n"]
        for entry in recent:
            lines.append(f"{'─' * 60}")
            lines.append(f"{entry.get('title', entry.get('video_id', ''))}")
            lines.append(f"URL: {entry.get('url', '')}")
            lines.append(f"Watched: {entry.get('watched_at', '')}\n")
            raw = entry.get("raw", {})
            text = raw.get("extract_wisdom") or raw.get("summarize") or next(iter(raw.values()), "")
            if text:
                lines.append(text)
            lines.append("")
        return "\n".join(lines)
