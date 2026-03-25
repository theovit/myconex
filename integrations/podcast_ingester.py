"""
MYCONEX Podcast Ingester
--------------------------
Downloads podcast episodes, transcribes audio with Whisper, runs Fabric patterns
on the transcript, and stores extracted wisdom in the shared knowledge base.

Pipeline per episode:
  1. Parse podcast RSS feed for new episodes
  2. Download audio with yt-dlp (respects rate limits, auto-format selection)
  3. Transcribe with whisper.cpp or openai-whisper (whichever is available)
  4. Run Fabric patterns on the transcript (extract_wisdom, summarize, etc.)
  5. Embed and store in Qdrant knowledge base

.env keys:
  PODCAST_FEEDS           — semicolon-separated podcast RSS feed URLs
  PODCAST_INGEST_INTERVAL — poll interval in minutes (default 120)
  PODCAST_INGEST_BATCH    — max new episodes to process per poll (default 3)
  PODCAST_FABRIC_PATTERNS — comma-separated Fabric patterns (default: extract_wisdom,summarize)
  PODCAST_FABRIC_ENABLED  — set false to skip Fabric (default: true)
  PODCAST_WHISPER_MODEL   — Whisper model size: tiny/base/small/medium/large (default: base)
  PODCAST_DOWNLOAD_DIR    — where to store audio files (default: ~/.myconex/podcast_audio)
  PODCAST_KEEP_AUDIO      — keep audio after transcription (default: false)
  PODCAST_MAX_DURATION    — skip episodes longer than N minutes (0=no limit, default: 180)

Requirements:
  pip install feedparser yt-dlp openai-whisper
  OR whisper.cpp compiled at ~/whisper.cpp/build/bin/whisper-cli

Outputs:
  ~/.myconex/podcast_seen_ids.json     — processed episode IDs
  ~/.myconex/podcast_insights.json     — per-episode insight log
  ~/.myconex/wisdom_store.json         — shared wisdom store
  ~/.myconex/interest_profile.json     — shared interest profile

Usage:
    from integrations.podcast_ingester import PodcastIngester
    ingester = PodcastIngester()
    asyncio.create_task(ingester.run_forever())
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR       = Path.home() / ".myconex"
_SEEN_FILE      = _BASE_DIR / "podcast_seen_ids.json"
_INSIGHTS_FILE  = _BASE_DIR / "podcast_insights.json"
_WISDOM_FILE    = _BASE_DIR / "wisdom_store.json"
_PROFILE_FILE   = _BASE_DIR / "interest_profile.json"
_FEEDS_FILE     = _BASE_DIR / "podcast_feeds.json"

# ── Config ─────────────────────────────────────────────────────────────────────
_INGEST_INTERVAL  = int(os.getenv("PODCAST_INGEST_INTERVAL", "120"))
_BATCH_SIZE       = int(os.getenv("PODCAST_INGEST_BATCH", "3"))
_FABRIC_PATTERNS  = [
    p.strip()
    for p in os.getenv("PODCAST_FABRIC_PATTERNS", "extract_wisdom,summarize").split(",")
    if p.strip()
]
_FABRIC_ENABLED   = os.getenv("PODCAST_FABRIC_ENABLED", "true").lower() != "false"
_WHISPER_MODEL    = os.getenv("PODCAST_WHISPER_MODEL", "base")
_DOWNLOAD_DIR     = Path(os.getenv("PODCAST_DOWNLOAD_DIR", str(_BASE_DIR / "podcast_audio")))
_KEEP_AUDIO       = os.getenv("PODCAST_KEEP_AUDIO", "false").lower() == "true"
_MAX_DURATION_MIN = int(os.getenv("PODCAST_MAX_DURATION", "180"))

_ENV_FEEDS = [
    f.strip()
    for f in os.getenv("PODCAST_FEEDS", "").split(";")
    if f.strip()
]


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


def _episode_id(url: str, title: str) -> str:
    key = url.strip() if url.strip() else title.strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Dependency detection ───────────────────────────────────────────────────────

def _find_whisper_cpp() -> str | None:
    """Locate whisper.cpp binary."""
    candidates = [
        Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        Path.home() / "whisper.cpp" / "build" / "bin" / "main",
        Path("/usr/local/bin/whisper"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    found = shutil.which("whisper-cli") or shutil.which("whisper")
    return found


def _whisper_available() -> bool:
    """Check if any Whisper implementation is available."""
    if _find_whisper_cpp():
        return True
    try:
        import whisper  # type: ignore[import]
        return True
    except ImportError:
        pass
    return False


def _ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None or shutil.which("yt_dlp") is not None


# ── Feed parsing ───────────────────────────────────────────────────────────────

async def _fetch_podcast_feed(url: str) -> list[dict[str, Any]]:
    """Parse a podcast RSS feed and return episode list."""
    try:
        import feedparser  # type: ignore[import]
    except ImportError:
        logger.warning("[podcast] feedparser not installed — run: pip install feedparser")
        return []

    try:
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, url)
    except Exception as exc:
        logger.warning("[podcast] feed fetch failed for %s: %s", url, exc)
        return []

    episodes: list[dict[str, Any]] = []
    feed_title = feed.get("feed", {}).get("title", urlparse(url).netloc)

    for entry in feed.get("entries", []):
        # Find audio enclosure
        audio_url = ""
        duration_min = 0
        for enc in entry.get("enclosures", []):
            mime = enc.get("type", "")
            if "audio" in mime or enc.get("href", "").endswith((".mp3", ".m4a", ".ogg")):
                audio_url = enc.get("href", "")
                # Duration can be in itunes:duration tag
                break

        if not audio_url:
            # Some feeds put the audio in the link
            link = entry.get("link", "")
            if any(link.endswith(ext) for ext in (".mp3", ".m4a", ".ogg", ".wav")):
                audio_url = link

        if not audio_url:
            continue

        # itunes duration (HH:MM:SS or seconds string)
        itunes_dur = entry.get("itunes_duration", "")
        if itunes_dur:
            try:
                parts = str(itunes_dur).split(":")
                if len(parts) == 3:
                    duration_min = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 2:
                    duration_min = int(parts[0])
                else:
                    duration_min = int(itunes_dur) // 60
            except (ValueError, IndexError):
                pass

        title = entry.get("title", "")
        published = entry.get("published", entry.get("updated", ""))
        ep_id = _episode_id(audio_url, title)

        episodes.append({
            "id":           ep_id,
            "title":        title,
            "audio_url":    audio_url,
            "published":    published,
            "feed_url":     url,
            "feed_title":   feed_title,
            "duration_min": duration_min,
            "description":  entry.get("summary", "")[:500],
        })

    return episodes


# ── Audio download ─────────────────────────────────────────────────────────────

async def _download_audio(url: str, output_dir: Path) -> str | None:
    """
    Download audio from URL using yt-dlp.
    Returns the path to the downloaded file, or None on failure.
    """
    if not _ytdlp_available():
        logger.warning("[podcast] yt-dlp not installed — run: pip install yt-dlp")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",      # ~128kbps, good balance for Whisper
        "--no-progress",
        "--quiet",
        "-o", output_template,
        url,
    ]

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            ),
        )
        if result.returncode != 0:
            logger.warning("[podcast] yt-dlp failed for %s: %s", url, result.stderr[:200])
            return None
        # Find the downloaded file
        mp3_files = sorted(output_dir.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
        if mp3_files:
            return str(mp3_files[-1])
    except subprocess.TimeoutExpired:
        logger.warning("[podcast] yt-dlp timed out for %s", url)
    except Exception as exc:
        logger.warning("[podcast] download error for %s: %s", url, exc)
    return None


# ── Transcription ──────────────────────────────────────────────────────────────

async def _transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using whisper.cpp or openai-whisper.
    Returns the transcript text.
    """
    whisper_bin = _find_whisper_cpp()

    if whisper_bin:
        return await _transcribe_whisper_cpp(audio_path, whisper_bin)

    # Fall back to openai-whisper Python package
    try:
        import whisper  # type: ignore[import]
        return await _transcribe_openai_whisper(audio_path, whisper)
    except ImportError:
        pass

    logger.warning("[podcast] no Whisper implementation found")
    return ""


async def _transcribe_whisper_cpp(audio_path: str, binary: str) -> str:
    """Transcribe using whisper.cpp binary."""
    model_path = Path.home() / "whisper.cpp" / "models" / f"ggml-{_WHISPER_MODEL}.bin"
    if not model_path.exists():
        # Try common alternate locations
        alt = Path.home() / ".cache" / "whisper" / f"ggml-{_WHISPER_MODEL}.bin"
        if alt.exists():
            model_path = alt
        else:
            logger.warning("[podcast] whisper.cpp model not found at %s", model_path)
            return ""

    cmd = [binary, "-m", str(model_path), "-f", audio_path, "--output-txt", "--no-prints"]
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            ),
        )
        if result.returncode == 0:
            # whisper.cpp writes to <audio_path>.txt
            txt_file = Path(audio_path + ".txt")
            if txt_file.exists():
                transcript = txt_file.read_text()
                txt_file.unlink()
                return transcript.strip()
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("[podcast] whisper.cpp timed out for %s", audio_path)
    except Exception as exc:
        logger.warning("[podcast] whisper.cpp error: %s", exc)
    return ""


async def _transcribe_openai_whisper(audio_path: str, whisper_module: Any) -> str:
    """Transcribe using the openai-whisper Python package."""
    try:
        loop = asyncio.get_event_loop()

        def _run() -> str:
            model = whisper_module.load_model(_WHISPER_MODEL)
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()

        return await loop.run_in_executor(None, _run)
    except Exception as exc:
        logger.warning("[podcast] openai-whisper error: %s", exc)
        return ""


# ── Fabric processing ──────────────────────────────────────────────────────────

async def _fabric_process(text: str, title: str) -> dict[str, Any]:
    """Run configured Fabric patterns on transcript. Returns result dict."""
    if not _FABRIC_ENABLED or not text.strip():
        return {}

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from integrations.fabric_client import apply_pattern, is_available
        if not is_available():
            return {}
    except ImportError as exc:
        logger.warning("[podcast] fabric_client import error: %s", exc)
        return {}

    result: dict[str, Any] = {"patterns_run": [], "raw": {}}
    for pattern in _FABRIC_PATTERNS:
        try:
            output = await apply_pattern(pattern, text)
            if output:
                result["patterns_run"].append(pattern)
                result["raw"][pattern] = output
        except Exception as exc:
            logger.warning("[podcast] pattern '%s' failed for '%s': %s", pattern, title[:50], exc)

    return result


# ── Profile / wisdom update ────────────────────────────────────────────────────

def _parse_fabric_sections(text: str) -> dict[str, list[str]]:
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


def _update_profile_from_episode(fabric_result: dict[str, Any], title: str) -> None:
    profile = _load_json(_PROFILE_FILE, {
        "topics": {}, "project_ideas": [], "likes": {}, "dislikes": {},
        "people": {}, "keywords": {}, "wisdom_items": [],
        "email_count": 0, "video_count": 0, "rss_count": 0,
        "podcast_count": 0, "last_updated": "",
    })

    section_map = {
        "IDEAS":                  "project_ideas",
        "INSIGHTS":               "topics",
        "RECOMMENDATIONS":        "project_ideas",
        "FACTS":                  "keywords",
        "QUOTES":                 "keywords",
        "ONE-SENTENCE TAKEAWAY":  "wisdom_items",
        "SUMMARY":                "wisdom_items",
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

    profile["project_ideas"]  = profile["project_ideas"][-200:]
    profile["wisdom_items"]   = profile["wisdom_items"][-500:]
    profile["podcast_count"]  = profile.get("podcast_count", 0) + 1
    profile["last_updated"]   = datetime.now(timezone.utc).isoformat()
    _save_json(_PROFILE_FILE, profile)


def _save_wisdom_entry(episode: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    if not fabric_result.get("patterns_run"):
        return
    store = _load_json(_WISDOM_FILE, [])
    store.append({
        "source":      "podcast",
        "episode_id":  episode["id"],
        "title":       episode["title"],
        "feed_title":  episode.get("feed_title", ""),
        "audio_url":   episode.get("audio_url", ""),
        "published":   episode.get("published", ""),
        "stored_at":   datetime.now(timezone.utc).isoformat(),
        "patterns":    fabric_result.get("patterns_run", []),
        "raw":         fabric_result.get("raw", {}),
    })
    _save_json(_WISDOM_FILE, store[-500:])


def _save_insight_entry(episode: dict[str, Any], fabric_result: dict[str, Any],
                        transcript_len: int = 0) -> None:
    log = _load_json(_INSIGHTS_FILE, [])
    log.append({
        "episode_id":      episode["id"],
        "title":           episode["title"],
        "feed_title":      episode.get("feed_title", ""),
        "feed_url":        episode.get("feed_url", ""),
        "audio_url":       episode.get("audio_url", ""),
        "published":       episode.get("published", ""),
        "processed_at":    datetime.now(timezone.utc).isoformat(),
        "transcript_len":  transcript_len,
        "fabric_patterns": fabric_result.get("patterns_run", []),
        "summary":         fabric_result.get("raw", {}).get("summarize", "")[:500],
    })
    _save_json(_INSIGHTS_FILE, log[-500:])


# ── Main ingester class ────────────────────────────────────────────────────────

class PodcastIngester:
    """
    Polls podcast RSS feeds, downloads new episodes, transcribes with Whisper,
    runs Fabric patterns on the transcript, and stores wisdom in the knowledge base.

    Usage:
        ingester = PodcastIngester()
        asyncio.create_task(ingester.run_forever())
    """

    def __init__(
        self,
        feeds: list[str] | None = None,
        interval_minutes: int = _INGEST_INTERVAL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        persisted: list[str] = _load_json(_FEEDS_FILE, [])
        combined = list(dict.fromkeys(persisted + _ENV_FEEDS + (feeds or [])))
        self._feeds: list[str] = combined
        self.interval   = interval_minutes * 60
        self.batch_size = batch_size
        self._running   = False

    # ── Feed management ───────────────────────────────────────────────────────

    def add_feed(self, url: str) -> bool:
        url = url.strip()
        if not url or url in self._feeds:
            return False
        self._feeds.append(url)
        _save_json(_FEEDS_FILE, self._feeds)
        return True

    def remove_feed(self, url: str) -> bool:
        url = url.strip()
        if url in self._feeds:
            self._feeds.remove(url)
            _save_json(_FEEDS_FILE, self._feeds)
            return True
        return False

    def list_feeds(self) -> list[str]:
        return list(self._feeds)

    # ── Capability check ──────────────────────────────────────────────────────

    @staticmethod
    def check_dependencies() -> str:
        lines = ["Podcast ingester dependencies:"]
        lines.append(f"  feedparser:    {'✅' if _check_import('feedparser') else '❌ pip install feedparser'}")
        lines.append(f"  yt-dlp:        {'✅' if _ytdlp_available() else '❌ pip install yt-dlp'}")
        whisper_cpp = _find_whisper_cpp()
        if whisper_cpp:
            lines.append(f"  whisper.cpp:   ✅ {whisper_cpp}")
        elif _check_import("whisper"):
            lines.append("  openai-whisper: ✅")
        else:
            lines.append("  whisper:        ❌ pip install openai-whisper  OR  build whisper.cpp")
        return "\n".join(lines)

    # ── Background task ───────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        if not self._feeds:
            logger.info("[podcast] no feeds configured — background polling disabled")
            return
        self._running = True
        logger.info(
            "[podcast] starting — %d feed(s), interval=%dm, whisper model=%s",
            len(self._feeds), self.interval // 60, _WHISPER_MODEL,
        )
        while self._running:
            try:
                count = await self.poll_all()
                if count:
                    logger.info("[podcast] processed %d new episode(s)", count)
            except Exception as exc:
                logger.warning("[podcast] poll error: %s", exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    # ── Poll all feeds ────────────────────────────────────────────────────────

    async def poll_all(self) -> int:
        if not self._feeds:
            return 0

        seen: set[str] = set(_load_json(_SEEN_FILE, []))
        all_new: list[dict[str, Any]] = []

        fetch_tasks = [_fetch_podcast_feed(url) for url in self._feeds]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for feed_url, episodes in zip(self._feeds, results):
            if isinstance(episodes, Exception):
                logger.warning("[podcast] error fetching %s: %s", feed_url, episodes)
                continue
            for ep in episodes:
                if ep["id"] not in seen:
                    # Skip episodes exceeding max duration
                    if _MAX_DURATION_MIN > 0 and ep["duration_min"] > _MAX_DURATION_MIN:
                        logger.info(
                            "[podcast] skipping long episode (%dm): %s",
                            ep["duration_min"], ep["title"][:60],
                        )
                        continue
                    all_new.append(ep)

        # Oldest → newest
        all_new.sort(key=lambda e: e.get("published", ""))
        batch = all_new[: self.batch_size]

        processed = 0
        for episode in batch:
            success = await self._process_episode(episode)
            if success:
                processed += 1
            seen.add(episode["id"])

        if batch:
            _save_json(_SEEN_FILE, list(seen))

        if processed:
            try:
                from core.notifications import notify
                profile = _load_json(_PROFILE_FILE, {})
                top_topics = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])[:3]
                topic_str = ", ".join(t for t, _ in top_topics) if top_topics else "various topics"
                show_names = list(dict.fromkeys(
                    e.get("feed_title", "") for e in batch[:processed]
                ))[:3]
                shows_str = ", ".join(s for s in show_names if s) or "podcasts"
                await notify(
                    f"🎙️ **Podcast digest** — {processed} new episode(s) from {shows_str}\n"
                    f"🏷️ Top topics: {topic_str}"
                )
            except Exception:
                pass

        return processed

    # ── Single episode ────────────────────────────────────────────────────────

    async def _process_episode(self, episode: dict[str, Any]) -> bool:
        """Download, transcribe, and extract wisdom from one episode. Returns True on success."""
        title = episode["title"][:70]
        logger.info("[podcast] processing: %s  [%s]", title, episode.get("feed_title", ""))

        # Download audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = await _download_audio(episode["audio_url"], Path(tmp_dir))
            if not audio_path:
                _save_insight_entry(episode, {})
                return False

            # Optionally persist audio
            if _KEEP_AUDIO:
                _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
                dest = _DOWNLOAD_DIR / Path(audio_path).name
                shutil.copy2(audio_path, dest)
                audio_path = str(dest)

            # Transcribe
            transcript = await _transcribe_audio(audio_path)

        if not transcript or len(transcript.strip()) < 100:
            logger.info("[podcast] empty/short transcript for: %s", title)
            _save_insight_entry(episode, {}, len(transcript))
            return False

        logger.info("[podcast] transcript %d chars for: %s", len(transcript), title)

        # Fabric extraction
        fabric_result = await _fabric_process(transcript, title)
        _save_insight_entry(episode, fabric_result, len(transcript))

        if fabric_result.get("patterns_run"):
            _save_wisdom_entry(episode, fabric_result)
            _update_profile_from_episode(fabric_result, title)

            # Embed into Qdrant
            wisdom_text = (
                fabric_result.get("raw", {}).get("extract_wisdom")
                or fabric_result.get("raw", {}).get("summarize", "")
            )
            if wisdom_text:
                try:
                    from integrations.knowledge_store import embed_and_store
                    await embed_and_store(
                        text=wisdom_text,
                        source="podcast",
                        metadata={
                            "title":      episode.get("title", ""),
                            "feed_title": episode.get("feed_title", ""),
                            "audio_url":  episode.get("audio_url", ""),
                            "published":  episode.get("published", ""),
                        },
                    )
                except Exception as exc:
                    logger.debug("[podcast] Qdrant embed skipped: %s", exc)

            return True
        return False

    # ── Summaries ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_recent_insights(n: int = 10) -> str:
        log = _load_json(_INSIGHTS_FILE, [])
        if not log:
            return "No podcast episodes have been processed yet."
        recent = log[-n:][::-1]
        lines = [f"Last {len(recent)} processed episode(s):\n"]
        for e in recent:
            lines.append(f"  [{e.get('feed_title', '')}]  {e.get('title', '')}")
            lines.append(f"    Published: {e.get('published', '')}")
            lines.append(f"    Transcript: {e.get('transcript_len', 0):,} chars")
            if e.get("summary"):
                lines.append(f"    Summary:   {e['summary'][:200]}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def get_wisdom(n: int = 5) -> str:
        store = _load_json(_WISDOM_FILE, [])
        pod_entries = [e for e in store if e.get("source") == "podcast"]
        if not pod_entries:
            return "No podcast wisdom extracted yet."
        recent = pod_entries[-n:][::-1]
        lines = [f"Fabric wisdom from {len(recent)} episode(s):\n"]
        for entry in recent:
            lines.append(f"{'─' * 60}")
            lines.append(f"[{entry.get('feed_title', '')}]  {entry.get('title', '')}")
            lines.append(f"Published: {entry.get('published', '')}\n")
            raw = entry.get("raw", {})
            text = raw.get("extract_wisdom") or raw.get("summarize") or next(iter(raw.values()), "")
            if text:
                lines.append(text[:1000])
            lines.append("")
        return "\n".join(lines)


# ── Utility ────────────────────────────────────────────────────────────────────

def _check_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
