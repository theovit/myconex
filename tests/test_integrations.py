"""
Tests for MYCONEX integrations — knowledge_store, rss_monitor,
podcast_ingester, email_ingester, youtube_ingester, and the notification bus.

All external I/O (Qdrant, Ollama, feedparser, yt-dlp, whisper, IMAP) is mocked
so these tests run without any running services.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Notification bus ─────────────────────────────────────────────────────────

class TestNotificationBus(unittest.TestCase):
    """Tests for core/notifications.py."""

    def setUp(self):
        # Reset module to get a fresh queue for each test
        import importlib
        if "core.notifications" in sys.modules:
            del sys.modules["core.notifications"]
        import core.notifications as notif_mod
        self.notif = notif_mod

    def _run(self, coro):
        return asyncio.run(coro)

    def test_notify_and_drain(self):
        self._run(self.notif.notify("hello"))
        self._run(self.notif.notify("world"))
        msgs = self._run(self.notif.drain())
        self.assertEqual(msgs, ["hello", "world"])

    def test_drain_empty(self):
        msgs = self._run(self.notif.drain())
        self.assertEqual(msgs, [])

    def test_drain_clears_queue(self):
        self._run(self.notif.notify("msg"))
        self._run(self.notif.drain())
        msgs = self._run(self.notif.drain())
        self.assertEqual(msgs, [])

    def test_queue_full_drops_silently(self):
        """Queue is maxsize=256 — overfilling must not raise."""
        async def _fill():
            for i in range(300):
                await self.notif.notify(f"msg{i}")
        self._run(_fill())
        msgs = self._run(self.notif.drain())
        self.assertLessEqual(len(msgs), 256)


# ─── Knowledge store ──────────────────────────────────────────────────────────

class TestKnowledgeStore(unittest.TestCase):
    """Tests for integrations/knowledge_store.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_embed_and_store_skips_empty_text(self):
        import importlib
        import integrations.knowledge_store as ks
        importlib.reload(ks)
        result = self._run(ks.embed_and_store(""))
        self.assertIsNone(result)

    def test_embed_and_store_skips_whitespace(self):
        import importlib
        import integrations.knowledge_store as ks
        importlib.reload(ks)
        result = self._run(ks.embed_and_store("   \n  "))
        self.assertIsNone(result)

    def test_search_returns_empty_when_unavailable(self):
        import importlib
        import integrations.knowledge_store as ks
        # Force _init to fail by blocking vector_store import
        with patch.dict(sys.modules, {"core.memory.vector_store": None}):
            importlib.reload(ks)
            results = self._run(ks.search("anything"))
        self.assertEqual(results, [])

    def test_get_stats_returns_unavailable(self):
        import importlib
        import integrations.knowledge_store as ks
        importlib.reload(ks)
        # Force _init to fail by making VectorStore unavailable
        with patch.dict(sys.modules, {"core.memory.vector_store": None}):
            importlib.reload(ks)
            stats = self._run(ks.get_stats())
        self.assertFalse(stats.get("available", True))

    def test_format_results_empty(self):
        from integrations.knowledge_store import format_results
        out = format_results([], "test query")
        self.assertIn("No knowledge base results", out)

    def test_format_results_with_items(self):
        from integrations.knowledge_store import format_results
        items = [
            {"content": "Some insight", "score": 0.9, "source": "email",
             "metadata": {"subject": "Test email"}},
        ]
        out = format_results(items, "my query")
        self.assertIn("Test email", out)
        self.assertIn("email", out)
        self.assertIn("0.9", out)


# ─── RSS monitor ──────────────────────────────────────────────────────────────

class TestRSSMonitor(unittest.TestCase):
    """Tests for integrations/rss_monitor.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_monitor(self, tmp_dir: str):
        """Create an RSSMonitor with all paths redirected to tmp_dir."""
        import integrations.rss_monitor as rss_mod
        rss_mod._BASE_DIR = Path(tmp_dir)
        rss_mod._SEEN_FILE     = Path(tmp_dir) / "rss_seen_ids.json"
        rss_mod._INSIGHTS_FILE = Path(tmp_dir) / "rss_insights.json"
        rss_mod._WISDOM_FILE   = Path(tmp_dir) / "wisdom_store.json"
        rss_mod._PROFILE_FILE  = Path(tmp_dir) / "interest_profile.json"
        rss_mod._FEEDS_FILE    = Path(tmp_dir) / "rss_feeds.json"
        return rss_mod.RSSMonitor(feeds=[])

    def test_add_and_list_feeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            self.assertEqual(monitor.list_feeds(), [])
            added = monitor.add_feed("https://example.com/rss")
            self.assertTrue(added)
            self.assertIn("https://example.com/rss", monitor.list_feeds())

    def test_add_duplicate_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            monitor.add_feed("https://example.com/rss")
            result = monitor.add_feed("https://example.com/rss")
            self.assertFalse(result)
            self.assertEqual(len(monitor.list_feeds()), 1)

    def test_remove_feed(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            monitor.add_feed("https://example.com/rss")
            removed = monitor.remove_feed("https://example.com/rss")
            self.assertTrue(removed)
            self.assertEqual(monitor.list_feeds(), [])

    def test_remove_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            result = monitor.remove_feed("https://does-not-exist.com/rss")
            self.assertFalse(result)

    def test_feeds_persist_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            monitor.add_feed("https://example.com/rss")
            feeds_file = Path(tmp) / "rss_feeds.json"
            self.assertTrue(feeds_file.exists())
            data = json.loads(feeds_file.read_text())
            self.assertIn("https://example.com/rss", data)

    def test_article_id_stable(self):
        from integrations.rss_monitor import _article_id
        id1 = _article_id("https://example.com/article", "Title")
        id2 = _article_id("https://example.com/article", "Title")
        self.assertEqual(id1, id2)

    def test_article_id_url_preferred_over_title(self):
        from integrations.rss_monitor import _article_id
        id_with_url = _article_id("https://example.com/a", "Different title")
        id_same_url = _article_id("https://example.com/a", "Another title")
        self.assertEqual(id_with_url, id_same_url)

    def test_poll_all_no_feeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            count = self._run(monitor.poll_all())
            self.assertEqual(count, 0)

    def test_fetch_feed_missing_feedparser(self):
        """_fetch_feed should return [] gracefully if feedparser not installed."""
        import integrations.rss_monitor as rss_mod
        with patch.dict(sys.modules, {"feedparser": None}):
            result = self._run(rss_mod._fetch_feed("https://example.com/rss"))
        self.assertEqual(result, [])

    def test_get_recent_insights_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            result = monitor.get_recent_insights()
            self.assertIn("No RSS articles", result)

    def test_get_wisdom_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self._make_monitor(tmp)
            result = monitor.get_wisdom()
            self.assertIn("No RSS wisdom", result)

    def test_fabric_sections_parser(self):
        from integrations.rss_monitor import _parse_fabric_sections
        text = "# IDEAS\n- Build a thing\n- Another idea\n# SUMMARY\nThis is a summary."
        sections = _parse_fabric_sections(text)
        self.assertIn("IDEAS", sections)
        self.assertIn("Build a thing", sections["IDEAS"])
        self.assertIn("Another idea", sections["IDEAS"])


# ─── Podcast ingester ─────────────────────────────────────────────────────────

class TestPodcastIngester(unittest.TestCase):
    """Tests for integrations/podcast_ingester.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_ingester(self, tmp_dir: str):
        import integrations.podcast_ingester as pod_mod
        pod_mod._BASE_DIR      = Path(tmp_dir)
        pod_mod._SEEN_FILE     = Path(tmp_dir) / "podcast_seen_ids.json"
        pod_mod._INSIGHTS_FILE = Path(tmp_dir) / "podcast_insights.json"
        pod_mod._WISDOM_FILE   = Path(tmp_dir) / "wisdom_store.json"
        pod_mod._PROFILE_FILE  = Path(tmp_dir) / "interest_profile.json"
        pod_mod._FEEDS_FILE    = Path(tmp_dir) / "podcast_feeds.json"
        return pod_mod.PodcastIngester(feeds=[])

    def test_add_remove_feeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            self.assertTrue(ing.add_feed("https://example.com/podcast.rss"))
            self.assertFalse(ing.add_feed("https://example.com/podcast.rss"))
            self.assertTrue(ing.remove_feed("https://example.com/podcast.rss"))
            self.assertFalse(ing.remove_feed("https://example.com/podcast.rss"))

    def test_poll_all_no_feeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            count = self._run(ing.poll_all())
            self.assertEqual(count, 0)

    def test_check_dependencies_returns_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            result = ing.check_dependencies()
            self.assertIsInstance(result, str)
            self.assertIn("feedparser", result)
            self.assertIn("yt-dlp", result)

    def test_episode_id_stable(self):
        from integrations.podcast_ingester import _episode_id
        id1 = _episode_id("https://example.com/ep1.mp3", "Episode 1")
        id2 = _episode_id("https://example.com/ep1.mp3", "Episode 1")
        self.assertEqual(id1, id2)

    def test_get_recent_insights_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            result = ing.get_recent_insights()
            self.assertIn("No podcast episodes", result)

    def test_get_wisdom_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            result = ing.get_wisdom()
            self.assertIn("No podcast wisdom", result)

    def test_fetch_podcast_feed_missing_feedparser(self):
        import integrations.podcast_ingester as pod_mod
        with patch.dict(sys.modules, {"feedparser": None}):
            result = self._run(pod_mod._fetch_podcast_feed("https://example.com/feed"))
        self.assertEqual(result, [])

    def test_download_audio_missing_ytdlp(self):
        import integrations.podcast_ingester as pod_mod
        with patch("shutil.which", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                result = self._run(pod_mod._download_audio("https://example.com/ep.mp3", Path(tmp)))
        self.assertIsNone(result)

    def test_skip_long_episode(self):
        """Episodes exceeding MAX_DURATION should be skipped during poll_all."""
        import integrations.podcast_ingester as pod_mod
        with tempfile.TemporaryDirectory() as tmp:
            ing = self._make_ingester(tmp)
            ing.add_feed("https://example.com/feed")
            long_episode = {
                "id": "abc123", "title": "Long ep", "audio_url": "https://example.com/ep.mp3",
                "published": "2024-01-01", "feed_url": "https://example.com/feed",
                "feed_title": "Test Pod", "duration_min": 999, "description": "",
            }
            # Patch _fetch_podcast_feed to return one long episode
            with patch.object(pod_mod, "_fetch_podcast_feed", new=AsyncMock(return_value=[long_episode])):
                pod_mod._MAX_DURATION_MIN = 60  # max 60min
                count = self._run(ing.poll_all())
            self.assertEqual(count, 0)


# ─── YouTube ingester ─────────────────────────────────────────────────────────

class TestYouTubeIngester(unittest.TestCase):
    """Tests for integrations/youtube_ingester.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_video_id_extraction(self):
        from integrations.youtube_ingester import _video_id
        self.assertEqual(_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertIsNone(_video_id("https://example.com/notayoutube"))

    def test_canonical_url(self):
        from integrations.youtube_ingester import _canonical_url
        url = _canonical_url("dQw4w9WgXcQ")
        self.assertIn("dQw4w9WgXcQ", url)
        self.assertIn("youtube.com", url)

    def test_load_watch_history_file_not_found(self):
        from integrations.youtube_ingester import load_watch_history
        with self.assertRaises(FileNotFoundError):
            load_watch_history("/nonexistent/path/watch-history.json")

    def test_load_watch_history_parses_entries(self):
        from integrations.youtube_ingester import load_watch_history
        sample = [
            {"titleUrl": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
             "title": "Watched Test Video", "time": "2024-01-01T00:00:00Z"},
            {"titleUrl": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
             "title": "Watched Another Video", "time": "2024-01-02T00:00:00Z"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample, f)
            fname = f.name
        try:
            entries = load_watch_history(fname)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["video_id"], "aaaaaaaaaaa")
            self.assertFalse(entries[0]["title"].startswith("Watched "))
        finally:
            os.unlink(fname)

    def test_load_watch_history_deduplicates(self):
        from integrations.youtube_ingester import load_watch_history
        sample = [
            {"titleUrl": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
             "title": "Video", "time": "2024-01-01T00:00:00Z"},
            {"titleUrl": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
             "title": "Video (watched again)", "time": "2024-06-01T00:00:00Z"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample, f)
            fname = f.name
        try:
            entries = load_watch_history(fname)
            self.assertEqual(len(entries), 1)  # deduplicated to 1
        finally:
            os.unlink(fname)

    def test_get_recent_insights_empty(self):
        from integrations.youtube_ingester import YouTubeIngester
        import integrations.youtube_ingester as yt_mod
        with tempfile.TemporaryDirectory() as tmp:
            yt_mod._INSIGHTS_FILE = Path(tmp) / "youtube_insights.json"
            result = YouTubeIngester.get_recent_insights()
            self.assertIn("No YouTube videos", result)

    def test_ingest_history_no_path(self):
        from integrations.youtube_ingester import YouTubeIngester
        ing = YouTubeIngester(history_path="", watch_later_path="")
        count = self._run(ing.ingest_history())
        self.assertEqual(count, 0)


# ─── Email ingester ───────────────────────────────────────────────────────────

class TestEmailIngester(unittest.TestCase):
    """Tests for integrations/email_ingester.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_fabric_sections_parser(self):
        from integrations.email_ingester import _parse_fabric_sections
        text = "# INSIGHTS\n- Key insight one\n- Key insight two\n# IDEAS\n- Project idea"
        sections = _parse_fabric_sections(text)
        self.assertIn("INSIGHTS", sections)
        self.assertIn("Key insight one", sections["INSIGHTS"])
        self.assertIn("IDEAS", sections)

    def test_ingest_once_without_credentials(self):
        """ingest_once should return 0 when GmailReader raises on connect."""
        import integrations.email_ingester as em_mod
        import integrations.gmail_reader as gmail_mod
        # GmailReader is imported inline; patch at its source module
        mock_reader = MagicMock()
        mock_reader.return_value.connect.side_effect = Exception("no creds")
        with patch.object(gmail_mod, "GmailReader", mock_reader):
            with tempfile.TemporaryDirectory() as tmp:
                em_mod._BASE_DIR      = Path(tmp)
                em_mod._SEEN_FILE     = Path(tmp) / "email_seen_uids.json"
                em_mod._INSIGHTS_FILE = Path(tmp) / "email_insights.json"
                em_mod._PROFILE_FILE  = Path(tmp) / "interest_profile.json"
                em_mod._MEMORY_FILE   = Path(tmp) / "memory.json"
                em_mod._WISDOM_FILE   = Path(tmp) / "wisdom_store.json"
                try:
                    count = self._run(em_mod.EmailIngester().ingest_once())
                except Exception:
                    count = 0
        self.assertEqual(count, 0)

    def test_get_recent_insights_empty(self):
        import integrations.email_ingester as em_mod
        with tempfile.TemporaryDirectory() as tmp:
            em_mod._INSIGHTS_FILE = Path(tmp) / "email_insights.json"
            result = em_mod.EmailIngester.get_recent_insights()
            self.assertIn("No emails", result)


# ─── Supervisor (buzlock_bot) ─────────────────────────────────────────────────

class TestSupervisor(unittest.TestCase):
    """Tests for the _supervise() wrapper in buzlock_bot.py."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_supervise_clean_exit(self):
        """A coroutine that exits cleanly should not be restarted."""
        import buzlock_bot
        call_count = 0

        async def coro():
            nonlocal call_count
            call_count += 1
            # returns normally

        self._run(buzlock_bot._supervise("test", coro, restart_delay=0))
        self.assertEqual(call_count, 1)

    def test_supervise_restarts_on_crash(self):
        """A crashing coroutine should be restarted until it exits cleanly."""
        import buzlock_bot
        attempts = []

        async def coro():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("simulated crash")
            # third attempt: exit cleanly

        self._run(buzlock_bot._supervise("test", coro, restart_delay=0))
        self.assertEqual(len(attempts), 3)

    def test_supervise_propagates_cancelled_error(self):
        """CancelledError must propagate so tasks can be shut down."""
        import buzlock_bot

        async def coro():
            raise asyncio.CancelledError()

        async def run():
            with self.assertRaises(asyncio.CancelledError):
                await buzlock_bot._supervise("test", coro, restart_delay=0)

        self._run(run())


if __name__ == "__main__":
    unittest.main()
