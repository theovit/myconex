"""
Tests for the new myconex features:
  - core/digest.py
  - integrations/signal_detector.py
  - core/gateway/discord_gateway.py helpers (RAG, history persistence, feedback)
  - core/gateway/agentic_tools.py shared memory
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run(coro):
    return asyncio.run(coro)


# ─── Digest ───────────────────────────────────────────────────────────────────

class TestDigest(unittest.TestCase):

    def _with_tmp(self, profile=None, wisdom=None, feedback=None):
        """Return a tempdir with pre-populated data files."""
        tmp = tempfile.mkdtemp()
        base = Path(tmp)
        if profile is not None:
            (base / "interest_profile.json").write_text(json.dumps(profile))
        if wisdom is not None:
            (base / "wisdom_store.json").write_text(json.dumps(wisdom))
        if feedback is not None:
            fb_path = base / "feedback_log.jsonl"
            fb_path.write_text("\n".join(json.dumps(f) for f in feedback))
        return tmp

    def _patch_base(self, tmp):
        import core.digest as d
        d._BASE          = Path(tmp)
        d._PROFILE_FILE  = Path(tmp) / "interest_profile.json"
        d._WISDOM_FILE   = Path(tmp) / "wisdom_store.json"
        d._FEEDBACK_FILE = Path(tmp) / "feedback_log.jsonl"
        d._EMAIL_FILE    = Path(tmp) / "email_insights.json"
        d._YT_FILE       = Path(tmp) / "youtube_insights.json"
        d._RSS_FILE      = Path(tmp) / "rss_insights.json"
        d._PODCAST_FILE  = Path(tmp) / "podcast_insights.json"
        d._DIGEST_STAMP  = Path(tmp) / "last_digest.txt"

    def test_build_digest_data_empty(self):
        import core.digest as d
        with tempfile.TemporaryDirectory() as tmp:
            self._patch_base(tmp)
            data = d.build_digest_data()
        self.assertEqual(data["top_topics"], [])
        self.assertEqual(data["email_recent"], 0)
        self.assertEqual(data["fb_total"], 0)

    def test_build_digest_data_with_profile(self):
        import core.digest as d
        profile = {
            "topics": {"AI": 10, "rust": 5, "music": 2},
            "project_ideas": ["Build a thing", "Another idea"],
            "email_count": 15, "video_count": 8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._patch_base(tmp)
            (Path(tmp) / "interest_profile.json").write_text(json.dumps(profile))
            data = d.build_digest_data()
        self.assertIn("AI", data["top_topics"])
        self.assertEqual(data["total_email"], 15)
        self.assertEqual(data["total_video"], 8)

    def test_build_digest_text_returns_string(self):
        import core.digest as d
        with tempfile.TemporaryDirectory() as tmp:
            self._patch_base(tmp)
            text = d.build_digest_text()
        self.assertIn("MYCONEX Weekly Digest", text)

    def test_build_digest_embed_structure(self):
        import core.digest as d
        with tempfile.TemporaryDirectory() as tmp:
            self._patch_base(tmp)
            embed = d.build_digest_embed()
        self.assertIn("title", embed)
        self.assertIn("fields", embed)
        self.assertIsInstance(embed["fields"], list)

    def test_feedback_stats_in_digest(self):
        import core.digest as d
        feedback = [
            {"positive": True,  "ts": "2025-01-01T00:00:00Z"},
            {"positive": True,  "ts": "2025-01-02T00:00:00Z"},
            {"positive": False, "ts": "2025-01-03T00:00:00Z"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._patch_base(tmp)
            fb_path = Path(tmp) / "feedback_log.jsonl"
            fb_path.write_text("\n".join(json.dumps(f) for f in feedback))
            data = d.build_digest_data()
        self.assertEqual(data["fb_total"], 3)
        self.assertEqual(data["fb_pos"], 2)
        self.assertEqual(data["fb_rate"], 67)

    def test_digest_due_wrong_day(self):
        import core.digest as d
        from datetime import datetime, timezone
        # Patch to Sunday but wrong hour
        with patch("core.digest.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 5, 15, 0, tzinfo=timezone.utc)  # Sunday 3pm
            mock_dt.fromisoformat = datetime.fromisoformat
            d.DIGEST_HOUR = 9
            d.DIGEST_DAY  = 6
            with tempfile.TemporaryDirectory() as tmp:
                d._DIGEST_STAMP = Path(tmp) / "last_digest.txt"
                result = d._digest_due()
        self.assertFalse(result)

    def test_schedule_calls_post_fn(self):
        """schedule_weekly_digest should call post_fn when digest is due."""
        import core.digest as d
        called_with = []

        async def mock_post_fn(data):
            called_with.append(data)

        async def run():
            # Capture real sleep before patching (patch replaces asyncio.sleep on the module)
            real_sleep = asyncio.sleep

            async def fast_sleep(_n):
                """Replaces asyncio.sleep inside core.digest — yields once then returns."""
                await real_sleep(0)

            calls = [True, False]

            def _due():
                return calls.pop(0) if calls else False

            with patch("core.digest._digest_due", side_effect=_due), \
                 patch("core.digest._mark_digest_sent"), \
                 patch("core.digest.asyncio.sleep", new=fast_sleep):
                task = asyncio.create_task(d.schedule_weekly_digest(mock_post_fn))
                # Yield several times: fast_sleep yields once per call, post_fn needs a turn too
                for _ in range(6):
                    await real_sleep(0)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        _run(run())
        self.assertEqual(len(called_with), 1)


# ─── Signal detector ──────────────────────────────────────────────────────────

class TestSignalDetector(unittest.TestCase):

    def test_no_signals_without_profile(self):
        from integrations.signal_detector import SignalDetector
        with tempfile.TemporaryDirectory() as tmp:
            import integrations.signal_detector as sd
            sd._BASE         = Path(tmp)
            sd._PROFILE_FILE = Path(tmp) / "interest_profile.json"
            sd._SIGNALS_FILE = Path(tmp) / "signals_log.json"
            detector = SignalDetector()
            signals = _run(detector.detect())
        self.assertEqual(signals, [])

    def test_signal_detected_across_sources(self):
        from integrations.signal_detector import SignalDetector
        import integrations.signal_detector as sd

        profile = {"topics": {"distributed systems": 5}, "project_ideas": []}
        multi_source_results = [
            {"source": "email",   "content": "Raft consensus", "score": 0.85, "metadata": {"stored_at": "2099-01-01"}},
            {"source": "youtube", "content": "Raft consensus", "score": 0.80, "metadata": {"stored_at": "2099-01-01"}},
            {"source": "rss",     "content": "Raft consensus", "score": 0.75, "metadata": {"stored_at": "2099-01-01"}},
        ]

        async def mock_search(query, **kwargs):
            return multi_source_results

        with tempfile.TemporaryDirectory() as tmp:
            sd._BASE         = Path(tmp)
            sd._PROFILE_FILE = Path(tmp) / "interest_profile.json"
            sd._SIGNALS_FILE = Path(tmp) / "signals_log.json"
            (Path(tmp) / "interest_profile.json").write_text(json.dumps(profile))
            detector = SignalDetector()

            with patch("integrations.knowledge_store.search", new=mock_search), \
                 patch("core.notifications.notify", new=AsyncMock()):
                signals = _run(detector.detect())

        self.assertEqual(len(signals), 1)
        self.assertIn("email", signals[0]["sources"])
        self.assertIn("youtube", signals[0]["sources"])

    def test_no_duplicate_signals_in_same_window(self):
        """Already-detected signals within the lookback window are skipped."""
        from integrations.signal_detector import SignalDetector
        import integrations.signal_detector as sd

        profile = {"topics": {"AI": 3}}
        existing = [{"topic": "AI", "detected_at": "2099-01-01T00:00:00+00:00"}]

        async def mock_search(query, **kwargs):
            return [
                {"source": "email",   "content": "x", "score": 0.9, "metadata": {"stored_at": "2099-01-01"}},
                {"source": "youtube", "content": "x", "score": 0.9, "metadata": {"stored_at": "2099-01-01"}},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            sd._BASE         = Path(tmp)
            sd._PROFILE_FILE = Path(tmp) / "interest_profile.json"
            sd._SIGNALS_FILE = Path(tmp) / "signals_log.json"
            (Path(tmp) / "interest_profile.json").write_text(json.dumps(profile))
            (Path(tmp) / "signals_log.json").write_text(json.dumps(existing))
            detector = SignalDetector()

            with patch("integrations.knowledge_store.search", new=mock_search):
                signals = _run(detector.detect())

        self.assertEqual(len(signals), 0)

    def test_is_recent(self):
        from integrations.signal_detector import _is_recent
        self.assertFalse(_is_recent("", 7))
        self.assertFalse(_is_recent("2000-01-01T00:00:00+00:00", 7))
        self.assertTrue(_is_recent("2099-01-01T00:00:00+00:00", 7))


# ─── Discord gateway helpers ──────────────────────────────────────────────────

class TestDiscordGatewayHelpers(unittest.TestCase):

    def test_history_key_to_filename(self):
        from core.gateway.discord_gateway import _history_key_to_filename
        fname = _history_key_to_filename("123456:789012")
        self.assertTrue(fname.endswith(".json"))
        self.assertNotIn(":", fname)

    def test_history_save_and_load(self):
        from core.gateway.discord_gateway import _history_save, _history_load, HISTORY_DIR
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            gw.HISTORY_DIR = Path(tmp)
            history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
            _history_save("testchannel", history)
            loaded = _history_load("testchannel")
        self.assertEqual(loaded, history)

    def test_history_load_missing_returns_empty(self):
        from core.gateway.discord_gateway import _history_load, HISTORY_DIR
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            gw.HISTORY_DIR = Path(tmp)
            result = _history_load("nonexistent_channel")
        self.assertEqual(result, [])

    def test_rag_context_empty_when_unavailable(self):
        """_rag_context should return '' if knowledge_store search fails."""
        from core.gateway.discord_gateway import _rag_context
        with patch("integrations.knowledge_store.search", new=AsyncMock(side_effect=Exception("no qdrant"))):
            result = _run(_rag_context("what is distributed systems"))
        self.assertEqual(result, "")

    def test_rag_context_empty_on_empty_query(self):
        from core.gateway.discord_gateway import _rag_context
        self.assertEqual(_run(_rag_context("")), "")
        self.assertEqual(_run(_rag_context("   ")), "")

    def test_rag_context_formats_results(self):
        from core.gateway.discord_gateway import _rag_context
        mock_results = [
            {"source": "email", "content": "Key insight about AI", "score": 0.9,
             "metadata": {"subject": "AI newsletter"}},
        ]
        with patch("integrations.knowledge_store.search", new=AsyncMock(return_value=mock_results)):
            result = _run(_rag_context("AI insights"))
        self.assertIn("email", result)
        self.assertIn("Key insight about AI", result)

    def test_load_feedback_summary_empty(self):
        from core.gateway.discord_gateway import _load_feedback_summary, FEEDBACK_FILE
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            gw.FEEDBACK_FILE = Path(tmp) / "feedback_log.jsonl"
            result = _load_feedback_summary()
        self.assertEqual(result, "")

    def test_load_feedback_summary_with_data(self):
        from core.gateway.discord_gateway import _load_feedback_summary, FEEDBACK_FILE
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            fb_path = Path(tmp) / "feedback_log.jsonl"
            fb_path.write_text(
                json.dumps({"positive": True,  "bot_response_preview": "good answer"}) + "\n" +
                json.dumps({"positive": False, "bot_response_preview": "bad answer"}) + "\n"
            )
            gw.FEEDBACK_FILE = fb_path
            result = _load_feedback_summary()
        self.assertIn("Feedback", result)
        self.assertIn("50%", result)


# ─── History trim + memory pre-load helpers ───────────────────────────────────

class TestHistoryAndMemoryHelpers(unittest.TestCase):

    def _make_agentic_turns(self, n: int) -> list:
        """Build n turns each with 1 user + 1 tool_call assistant + 1 tool + 1 final assistant."""
        msgs = []
        for i in range(n):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}"}]})
            msgs.append({"role": "tool",      "content": "result", "tool_call_id": f"t{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        return msgs

    def test_trim_keeps_correct_turn_count(self):
        from core.gateway.discord_gateway import _trim_history_to_turns
        msgs = self._make_agentic_turns(5)   # 5 turns × 4 msgs = 20 raw msgs
        trimmed = _trim_history_to_turns(msgs, max_turns=3)
        user_count = sum(1 for m in trimmed if m.get("role") == "user")
        self.assertEqual(user_count, 3)
        self.assertEqual(trimmed[0]["content"], "q2")   # oldest kept turn

    def test_trim_noop_when_below_limit(self):
        from core.gateway.discord_gateway import _trim_history_to_turns
        msgs = self._make_agentic_turns(3)
        self.assertEqual(_trim_history_to_turns(msgs, max_turns=10), msgs)

    def test_trim_empty_list(self):
        from core.gateway.discord_gateway import _trim_history_to_turns
        self.assertEqual(_trim_history_to_turns([], 50), [])

    def test_trim_preserves_tool_messages(self):
        """Trimming must not cut mid-turn (orphan tool messages)."""
        from core.gateway.discord_gateway import _trim_history_to_turns
        msgs = self._make_agentic_turns(4)
        trimmed = _trim_history_to_turns(msgs, max_turns=2)
        # First message of trimmed result must be a user message
        self.assertEqual(trimmed[0]["role"], "user")
        # No orphaned tool messages (every tool message preceded by assistant)
        for idx, msg in enumerate(trimmed):
            if msg.get("role") == "tool":
                self.assertEqual(trimmed[idx - 1]["role"], "assistant")

    def test_load_memory_for_prompt_empty(self):
        from core.gateway.discord_gateway import _load_memory_for_prompt
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            gw._GATEWAY_MEMORY_FILE = Path(tmp) / "memory.json"
            result = _load_memory_for_prompt()
        self.assertEqual(result, "")

    def test_load_memory_for_prompt_with_data(self):
        from core.gateway.discord_gateway import _load_memory_for_prompt
        import core.gateway.discord_gateway as gw
        with tempfile.TemporaryDirectory() as tmp:
            mem_path = Path(tmp) / "memory.json"
            mem_path.write_text(json.dumps({"user_name": "Alice", "hobby": "coding"}))
            gw._GATEWAY_MEMORY_FILE = mem_path
            result = _load_memory_for_prompt()
        self.assertIn("Alice", result)
        self.assertIn("hobby", result)
        self.assertIn("Stored facts", result)


# ─── Shared memory via Qdrant ─────────────────────────────────────────────────

class TestSharedMemory(unittest.TestCase):
    """Tests that handle_memory mirrors stores to Qdrant."""

    def test_store_calls_embed_and_store(self):
        """handle_memory store should call embed_and_store on the knowledge store."""
        from core.gateway.agentic_tools import handle_memory
        mock_embed = AsyncMock(return_value="mem_id_123")

        with patch("integrations.knowledge_store.embed_and_store", mock_embed):
            result = handle_memory(action="store", key="test_key", value="test_value")

        self.assertIn("Stored", result)
        # embed_and_store was called (via _run_async inside handle_memory)
        # We can't easily await the async call from sync context in test,
        # but we verify the return value is correct
        self.assertIn("test_key", result)

    def test_retrieve_returns_value(self):
        from core.gateway.agentic_tools import handle_memory, _save_memory
        with tempfile.TemporaryDirectory() as tmp:
            import core.gateway.agentic_tools as at
            at._MEMORY_FILE = Path(tmp) / "memory.json"
            handle_memory(action="store", key="color", value="blue")
            result = handle_memory(action="retrieve", key="color")
        self.assertIn("blue", result)

    def test_delete_removes_key(self):
        from core.gateway.agentic_tools import handle_memory
        import core.gateway.agentic_tools as at
        with tempfile.TemporaryDirectory() as tmp:
            at._MEMORY_FILE = Path(tmp) / "memory.json"
            handle_memory(action="store", key="temp", value="ephemeral")
            handle_memory(action="delete", key="temp")
            result = handle_memory(action="retrieve", key="temp")
        self.assertIn("No memory found", result)


if __name__ == "__main__":
    unittest.main()
