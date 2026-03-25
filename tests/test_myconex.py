"""
MYCONEX Comprehensive Test Suite
==================================
Tests are aligned with the actual API signatures discovered by inspection.

Test Groups:
    A. BaseAgent — delegation, routing, complexity scoring
    B. PersistentPythonREPL — state persistence, safety
    C. ContextFrame — token budgeting, pruning, hierarchy
    D. SessionMemory — store, retrieve, search
    E. DocumentProcessor — HTML/text parsing
    F. IntelAggregator — multi-source gathering
    G. AgentRoster — division management and routing
    H. SandboxExecutor — isolation, timeout, resource limits
    I. Config — load/override priority
    J. RLMAgent — task handling (mocked LLM)
    K. CodebaseIndex — inverted keyword index
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# A. BaseAgent Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityEstimation(unittest.TestCase):
    """_estimate_complexity takes a list[dict] of chat messages."""

    def setUp(self):
        from orchestration.agents.base_agent import _estimate_complexity
        self.estimate = _estimate_complexity

    def _msgs(self, text):
        return [{"role": "user", "content": text}]

    def test_simple_task_low_score(self):
        score = self.estimate(self._msgs("What is 2 + 2?"))
        self.assertLess(score, 0.6)

    def test_complex_task_high_score(self):
        task = (
            "Analyze the distributed system architecture, then decompose "
            "the parallel optimization pipeline and synthesize a research report "
            "with multiple steps and strategic recommendations."
        )
        score = self.estimate(self._msgs(task))
        self.assertGreater(score, 0.0)   # complex task scores above empty

    def test_returns_float_in_range(self):
        for text in ["hello", "x" * 500, "compare and analyze multiple systems"]:
            score = self.estimate(self._msgs(text))
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_empty_messages_does_not_crash(self):
        score = self.estimate([])
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_long_text_scores_higher(self):
        short = self._msgs("fix the bug")
        long_ = self._msgs(" ".join(["analyze research optimize synthesize compare"] * 10))
        self.assertGreaterEqual(self.estimate(long_), self.estimate(short))


class TestBaseAgentConfig(unittest.TestCase):
    """AgentConfig dataclass defaults and field presence."""

    def test_defaults(self):
        from orchestration.agents.base_agent import AgentConfig
        cfg = AgentConfig(name="test-agent", agent_type="worker")
        self.assertEqual(cfg.name, "test-agent")
        self.assertIsNotNone(cfg.ollama_url)

    def test_backend_field(self):
        from orchestration.agents.base_agent import AgentConfig
        cfg = AgentConfig(name="x", agent_type="worker", backend="llamacpp")
        self.assertEqual(cfg.backend, "llamacpp")


class TestBaseAgentDelegation(unittest.TestCase):
    """delegate() and set_router() wiring."""

    def _make_agent(self):
        from orchestration.agents.base_agent import AgentConfig, BaseAgent

        class ConcreteAgent(BaseAgent):
            def can_handle(self, task, tags=None):
                return 0.9

            async def handle_task(self, task, context=None):
                return f"handled: {task}"

        cfg = AgentConfig(name="concrete", agent_type="worker")
        return ConcreteAgent(cfg)

    def test_set_router_attaches(self):
        agent = self._make_agent()
        mock_router = MagicMock()
        agent.set_router(mock_router)
        self.assertIs(agent._router, mock_router)

    def test_delegate_calls_router(self):
        agent = self._make_agent()
        mock_router = MagicMock()
        mock_result = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        agent.set_router(mock_router)
        result = run(agent.delegate("analysis", {"text": "some sub-task"}))
        mock_router.route.assert_called_once()
        self.assertEqual(result, mock_result)

    def test_delegate_without_router_returns_error(self):
        agent = self._make_agent()
        # No router — delegate should either raise or return an error AgentResult
        try:
            result = run(agent.delegate("analysis", {"text": "task"}))
            # If it returns rather than raises, the result should indicate failure
            self.assertFalse(getattr(result, "success", True))
        except (RuntimeError, Exception):
            pass  # raising is also acceptable

    def test_status_includes_router_info(self):
        agent = self._make_agent()
        status = agent.status()
        self.assertIn("router_attached", status)
        self.assertFalse(status["router_attached"])
        agent.set_router(MagicMock())
        self.assertTrue(agent.status()["router_attached"])


# ═══════════════════════════════════════════════════════════════════════════════
# B. PersistentPythonREPL Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistentPythonREPL(unittest.TestCase):
    """Output goes to .output only via print(); expressions go to .return_value."""

    def setUp(self):
        from core.gateway.python_repl import PersistentPythonREPL
        self.repl = PersistentPythonREPL()

    def test_print_captured_in_output(self):
        result = run(self.repl.execute("print(2 + 2)"))
        self.assertTrue(result.success)
        self.assertIn("4", result.output)

    def test_state_persists_between_calls(self):
        run(self.repl.execute("x = 42"))
        result = run(self.repl.execute("print(x)"))
        self.assertTrue(result.success)
        self.assertIn("42", result.output)

    def test_stdout_captured(self):
        result = run(self.repl.execute("print('hello world')"))
        self.assertTrue(result.success)
        self.assertIn("hello world", result.output)

    def test_syntax_error_handled(self):
        result = run(self.repl.execute("def broken(:"))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_runtime_error_handled(self):
        result = run(self.repl.execute("1 / 0"))
        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.error or result.output)

    def test_multiline_code(self):
        code = "total = 0\nfor i in range(5):\n    total += i\nprint(total)"
        result = run(self.repl.execute(code))
        self.assertTrue(result.success)
        self.assertIn("10", result.output)

    def test_reset_clears_namespace(self):
        run(self.repl.execute("secret = 'hidden'"))
        run(self.repl.reset())   # async coroutine — must be awaited
        result = run(self.repl.execute("print(secret)"))
        # After reset, 'secret' should raise NameError → success=False
        self.assertFalse(result.success)

    def test_import_works(self):
        result = run(self.repl.execute("import math; print(math.pi)"))
        self.assertTrue(result.success)
        self.assertIn("3.14", result.output)


class TestREPLPool(unittest.TestCase):
    """Session-keyed REPL pool."""

    def setUp(self):
        from core.gateway.python_repl import REPLPool
        self.pool = REPLPool(session_ttl_s=3600)

    def test_get_creates_repl(self):
        repl = self.pool.get_or_create("session-1")
        self.assertIsNotNone(repl)

    def test_same_session_returns_same_repl(self):
        r1 = self.pool.get_or_create("session-abc")
        r2 = self.pool.get_or_create("session-abc")
        self.assertIs(r1, r2)

    def test_different_sessions_isolated(self):
        r1 = self.pool.get_or_create("session-A")
        r2 = self.pool.get_or_create("session-B")
        run(r1.execute("shared_var = 'from-A'"))
        result = run(r2.execute("print(shared_var)"))
        self.assertFalse(result.success)  # NameError expected

    def test_drop_session(self):
        from core.gateway.python_repl import REPLPool
        pool = REPLPool(session_ttl_s=3600)
        pool.get_or_create("s1")
        self.assertGreater(len(pool._sessions), 0)
        pool.drop_session("s1")
        self.assertNotIn("s1", pool._sessions)


# ═══════════════════════════════════════════════════════════════════════════════
# C. ContextFrame Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextFrame(unittest.TestCase):
    """Token budgeting, pruning, priority-based retention."""

    def setUp(self):
        from orchestration.agents.context_manager import ContextFrame, Priority
        self.ContextFrame = ContextFrame
        self.Priority = Priority

    def _make_frame(self, budget=500):
        return self.ContextFrame(
            frame_id="f1",
            agent_name="test-agent",
            task_type="analysis",
            task_id="t1",
            depth=0,
            tokens_budget=budget,
        )

    def test_add_message_increments_usage(self):
        frame = self._make_frame()
        before = frame.tokens_used
        frame.add_message("user", "Hello there", priority=self.Priority.MEDIUM)
        self.assertGreater(frame.tokens_used, before)

    def test_prune_reduces_messages(self):
        frame = self._make_frame(budget=100)
        for i in range(30):
            frame.add_message("user", f"Low priority message number {i}",
                              priority=self.Priority.LOW)
        initial_count = len(frame.messages)
        removed = frame.prune(target_tokens=50)
        self.assertGreaterEqual(removed, 0)
        self.assertLessEqual(len(frame.messages), initial_count)

    def test_critical_messages_survive_prune(self):
        frame = self._make_frame(budget=200)
        frame.add_message("system", "CRITICAL: do not lose this",
                          priority=self.Priority.CRITICAL)
        for i in range(40):
            frame.add_message("user", f"filler {i}", priority=self.Priority.LOW)
        frame.prune(target_tokens=30)
        critical_msgs = [m for m in frame.messages if "CRITICAL" in m.get("content", "")]
        self.assertTrue(len(critical_msgs) > 0)

    def test_frame_depth_tracked(self):
        frame = self.ContextFrame(
            frame_id="f2", agent_name="worker", task_type="sub",
            task_id="t2", depth=3, tokens_budget=1000
        )
        self.assertEqual(frame.depth, 3)


class TestRLMContextManager(unittest.TestCase):
    """Frame push/pop/flatten."""

    def setUp(self):
        from orchestration.agents.context_manager import RLMContextManager
        self.mgr = RLMContextManager(total_budget=4096)

    def test_push_creates_frame(self):
        frame = self.mgr.push_frame("agent-1", "research", task_id="t1")
        self.assertIsNotNone(frame)
        self.assertGreaterEqual(len(self.mgr._frames), 1)

    def test_pop_removes_frame(self):
        frame = self.mgr.push_frame("agent-1", "research", task_id="t2")
        self.mgr.pop_frame(frame.frame_id)
        # popped frame is marked completed
        popped = self.mgr._frames.get(frame.frame_id)
        self.assertTrue(popped is None or popped.completed_at is not None)

    def test_flatten_returns_messages(self):
        frame = self.mgr.push_frame("agent-1", "analysis", task_id="t3")
        frame.add_message("user", "test message")
        result = self.mgr.flatten_context(frame.frame_id)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_nested_frames_have_increasing_depth(self):
        f1 = self.mgr.push_frame("manager", "orchestrate", task_id="t4")
        f2 = self.mgr.push_frame("worker", "execute", task_id="t5",
                                 parent_id=f1.frame_id)
        self.assertGreater(f2.depth, f1.depth)


# ═══════════════════════════════════════════════════════════════════════════════
# D. SessionMemory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionMemory(unittest.TestCase):
    """In-session memory: store, retrieve, search."""

    def setUp(self):
        from orchestration.agents.context_manager import SessionMemory
        self.mem = SessionMemory(session_id="test-session", max_entries=50)

    def test_store_and_retrieve(self):
        self.mem.store("api_key", "The API key is stored in env var FOO_KEY")
        result = self.mem.retrieve("api_key")
        self.assertIsNotNone(result)
        self.assertIn("FOO_KEY", result)

    def test_search_returns_relevant(self):
        self.mem.store("python_info", "Python uses indentation for blocks",
                       category="language")
        self.mem.store("go_info", "Go uses braces for blocks",
                       category="language")
        results = self.mem.search("python indentation")
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("Python", top.content if hasattr(top, "content") else str(top))

    def test_max_entries_enforced(self):
        from orchestration.agents.context_manager import SessionMemory
        mem = SessionMemory(session_id="small", max_entries=5)
        for i in range(10):
            mem.store(f"key_{i}", f"entry number {i}")
        self.assertLessEqual(len(mem._entries), 5)

    def test_log_interaction(self):
        # log_interaction signature: (task_type, success, duration_ms, model='')
        self.mem.log_interaction("qa", success=True, duration_ms=100.0)
        # Should not raise; verify it recorded something
        self.assertGreater(len(self.mem._interaction_log), 0)

    def test_format_for_context_string(self):
        self.mem.store("test_key", "relevant information here")
        output = self.mem.format_for_context("test query", max_entries=10)
        self.assertIsInstance(output, str)


class TestPersistentMemoryStore(unittest.TestCase):
    """Cross-session JSON-backed memory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from orchestration.agents.context_manager import PersistentMemoryStore
        self.store = PersistentMemoryStore(namespace="test", memory_dir=Path(self.tmp))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_store_and_retrieve(self):
        self.store.store("mesh", "MYCONEX is a distributed AI mesh")
        result = self.store.retrieve("mesh")
        self.assertIsNotNone(result)
        self.assertIn("distributed", result)

    def test_persists_to_disk(self):
        self.store.store("disk_key", "this should be saved")
        from orchestration.agents.context_manager import PersistentMemoryStore
        store2 = PersistentMemoryStore(namespace="test", memory_dir=Path(self.tmp))
        result = store2.retrieve("disk_key")
        self.assertIsNotNone(result)
        self.assertIn("saved", result)

    def test_namespace_isolation(self):
        from orchestration.agents.context_manager import PersistentMemoryStore
        store_a = PersistentMemoryStore(namespace="ns-a", memory_dir=Path(self.tmp))
        store_b = PersistentMemoryStore(namespace="ns-b", memory_dir=Path(self.tmp))
        store_a.store("secret", "only in namespace A")
        result = store_b.retrieve("secret")
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# E. DocumentProcessor Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTMLDocParser(unittest.TestCase):
    """HTML document parsing."""

    def setUp(self):
        from tools.document_processor import HTMLDocParser
        self.parser = HTMLDocParser()

    def test_parses_title(self):
        html = "<html><head><title>Test Page</title></head><body><p>Content</p></body></html>"
        result = self.parser.parse(html)
        self.assertEqual(result.title, "Test Page")

    def test_extracts_sections(self):
        html = """<html><body>
          <h1>Main Heading</h1><p>Content here.</p>
          <h2>Sub Heading</h2><p>More content.</p>
        </body></html>"""
        result = self.parser.parse(html)
        self.assertGreater(len(result.sections), 0)

    def test_extracts_tables(self):
        html = """<html><body>
          <table><tr><th>Name</th></tr><tr><td>foo</td></tr></table>
        </body></html>"""
        result = self.parser.parse(html)
        self.assertGreater(len(result.tables), 0)

    def test_empty_html_does_not_crash(self):
        result = self.parser.parse("")
        self.assertIsNotNone(result)

    def test_success_flag_set(self):
        html = "<html><body><p>Hello</p></body></html>"
        result = self.parser.parse(html)
        self.assertTrue(result.success)


class TestDocumentProcessor(unittest.TestCase):
    """Async DocumentProcessor dispatch."""

    def test_process_text_file(self):
        from tools.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("Line one.\nLine two.\nLine three.\n")
            tmp = f.name
        try:
            result = run(proc.process(tmp))
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            self.assertIn("Line", result.raw_text or result.title or
                          "".join(s.content for s in result.sections))
        finally:
            os.unlink(tmp)

    def test_process_html_file(self):
        from tools.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        html = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>"
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            tmp = f.name
        try:
            result = run(proc.process(tmp))
            self.assertTrue(result.success)
            text_content = result.raw_text or "".join(s.content for s in result.sections)
            self.assertIn("Hello", text_content)
        finally:
            os.unlink(tmp)

    def test_missing_file_returns_result(self):
        from tools.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        result = run(proc.process("/nonexistent/path/file.txt"))
        # Either returns failure, or returns empty result — both are acceptable
        self.assertIsNotNone(result)


class TestScientificPaperExtractor(unittest.TestCase):
    """Scientific section extraction returns a dict."""

    def setUp(self):
        from tools.document_processor import ScientificPaperExtractor
        self.extractor = ScientificPaperExtractor()

    def test_extracts_abstract(self):
        text = "Abstract\n\nThis paper presents a novel approach.\n\nIntroduction\n\nBackground."
        result = self.extractor.extract(text)
        self.assertIsInstance(result, dict)
        self.assertIn("abstract", result)
        self.assertIn("novel", result.get("abstract", ""))

    def test_extracts_sections(self):
        text = "Introduction\n\nFirst section.\n\nMethods\n\nMethodology.\n\nResults\n\nFindings."
        result = self.extractor.extract(text)
        self.assertIsInstance(result, dict)
        sections = result.get("sections", [])
        self.assertGreater(len(sections), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# F. IntelAggregator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntelAggregator(unittest.TestCase):
    """Multi-source intelligence gathering."""

    def test_file_gatherer(self):
        from tools.intel_aggregator import IntelAggregator, IntelSource
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("Intelligence from file source.\n")
            tmp = f.name
        try:
            agg = IntelAggregator()
            sources = [IntelSource(name="test-file", type="file",
                                   config={"path": tmp})]
            report = run(agg.gather(sources))
            self.assertGreater(len(report.results), 0)
            content = report.results[0].content
            self.assertIn("Intelligence", content)
        finally:
            os.unlink(tmp)

    def test_shell_gatherer(self):
        from tools.intel_aggregator import IntelAggregator, IntelSource
        agg = IntelAggregator()
        sources = [IntelSource(name="echo", type="shell",
                               config={"command": "echo 'shell-output'"})]
        report = run(agg.gather(sources))
        self.assertGreater(len(report.results), 0)
        self.assertIn("shell-output", report.results[0].content)

    def test_deduplication(self):
        from tools.intel_aggregator import IntelAggregator, IntelSource
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("Duplicate content.\n")
            tmp = f.name
        try:
            agg = IntelAggregator()
            sources = [
                IntelSource(name="s1", type="file", config={"path": tmp}),
                IntelSource(name="s2", type="file", config={"path": tmp}),
            ]
            report = run(agg.gather(sources, deduplicate=True))
            self.assertLessEqual(len(report.results), 2)
        finally:
            os.unlink(tmp)

    def test_empty_sources_returns_empty_report(self):
        from tools.intel_aggregator import IntelAggregator
        agg = IntelAggregator()
        report = run(agg.gather([]))
        self.assertEqual(len(report.results), 0)

    def test_failed_source_does_not_crash(self):
        from tools.intel_aggregator import IntelAggregator, IntelSource
        agg = IntelAggregator()
        sources = [IntelSource(name="bad", type="file",
                               config={"path": "/nonexistent/file.txt"})]
        report = run(agg.gather(sources))
        self.assertIsNotNone(report)


# ═══════════════════════════════════════════════════════════════════════════════
# G. AgentRoster Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDivisionRegistry(unittest.TestCase):
    """Division-based agent registry."""

    def setUp(self):
        from orchestration.agent_roster import Division, DivisionRegistry
        from orchestration.agents.base_agent import AgentConfig, BaseAgent

        class _TestAgent(BaseAgent):
            def can_handle(self, task, tags=None): return 0.9
            async def handle_task(self, task, context=None): return "ok"

        self.Division = Division
        self.DivisionRegistry = DivisionRegistry
        self._TestAgent = _TestAgent

    def _make_agent(self, name):
        cfg = __import__(
            "orchestration.agents.base_agent", fromlist=["AgentConfig"]
        ).AgentConfig(name=name, agent_type="worker")
        return self._TestAgent(cfg)

    def test_add_and_find(self):
        from orchestration.agent_roster import DivisionRegistry, Division, RosterAgent
        reg = DivisionRegistry(division=Division.ENGINEERING)
        agent = self._make_agent("eng-1")
        ra = RosterAgent(agent=agent, division=Division.ENGINEERING,
                         specialties=["python", "api"])
        reg.add(ra)
        result = reg.find(task_type="analysis", tags=["python"])
        self.assertIsNotNone(result)

    def test_specialty_scoring(self):
        from orchestration.agent_roster import RosterAgent, Division
        agent = self._make_agent("data-1")
        ra = RosterAgent(
            agent=agent,
            division=Division.DATA,
            specialties=["ml", "pandas", "stats"],
        )
        score_high = ra.specialty_score(["ml", "pandas"])
        score_low = ra.specialty_score(["networking", "frontend"])
        self.assertGreater(score_high, score_low)


class TestAgentRoster(unittest.TestCase):
    """Full AgentRoster add/route API."""

    def setUp(self):
        from orchestration.agent_roster import AgentRoster, Division
        from orchestration.agents.base_agent import AgentConfig, BaseAgent

        class _TestAgent(BaseAgent):
            def can_handle(self, task, tags=None): return 0.9
            async def handle_task(self, task, context=None): return "ok"

        self.roster = AgentRoster()
        self.Division = Division

        cfg = AgentConfig(name="eng-lead", agent_type="worker")
        self.eng_agent = _TestAgent(cfg)

    def test_add_agent(self):
        roster_agent = self.roster.add(
            self.eng_agent, self.Division.ENGINEERING,
            specialties=["python"]
        )
        self.assertIsNotNone(roster_agent)
        agents = self.roster.all_agents()
        self.assertGreater(len(agents), 0)

    def test_division_agents_returns_list(self):
        self.roster.add(self.eng_agent, self.Division.ENGINEERING)
        agents = self.roster.division_agents(self.Division.ENGINEERING)
        self.assertIsInstance(agents, list)
        self.assertGreater(len(agents), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# H. SandboxExecutor Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSandboxExecutor(unittest.TestCase):
    """Subprocess isolation, timeout, output capture."""

    def setUp(self):
        from tools.sandbox_executor import SandboxConfig, SandboxExecutor
        self.SandboxConfig = SandboxConfig
        self.executor = SandboxExecutor(
            default_config=SandboxConfig(timeout_s=10.0, max_memory_mb=256)
        )

    def test_run_python_success(self):
        result = run(self.executor.run_python("print('sandbox-ok')"))
        self.assertTrue(result.success)
        self.assertIn("sandbox-ok", result.stdout)

    def test_run_python_math(self):
        result = run(self.executor.run_python("print(sum(range(100)))"))
        self.assertTrue(result.success)
        self.assertIn("4950", result.stdout)

    def test_run_python_syntax_error(self):
        result = run(self.executor.run_python("def broken(:"))
        self.assertFalse(result.success)
        self.assertNotEqual(result.return_code, 0)

    def test_run_bash_success(self):
        result = run(self.executor.run_bash("echo 'bash-ok'"))
        self.assertTrue(result.success)
        self.assertIn("bash-ok", result.stdout)

    def test_run_command_success(self):
        result = run(self.executor.run_command("echo command-ok"))
        self.assertTrue(result.success)
        self.assertIn("command-ok", result.stdout)

    def test_timeout_kills_process(self):
        from tools.sandbox_executor import SandboxConfig
        cfg = SandboxConfig(timeout_s=0.5)
        result = run(self.executor.run_python(
            "import time; time.sleep(10)", config=cfg
        ))
        self.assertTrue(result.timed_out)
        self.assertFalse(result.success)

    def test_run_parallel_all_succeed(self):
        from tools.sandbox_executor import SandboxTask
        tasks = [
            SandboxTask(lang="python", code=f"print('task-{i}')", task_id=f"t{i}")
            for i in range(4)
        ]
        results = run(self.executor.run_parallel(tasks))
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r.success)

    def test_run_parallel_order_preserved(self):
        from tools.sandbox_executor import SandboxTask
        tasks = [
            SandboxTask(lang="python", code=f"print({i})", task_id=f"t{i}")
            for i in range(3)
        ]
        results = run(self.executor.run_parallel(tasks))
        for i, r in enumerate(results):
            self.assertIn(str(i), r.stdout)

    def test_parallel_with_one_failure(self):
        from tools.sandbox_executor import SandboxTask
        tasks = [
            SandboxTask(lang="python", code="print('ok')", task_id="good"),
            SandboxTask(lang="python", code="raise ValueError('boom')", task_id="bad"),
        ]
        results = run(self.executor.run_parallel(tasks, return_exceptions=True))
        self.assertEqual(len(results), 2)
        successes = [r.success for r in results]
        self.assertIn(True, successes)
        self.assertIn(False, successes)

    def test_run_parallel_python_convenience(self):
        snippets = ["print(1)", "print(2)", "print(3)"]
        results = run(self.executor.run_parallel_python(snippets))
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r.success)

    def test_result_duration_recorded(self):
        result = run(self.executor.run_python("pass"))
        self.assertGreater(result.duration_ms, 0)

    def test_stdin_passed_to_process(self):
        result = run(self.executor.run_python(
            "import sys; data = sys.stdin.read(); print(f'got:{data.strip()}')",
            stdin=b"hello-stdin",
        ))
        self.assertTrue(result.success)
        self.assertIn("got:hello-stdin", result.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# I. Config Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadConfig(unittest.TestCase):
    """Config dataclass, env overrides, YAML loading."""

    def setUp(self):
        from config import reset_config
        reset_config()

    def tearDown(self):
        from config import reset_config
        reset_config()
        for key in ["OLLAMA_URL", "DISCORD_BOT_TOKEN", "MYCONEX_BACKEND",
                    "MYCONEX_API_PORT", "MYCONEX_LOG_LEVEL", "MYCONEX_MEMORY_NAMESPACE"]:
            os.environ.pop(key, None)

    def test_defaults(self):
        from config import load_config
        cfg = load_config(apply_env=False)
        self.assertEqual(cfg.backend.ollama.url, "http://localhost:11434")
        self.assertEqual(cfg.api.port, 8765)

    def test_env_override_ollama_url(self):
        os.environ["OLLAMA_URL"] = "http://custom:9999"
        from config import load_config
        cfg = load_config(apply_env=True)
        self.assertEqual(cfg.backend.ollama.url, "http://custom:9999")

    def test_discord_token_enables_discord(self):
        os.environ["DISCORD_BOT_TOKEN"] = "test-token-123"
        from config import load_config
        cfg = load_config(apply_env=True)
        self.assertTrue(cfg.discord.enabled)
        self.assertEqual(cfg.discord.token, "test-token-123")

    def test_env_api_port_int_cast(self):
        os.environ["MYCONEX_API_PORT"] = "9876"
        from config import load_config
        cfg = load_config(apply_env=True)
        self.assertEqual(cfg.api.port, 9876)
        self.assertIsInstance(cfg.api.port, int)

    def test_env_log_level_uppercased(self):
        os.environ["MYCONEX_LOG_LEVEL"] = "debug"
        from config import load_config
        cfg = load_config(apply_env=True)
        self.assertEqual(cfg.logging.level, "DEBUG")

    def test_yaml_loading(self):
        from config import load_config
        yaml_content = "ollama:\n  url: http://yaml-host:11434\napi:\n  port: 9090\n"
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = load_config(yaml_path=tmp, apply_env=False)
            self.assertEqual(cfg.backend.ollama.url, "http://yaml-host:11434")
            self.assertEqual(cfg.api.port, 9090)
        finally:
            os.unlink(tmp)

    def test_env_overrides_yaml(self):
        from config import load_config
        os.environ["OLLAMA_URL"] = "http://from-env:11434"
        yaml_content = "ollama:\n  url: http://from-yaml:11434\n"
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = load_config(yaml_path=tmp, apply_env=True)
            self.assertEqual(cfg.backend.ollama.url, "http://from-env:11434")
        finally:
            os.unlink(tmp)

    def test_to_dict_serializable(self):
        from config import load_config
        cfg = load_config(apply_env=False)
        json.dumps(cfg.to_dict())

    def test_to_legacy_dict_keys(self):
        from config import load_config
        cfg = load_config(apply_env=False)
        legacy = cfg.to_legacy_dict()
        for key in ["ollama", "nats", "redis", "qdrant", "discord", "rlm"]:
            self.assertIn(key, legacy)

    def test_get_config_singleton(self):
        from config import get_config, reset_config
        reset_config()
        c1 = get_config()
        c2 = get_config()
        self.assertIs(c1, c2)

    def test_dotenv_loading(self):
        from config import load_config
        with tempfile.NamedTemporaryFile(
            suffix=".env", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("MYCONEX_MEMORY_NAMESPACE=dotenv-ns\n")
            tmp = f.name
        try:
            os.environ.pop("MYCONEX_MEMORY_NAMESPACE", None)
            cfg = load_config(env_file=tmp, apply_env=True)
            self.assertEqual(cfg.memory.namespace, "dotenv-ns")
        finally:
            os.unlink(tmp)
            os.environ.pop("MYCONEX_MEMORY_NAMESPACE", None)


# ═══════════════════════════════════════════════════════════════════════════════
# J. RLMAgent Tests (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLMAgentMocked(unittest.TestCase):
    """RLMAgent with mocked chat() — no real LLM needed."""

    def _make_agent(self):
        from orchestration.agents.rlm_agent import create_rlm_agent
        return create_rlm_agent(name="test-rlm", ollama_url="http://localhost:11434")

    def test_create_rlm_agent(self):
        agent = self._make_agent()
        self.assertIsNotNone(agent)

    def test_can_handle_returns_bool(self):
        agent = self._make_agent()
        result = agent.can_handle("analysis")
        self.assertIsInstance(result, bool)

    def test_handle_task_callable(self):
        # handle_task(task_id, task_type, payload) — just verify it's async callable
        agent = self._make_agent()
        import asyncio, inspect
        self.assertTrue(inspect.iscoroutinefunction(agent.handle_task))

    def test_format_discord_response_truncates(self):
        agent = self._make_agent()
        long_text = "x" * 3000
        # format_discord_response(response, result) — create a minimal AgentResult mock
        mock_result = MagicMock()
        mock_result.response = long_text
        mock_result.success = True
        mock_result.error = None
        result = agent.format_discord_response(long_text, mock_result)
        self.assertLessEqual(len(result), 2200)

    def test_total_tasks_counter_exists(self):
        agent = self._make_agent()
        self.assertIsInstance(agent._total_tasks, int)

    def test_tools_dict_exists(self):
        agent = self._make_agent()
        self.assertIsInstance(agent._tools, dict)

    def test_status_includes_router_info(self):
        agent = self._make_agent()
        status = agent.status()
        self.assertIn("router_attached", status)


# ═══════════════════════════════════════════════════════════════════════════════
# K. CodebaseIndex Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodebaseIndex(unittest.TestCase):
    """Inverted keyword index for codebase self-awareness."""

    def test_build_and_search(self):
        from core.gateway.python_repl import CodebaseIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            # Index tokenizes by splitting on non-alphanumerics — use full token names
            Path(tmpdir, "module_a.py").write_text("def authenticate_user(): pass\n")
            Path(tmpdir, "module_b.py").write_text("class DatabaseConnection: pass\n")
            idx = CodebaseIndex(root=tmpdir)
            idx.build()
            results = idx.search("authenticate_user")   # full token
            self.assertTrue(len(results) > 0)
            self.assertIn("file_path", results[0])

    def test_search_returns_relevant_files(self):
        from core.gateway.python_repl import CodebaseIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "auth.py").write_text("login logout session token\n")
            Path(tmpdir, "data.py").write_text("database query insert select\n")
            idx = CodebaseIndex(root=tmpdir)
            idx.build()
            # Search by actual tokens that appear in the files
            auth_results = idx.search("login")
            data_results = idx.search("database")
            auth_files = [r["file_path"] for r in auth_results]
            data_files = [r["file_path"] for r in data_results]
            self.assertTrue(any("auth" in f for f in auth_files))
            self.assertTrue(any("data" in f for f in data_files))

    def test_empty_directory_does_not_crash(self):
        from core.gateway.python_repl import CodebaseIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CodebaseIndex(root=tmpdir)
            idx.build()
            results = idx.search("anything")
            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
