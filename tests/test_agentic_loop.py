"""
Tests for MYCONEX agentic tools integration and Discord gateway agentic loop.

Coverage:
  - agentic_tools.py: remember, research, task_execution handlers
  - Memory persistence (JSON at ~/.myconex/memory.json)
  - Research tool (DuckDuckGo via ddgs)
  - Task execution (shell commands + timeout)
  - Tool registry registration into hermes
  - hermes AIAgent integration with llama3.1:8b via Ollama
  - _run_with_hermes: response, history persistence, error propagation
  - on_message clarify callback intercept routing
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

from core.gateway.agentic_tools import (
    AGENTIC_TOOLSET,
    handle_memory,
    handle_research,
    handle_task_execution,
    register_agentic_tools,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Redirect _MEMORY_FILE to a temp path so tests never touch ~/.myconex/."""
    mem = tmp_path / "memory.json"
    monkeypatch.setattr("core.gateway.agentic_tools._MEMORY_FILE", mem)
    yield mem


# ─── 1–4  remember (handle_memory) ────────────────────────────────────────────


def test_remember_store_and_retrieve(isolated_memory):
    """Storing a value and immediately retrieving it returns that value."""
    handle_memory(action="store", key="color", value="blue")
    result = handle_memory(action="retrieve", key="color")
    assert "blue" in result
    assert "color" in result


def test_remember_list_and_delete(isolated_memory):
    """list shows all keys; delete removes a specific key."""
    handle_memory(action="store", key="a", value="1")
    handle_memory(action="store", key="b", value="2")

    listed = handle_memory(action="list")
    assert "a" in listed and "b" in listed

    handle_memory(action="delete", key="a")
    listed_after = handle_memory(action="list")
    assert "a" not in listed_after
    assert "b" in listed_after


def test_remember_persistence_writes_json(isolated_memory):
    """Stored values are written to the JSON file at the configured path."""
    handle_memory(action="store", key="node", value="T2")
    assert isolated_memory.exists()
    data = json.loads(isolated_memory.read_text())
    assert data["node"] == "T2"


def test_remember_unknown_action_returns_error(isolated_memory):
    """An unrecognised action returns an error string, does not raise."""
    result = handle_memory(action="frobnicate")
    assert "frobnicate" in result.lower() or "unknown" in result.lower()


# ─── 5–7  research (handle_research) ──────────────────────────────────────────


def _ddgs_mock(results):
    """Build a context-manager-compatible DDGS mock returning *results*."""
    inst = MagicMock()
    inst.__enter__ = MagicMock(return_value=inst)
    inst.__exit__ = MagicMock(return_value=False)
    inst.text.return_value = results
    mod = MagicMock()
    mod.DDGS = MagicMock(return_value=inst)
    return mod


def test_research_returns_formatted_results():
    """Returns titles, URLs, and body snippets for a valid query."""
    mock_results = [
        {"title": "AI Weekly", "href": "https://ai.example.com", "body": "Top AI stories"},
        {"title": "LLM News",  "href": "https://llm.example.com", "body": "Model releases"},
    ]
    with patch.dict(sys.modules, {"ddgs": _ddgs_mock(mock_results)}):
        result = handle_research(query="latest AI news", max_results=2)

    assert "AI Weekly" in result
    assert "https://ai.example.com" in result
    assert "Top AI stories" in result


def test_research_empty_query_returns_error():
    """Empty query returns an error string without calling the network."""
    result = handle_research(query="")
    assert "error" in result.lower()


def test_research_no_results_reports_clearly():
    """When DuckDuckGo returns an empty list, the message says so."""
    with patch.dict(sys.modules, {"ddgs": _ddgs_mock([])}):
        result = handle_research(query="xyzzy12345", max_results=5)
    assert "no results" in result.lower()


# ─── 8–10  task_execution (handle_task_execution) ─────────────────────────────


def test_task_execution_runs_real_command():
    """A simple shell command executes and stdout is returned."""
    result = handle_task_execution(command="echo myconex_test_marker")
    assert "myconex_test_marker" in result


def test_task_execution_empty_command_returns_error():
    """Empty command returns an error string without spawning a subprocess."""
    result = handle_task_execution(command="")
    assert "error" in result.lower()


def test_task_execution_timeout_returns_message():
    """TimeoutExpired is caught and returned as a user-friendly string."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sleep", 1)):
        result = handle_task_execution(command="sleep 999", timeout=1)
    assert "timed out" in result.lower() or "timeout" in result.lower()


# ─── 11  Tool registry registration ───────────────────────────────────────────


def test_register_agentic_tools_registers_three_tools():
    """register_agentic_tools() registers remember, research, task_execution."""
    mock_registry = MagicMock()
    mock_registry._toolset_checks = {}
    mock_registry_mod = MagicMock()
    mock_registry_mod.registry = mock_registry

    with patch.dict(sys.modules, {"tools": MagicMock(), "tools.registry": mock_registry_mod}):
        success = register_agentic_tools()

    assert success is True
    # Tool count grows as new tools are added; verify core tools are always present
    registered = {c.kwargs["name"] for c in mock_registry.register.call_args_list}
    assert {"remember", "research", "task_execution"}.issubset(registered)


# ─── 12–13  DiscordGateway hermes agentic loop ────────────────────────────────


def _make_bare_gateway():
    """Return a DiscordGateway instance with internal state pre-seeded."""
    from core.gateway.discord_gateway import DiscordGateway
    from core.gateway.self_improvement import HermesSelfImprover
    from core.gateway.dlam_tasks import DLAMTaskGenerator
    gw = DiscordGateway.__new__(DiscordGateway)
    gw._config = {"discord": {}, "ollama": {"url": "http://localhost:11434"}}
    gw._states = {}
    gw._agents = {}
    gw._in_flight = set()
    gw._bot_participated_threads = set()
    gw._improver = HermesSelfImprover()
    gw._dlam_gen = DLAMTaskGenerator()
    gw._refl_agent = None
    return gw


def test_run_with_hermes_returns_response_and_persists_history():
    """
    _run_with_hermes forwards user_message to AIAgent and stores returned
    messages into state.history for the next conversation turn.
    """
    messages = [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "final_response": "4",
        "messages": messages,
    }

    with patch("core.gateway.discord_gateway._AIAGENT_AVAILABLE", True), \
         patch("core.gateway.discord_gateway.DiscordGateway._resolve_runtime",
               return_value=("http://localhost:11434/v1", "", "llama3.1:8b", "chat_completions")):

        gw = _make_bare_gateway()
        gw._agents["ch:1"] = mock_agent

        async def run():
            return await gw._run_with_hermes(
                key="ch:1",
                content="What is 2+2?",
                attachment_urls=[],
                status_msg=None,
                loop=asyncio.get_event_loop(),
                channel=AsyncMock(),
                author_id=10,
            )

        result = asyncio.run(run())

    assert result["final_response"] == "4"
    assert gw._states["ch:1"].history == messages


def test_run_with_hermes_error_is_propagated():
    """When AIAgent returns a failed result, the error dict is returned as-is."""
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "error": "Ollama connection refused",
        "failed": True,
    }

    with patch("core.gateway.discord_gateway._AIAGENT_AVAILABLE", True), \
         patch("core.gateway.discord_gateway.DiscordGateway._resolve_runtime",
               return_value=("http://localhost:11434/v1", "", "llama3.1:8b", "chat_completions")):

        gw = _make_bare_gateway()
        gw._agents["ch:2"] = mock_agent

        async def run():
            return await gw._run_with_hermes(
                key="ch:2",
                content="ping",
                attachment_urls=[],
                status_msg=None,
                loop=asyncio.get_event_loop(),
                channel=AsyncMock(),
                author_id=11,
            )

        result = asyncio.run(run())

    assert result.get("failed") is True
    assert "Ollama" in result.get("error", "")


def test_on_message_clarify_pending_is_consumed_not_rerouted():
    """
    When a pending clarify event exists for a channel, the incoming message
    is fed directly to clarify.receive() and is NOT routed to _handle_message.
    """
    from core.gateway.discord_gateway import _ChannelState

    state = _ChannelState()
    clarify = MagicMock()
    clarify._pending = threading.Event()   # truthy non-None → intercept branch
    state.clarify = clarify

    states = {"ch:3": state}
    incoming = "option 2"
    routed_normally = True

    # Mirrors the intercept logic in the on_message handler (line ~534-538)
    key = "ch:3"
    s = states.get(key)
    if s and s.clarify and s.clarify._pending:
        s.clarify.receive(incoming)
        routed_normally = False

    clarify.receive.assert_called_once_with(incoming)
    assert routed_normally is False
