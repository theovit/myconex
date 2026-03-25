"""
MYCONEX Agentic Tool Handlers
------------------------------
Tools registered into the hermes-agent tool registry so agents have reliable,
clearly-named capabilities:

  remember        — store and retrieve persistent facts
  research        — DuckDuckGo web search with result summaries
  task_execution  — run shell commands on the host machine
  python_repl     — execute Python in a persistent namespace (Phase 2 RLM)
  web_read        — fetch a URL and return its structure as text (Phase 2 RLM)
  codebase_search — keyword search over the MYCONEX source tree (Phase 2 RLM)
  gguf_infer      — run a local GGUF model via llama-cpp-python (Phase 2 RLM)

  ── Filesystem suite (OpenClaw-style) ──────────────────────────────────────
  read_file       — read file contents with optional line range
  write_file      — write/overwrite a file
  edit_file       — exact string replacement inside a file
  list_dir        — list directory contents
  glob_files      — find files by glob pattern
  grep_files      — search file contents by regex

Usage (automatic — called at gateway startup):
    from core.gateway.agentic_tools import register_agentic_tools, AGENTIC_TOOLSET
    register_agentic_tools()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import glob as _glob
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run_async(coro) -> Any:
    """
    Safely run an async coroutine from a synchronous tool handler.

    Works in both contexts:
      - No event loop running  → asyncio.run() directly
      - Inside a running loop  → submit to a thread pool with its own loop
        (avoids the 'cannot run nested event loop' RuntimeError)
    """
    try:
        asyncio.get_running_loop()
        # We're inside a running loop — offload to a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

logger = logging.getLogger(__name__)

AGENTIC_TOOLSET = "agentic"

# Phase 2: lazy-import RLM tools to avoid hard dependency at module load time
def _get_repl_pool():
    from core.gateway.python_repl import REPL_POOL
    return REPL_POOL

def _get_web_reader():
    from core.gateway.python_repl import WebPageReader
    return WebPageReader()

def _get_codebase_index():
    from core.gateway.python_repl import get_codebase_index
    return get_codebase_index()

# ─── Memory store ─────────────────────────────────────────────────────────────
# Simple JSON file at ~/.myconex/memory.json — independent of hermes memory
# so it works even when hermes memory toolset has issues.
_MEMORY_FILE = Path.home() / ".myconex" / "memory.json"


def _load_memory() -> Dict[str, Any]:
    try:
        _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _MEMORY_FILE.exists():
            return json.loads(_MEMORY_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_memory(data: Dict[str, Any]) -> None:
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MEMORY_FILE.write_text(json.dumps(data, indent=2))


# ─── Tool handlers ────────────────────────────────────────────────────────────

def handle_memory(action: str, key: str = "", value: str = "", **_) -> str:
    """
    Store or retrieve a persistent fact.

    action="store"    — save value under key
    action="retrieve" — return value for key (or all keys if key is empty)
    action="list"     — list all stored keys
    action="delete"   — remove a key
    """
    action = (action or "").strip().lower()
    store = _load_memory()

    if action == "store":
        if not key:
            return "error: key required for store"
        store[key] = value
        _save_memory(store)
        # Mirror to Qdrant so all mesh nodes can retrieve it via search_memory
        try:
            from integrations.knowledge_store import embed_and_store
            _run_async(embed_and_store(
                text=f"{key}: {value}",
                source="remember",
                metadata={"key": key, "value": value[:500]},
                memory_type="remember",
            ))
        except Exception:
            pass
        return f"Stored: {key} = {value!r}"

    if action == "retrieve":
        if not key:
            if not store:
                return "Memory is empty."
            return "\n".join(f"{k}: {v}" for k, v in store.items())
        val = store.get(key)
        if val is None:
            return f"No memory found for key: {key!r}"
        return f"{key}: {val}"

    if action == "list":
        if not store:
            return "Memory is empty."
        return "Stored keys: " + ", ".join(store.keys())

    if action == "delete":
        if key in store:
            del store[key]
            _save_memory(store)
            return f"Deleted: {key}"
        return f"Key not found: {key!r}"

    return f"Unknown action: {action!r}. Use store, retrieve, list, or delete."


def handle_research(query: str = "", max_results: int = 5, **_) -> str:
    """
    Search the web using DuckDuckGo and return a summary of top results.
    Returns titles, URLs, and snippets.
    """
    if not query:
        return "error: query required"

    try:
        try:
            from ddgs import DDGS  # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # legacy name fallback
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=int(max_results)))
        if not results:
            return f"No results found for: {query!r}"
        lines = [f"Search results for: {query!r}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r.get('title', 'No title')}**")
            lines.append(f"   {r.get('href', '')}")
            snippet = r.get("body", "")
            if snippet:
                lines.append(f"   {snippet[:200]}")
            lines.append("")
        return "\n".join(lines).strip()
    except ImportError:
        return "Search library not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Research error: {e}"


def handle_task_execution(command: str = "", timeout: int = 30, **_) -> str:
    """
    Execute a shell command on the host Linux machine and return its output.
    Runs through SandboxExecutor for resource limits and process isolation.
    Use for launching applications, running scripts, checking system state, etc.
    Examples: "steam &", "ls ~/Downloads", "systemctl status myconex"
    """
    if not command:
        return "error: command required"

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from tools.sandbox_executor import get_sandbox_executor, SandboxConfig
        executor = get_sandbox_executor(SandboxConfig(timeout_s=float(timeout)))
        result = _run_async(executor.run_command(command))
        return result.output
    except ImportError:
        # Fallback to raw subprocess if sandbox not available
        logger.warning("[task_execution] sandbox unavailable, using subprocess fallback")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=int(timeout),
            )
            parts = []
            if result.stdout.strip():
                parts.append(result.stdout.strip())
            if result.stderr.strip():
                parts.append(f"stderr: {result.stderr.strip()}")
            if result.returncode != 0:
                parts.append(f"exit code: {result.returncode}")
            return "\n".join(parts) if parts else "(command completed with no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as exc:
            return f"Execution error: {exc}"
    except Exception as exc:
        return f"Execution error: {exc}"


# ─── OpenAI-format tool schemas ───────────────────────────────────────────────

# Schemas use the flat format expected by the hermes registry:
# {"name": ..., "description": ..., "parameters": {...}}
# model_tools.py wraps these in {"type": "function", "function": schema} itself.

# Note: "memory" is reserved by hermes's agent loop (intercepted before registry).
# We use "remember" to avoid the collision — same semantics, different name.
_REMEMBER_SCHEMA = {
    "name": "remember",
    "description": (
        "Store or retrieve persistent facts across conversations. "
        "Use action='store' to save something (e.g. user preferences, facts). "
        "Use action='retrieve' to recall stored info by key. "
        "Use action='list' to see all stored keys."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "retrieve", "list", "delete"],
                "description": "The memory operation to perform.",
            },
            "key": {
                "type": "string",
                "description": "The key to store or retrieve (e.g. 'favorite_color', 'user_name').",
            },
            "value": {
                "type": "string",
                "description": "The value to store (only needed for action='store').",
            },
        },
        "required": ["action"],
    },
}

_RESEARCH_SCHEMA = {
    "name": "research",
    "description": (
        "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
        "Use for current events, facts, documentation, or any web lookup."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

_TASK_EXECUTION_SCHEMA = {
    "name": "task_execution",
    "description": (
        "Run a shell command on the host Linux machine. "
        "Use to launch applications (steam, firefox, vlc), run scripts, "
        "check system state, manage files, install packages, etc. "
        "For background processes, append '&' to the command."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30). Use higher for long-running tasks.",
                "default": 30,
            },
        },
        "required": ["command"],
    },
}


# ─── Phase 2 Tool Handlers ────────────────────────────────────────────────────

def handle_python_repl(code: str = "", session_id: str = "default", timeout: int = 30, **_) -> str:
    """
    Execute Python code in a persistent REPL session.

    State persists across calls within the same session_id — variables,
    imports, and defined functions survive between invocations.
    """
    if not code:
        return "error: code required"
    try:
        pool = _get_repl_pool()
        result = _run_async(pool.execute(session_id, code, timeout=float(timeout)))
        return str(result)
    except Exception as exc:
        return f"REPL error: {exc}"


def handle_web_read(url: str = "", **_) -> str:
    """
    Fetch a URL and return its structure as LLM-readable text.

    Extracts title, headings, main text content, and links without
    requiring a browser or taking screenshots.
    """
    if not url:
        return "error: url required"
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http:// or https://"
    try:
        reader = _get_web_reader()
        result = _run_async(reader.read(url))
        return reader.format_for_llm(result)
    except Exception as exc:
        return f"web_read error: {exc}"


def handle_codebase_search(query: str = "", top_k: int = 5, **_) -> str:
    """
    Keyword search over the MYCONEX source tree.

    Gives the agent self-awareness: it can find relevant code in its own
    codebase by describing what it is looking for in plain English.
    """
    if not query:
        return "error: query required"
    try:
        idx = _get_codebase_index()
        results = idx.search(query, top_k=int(top_k))
        if not results:
            return f"No codebase matches for: {query!r}"
        lines = [f"Codebase search: {query!r}\n"]
        for r in results:
            lines.append(f"📄 {r['file_path']}:{r['start_line']} (score={r['score']})")
            lines.append("```")
            lines.append(r["content"][:600])
            lines.append("```\n")
        return "\n".join(lines).strip()
    except Exception as exc:
        return f"codebase_search error: {exc}"


def handle_gguf_infer(
    model_path: str = "",
    prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    **_,
) -> str:
    """
    Run inference on a local GGUF model via llama-cpp-python.

    Loads the model on first call (may be slow). Subsequent calls to the same
    model_path reuse the loaded instance.
    """
    if not model_path:
        return "error: model_path required (path to a .gguf file)"
    if not prompt:
        return "error: prompt required"
    try:
        from core.gateway.python_repl import GGUFBackend
        backend = GGUFBackend(model_path=model_path)
        if not backend.available:
            return (
                f"GGUF backend unavailable: model_path={model_path!r}. "
                "Install llama-cpp-python: pip install llama-cpp-python"
            )
        result = _run_async(backend.generate(prompt, max_tokens=max_tokens, temperature=temperature))
        return result
    except Exception as exc:
        return f"gguf_infer error: {exc}"


# ─── Phase 2 Schemas ──────────────────────────────────────────────────────────

_PYTHON_REPL_SCHEMA = {
    "name": "python_repl",
    "description": (
        "Execute Python code in a persistent REPL session. "
        "Variables, imports, and functions persist across multiple calls "
        "within the same session_id. Use to transform data, run analysis, "
        "or compute results without loading everything into the LLM context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."},
            "session_id": {
                "type": "string",
                "description": "Session identifier for state persistence (default: 'default').",
                "default": "default",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default 30).",
                "default": 30,
            },
        },
        "required": ["code"],
    },
}

_DLAM_SCHEMA = {
    "name": "dlam",
    "description": (
        "Delegate a web research or computer-use task to DLAM (https://dlam.rabbit.tech/), "
        "the Rabbit R1-powered browser agent running in Brave. "
        "Use this as the PRIMARY tool for web research, browsing, and keyboard/mouse tasks — "
        "it offloads the heavy lifting to a real browser so you don't burn tokens "
        "fetching and parsing HTML yourself.\n\n"
        "action='browse': Navigate to a URL and perform a task on the page (summarise, extract, fill form).\n"
        "action='search': Web-search a query and return structured top results.\n"
        "action='task':   Run a free-form computer-use task (open app, type text, click button).\n"
        "action='status': Check whether DLAM is available (Brave CDP + R1 connected)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["browse", "search", "task", "status"],
                "description": "What kind of DLAM task to run.",
            },
            "url": {
                "type": "string",
                "description": "URL to visit (required for action='browse', optional for action='task').",
            },
            "task": {
                "type": "string",
                "description": (
                    "For action='browse': what to do on the page "
                    "(e.g. 'summarise the abstract', 'extract all code examples'). "
                    "For action='task': the full computer-use instruction."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query (required for action='search').",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to return (default 5, max 10). Only for action='search'.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait for DLAM to complete the task (default 120).",
            },
        },
        "required": ["action"],
    },
}


def handle_dlam(
    action: str = "status",
    url: str = "",
    task: str = "",
    query: str = "",
    num_results: int = 5,
    timeout: int = 120,
    **_,
) -> str:
    """
    Synchronous wrapper around the async DLAM client.
    Routes to dlam_browse / dlam_search / dlam_task / dlam_status.
    """
    action = (action or "status").strip().lower()

    async def _run():
        from integrations.dlam_client import dlam_browse, dlam_search, dlam_task, dlam_status
        if action == "browse":
            if not url:
                return "[dlam] 'browse' requires a url parameter."
            return await dlam_browse(url=url, task=task, timeout=timeout)
        elif action == "search":
            if not query:
                return "[dlam] 'search' requires a query parameter."
            return await dlam_search(query=query, num_results=num_results, timeout=timeout)
        elif action == "task":
            if not task:
                return "[dlam] 'task' requires a task parameter."
            return await dlam_task(description=task, url=url, timeout=timeout)
        elif action == "status":
            info = await dlam_status()
            q = info.get("queue", {})
            lines = [
                "DLAM status:",
                f"  cdp available (Brave debug port): {info['cdp_available']}",
                f"  dlam tab open: {info['dlam_tab_open']}",
                f"  r1 connected: {info['r1_connected']}",
                f"  queue — pending:{q.get('pending',0)} "
                f"processing:{q.get('processing',0)} "
                f"completed:{q.get('completed',0)} "
                f"failed:{q.get('failed',0)}",
            ]
            if info.get("setup_hint"):
                lines.append(f"  hint: {info['setup_hint']}")
            return "\n".join(lines)
        else:
            return f"[dlam] unknown action '{action}'. Use: browse / search / task / status"

    return _run_async(_run())


_WEB_READ_SCHEMA = {
    "name": "web_read",
    "description": (
        "Fetch a web page and return its structure as text. "
        "Extracts title, headings, main content, and links — no browser needed. "
        "Use when you need the full content of a specific URL, not just a search snippet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The HTTP/HTTPS URL to fetch."},
        },
        "required": ["url"],
    },
}

_CODEBASE_SEARCH_SCHEMA = {
    "name": "codebase_search",
    "description": (
        "Search the MYCONEX codebase by keyword for relevant code. "
        "Use to find where a feature is implemented, how a class is used, "
        "or what files handle a specific task type. "
        "Returns file path, line number, and matching code chunks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language or code-keyword search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

_GGUF_INFER_SCHEMA = {
    "name": "gguf_infer",
    "description": (
        "Run inference on a local GGUF model file via llama-cpp-python. "
        "Use for private/offline inference on locally stored models. "
        "Requires llama-cpp-python installed and a valid .gguf file path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_path": {
                "type": "string",
                "description": "Absolute path to the .gguf model file.",
            },
            "prompt": {"type": "string", "description": "Text prompt for generation."},
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to generate (default 256).",
                "default": 256,
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature 0.0–2.0 (default 0.7).",
                "default": 0.7,
            },
        },
        "required": ["model_path", "prompt"],
    },
}


# ─── Semantic Memory Search ───────────────────────────────────────────────────

def handle_search_memory(
    query: str = "",
    limit: int = 8,
    source: str = "",
    **_,
) -> str:
    """Semantic search over the Qdrant knowledge base (emails + YouTube + RSS)."""
    if not query:
        return "error: query required"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.knowledge_store import search, format_results, get_stats
    except ImportError as exc:
        return f"search_memory: import error — {exc}"

    try:
        results = _run_async(search(query, limit=int(limit), source_filter=source or None))
        return format_results(results, query)
    except Exception as exc:
        return f"search_memory error: {exc}"


def handle_memory_stats(**_) -> str:
    """Return Qdrant knowledge base statistics."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.knowledge_store import get_stats
        stats = _run_async(get_stats())
        if not stats.get("available"):
            return "Knowledge base unavailable — is Qdrant running? (docker compose up qdrant)"
        return (
            f"Knowledge base stats:\n"
            f"  Collection:  {stats.get('collection', '')}\n"
            f"  Vectors:     {stats.get('vector_count', 0):,}\n"
            f"  Vector size: {stats.get('vector_size', 0)}\n"
            f"  Distance:    {stats.get('distance_metric', '')}\n"
            f"  Qdrant:      available"
        )
    except Exception as exc:
        return f"memory_stats error: {exc}"


_SEARCH_MEMORY_SCHEMA = {
    "name": "search_memory",
    "description": (
        "Semantic search over the unified knowledge base built from emails, YouTube videos, "
        "RSS articles, and podcasts. Uses vector similarity — finds conceptually related content "
        "even if exact words don't match. Use this when you need to recall what was learned "
        "about a topic, find related project ideas, or retrieve past wisdom."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 8).",
                "default": 8,
            },
            "source": {
                "type": "string",
                "description": "Filter by source: 'email', 'youtube', 'rss', 'podcast'. Omit for all sources.",
                "default": "",
            },
        },
        "required": ["query"],
    },
}

_MEMORY_STATS_SCHEMA = {
    "name": "memory_stats",
    "description": "Show how many items are stored in the vector knowledge base and whether Qdrant is reachable.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ─── YouTube Tool Handler ─────────────────────────────────────────────────────

def handle_youtube_profile(
    action: str = "show",
    url: str = "",
    patterns: str = "",
    **_,
) -> str:
    """
    Access the YouTube knowledge base built from watch history and processed videos.

    action="show"    — recent processed videos + stats
    action="wisdom"  — raw Fabric wisdom extractions from videos
    action="process" — process a single YouTube URL right now
    action="refresh" — re-scan watch history for new videos
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.youtube_ingester import YouTubeIngester
    except ImportError as exc:
        return f"youtube_profile: import error — {exc}"

    action = (action or "show").strip().lower()

    if action == "show":
        return YouTubeIngester.get_recent_insights(n=10)

    if action == "wisdom":
        return YouTubeIngester.get_wisdom(n=5)

    if action == "process":
        if not url:
            return "error: url required for action='process'"
        pat_list = [p.strip() for p in patterns.split(",") if p.strip()] if patterns else None
        ingester = YouTubeIngester()
        try:
            return _run_async(ingester.process_url(url, patterns=pat_list))
        except Exception as exc:
            return f"youtube process error: {exc}"

    if action == "refresh":
        ingester = YouTubeIngester()
        try:
            count = _run_async(ingester.ingest_history())
            return f"Processed {count} new video(s) from watch history.\n\n" + YouTubeIngester.get_recent_insights(n=5)
        except Exception as exc:
            return f"youtube refresh error: {exc}"

    return f"Unknown action: {action!r}. Use show, wisdom, process, or refresh."


_YOUTUBE_PROFILE_SCHEMA = {
    "name": "youtube_profile",
    "description": (
        "Access wisdom and insights extracted from YouTube videos. "
        "Use action='process' with a URL to extract wisdom from any YouTube video right now. "
        "Use action='wisdom' to see Fabric-extracted insights from previously watched videos. "
        "Use action='show' to see recently processed videos. "
        "Use action='refresh' to scan watch history for new unprocessed videos. "
        "Requires Fabric to be installed for transcript extraction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["show", "wisdom", "process", "refresh"],
                "description": "Operation to perform (default: show).",
                "default": "show",
            },
            "url": {
                "type": "string",
                "description": "YouTube URL — required for action='process'.",
            },
            "patterns": {
                "type": "string",
                "description": (
                    "Comma-separated Fabric patterns to apply (default: extract_wisdom,summarize). "
                    "Only used with action='process'."
                ),
                "default": "",
            },
        },
        "required": ["action"],
    },
}


# ─── RSS Feed Tool Handler ────────────────────────────────────────────────────

def handle_rss_feed(
    action: str = "show",
    url: str = "",
    **_,
) -> str:
    """
    Manage and query the RSS feed monitor.

    action="show"   — recent processed articles + stats
    action="wisdom" — Fabric-extracted wisdom from RSS articles
    action="feeds"  — list all configured feed URLs
    action="add"    — add a new feed URL (requires url=)
    action="remove" — remove a feed URL (requires url=)
    action="poll"   — trigger an immediate poll of all feeds
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.rss_monitor import RSSMonitor
    except ImportError as exc:
        return f"rss_feed: import error — {exc}"

    action = (action or "show").strip().lower()
    monitor = RSSMonitor()

    if action == "show":
        return monitor.get_recent_insights(n=10)

    if action == "wisdom":
        return monitor.get_wisdom(n=5)

    if action == "feeds":
        feeds = monitor.list_feeds()
        if not feeds:
            return "No feeds configured. Use action='add' with a URL to add one."
        return "Configured RSS feeds:\n" + "\n".join(f"  • {f}" for f in feeds)

    if action == "add":
        if not url:
            return "error: url required for action='add'"
        added = monitor.add_feed(url)
        return f"Feed {'added' if added else 'already present'}: {url}"

    if action == "remove":
        if not url:
            return "error: url required for action='remove'"
        removed = monitor.remove_feed(url)
        return f"Feed {'removed' if removed else 'not found'}: {url}"

    if action == "poll":
        try:
            count = _run_async(monitor.poll_all())
            return f"Polled all feeds — processed {count} new article(s).\n\n" + monitor.get_recent_insights(n=5)
        except Exception as exc:
            return f"rss poll error: {exc}"

    return f"Unknown action: {action!r}. Use show, wisdom, feeds, add, remove, or poll."


_RSS_FEED_SCHEMA = {
    "name": "rss_feed",
    "description": (
        "Manage the RSS/Atom feed monitor and query extracted wisdom from news and blog articles. "
        "Use action='feeds' to see configured feeds. "
        "Use action='add' with url= to subscribe to a new feed. "
        "Use action='remove' with url= to unsubscribe. "
        "Use action='poll' to fetch new articles from all feeds right now. "
        "Use action='wisdom' to read Fabric-extracted insights from recent articles. "
        "Use action='show' to see recently processed articles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["show", "wisdom", "feeds", "add", "remove", "poll"],
                "description": "Operation to perform (default: show).",
                "default": "show",
            },
            "url": {
                "type": "string",
                "description": "Feed URL — required for action='add' or action='remove'.",
            },
        },
        "required": ["action"],
    },
}


# ─── Podcast Tool Handler ─────────────────────────────────────────────────────

def handle_podcast(
    action: str = "show",
    url: str = "",
    **_,
) -> str:
    """
    Manage the podcast ingester and query extracted wisdom from episodes.

    action="show"    — recently processed episodes
    action="wisdom"  — Fabric-extracted wisdom from episodes
    action="feeds"   — list configured podcast feed URLs
    action="add"     — subscribe to a podcast feed (requires url=)
    action="remove"  — unsubscribe from a feed (requires url=)
    action="poll"    — trigger immediate poll of all podcast feeds
    action="check"   — check if required dependencies are installed
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.podcast_ingester import PodcastIngester
    except ImportError as exc:
        return f"podcast: import error — {exc}"

    action = (action or "show").strip().lower()
    ingester = PodcastIngester()

    if action == "show":
        return ingester.get_recent_insights(n=10)

    if action == "wisdom":
        return ingester.get_wisdom(n=5)

    if action == "feeds":
        feeds = ingester.list_feeds()
        if not feeds:
            return "No podcast feeds configured. Use action='add' with a feed URL to add one."
        return "Configured podcast feeds:\n" + "\n".join(f"  • {f}" for f in feeds)

    if action == "add":
        if not url:
            return "error: url required for action='add'"
        added = ingester.add_feed(url)
        return f"Feed {'added' if added else 'already present'}: {url}"

    if action == "remove":
        if not url:
            return "error: url required for action='remove'"
        removed = ingester.remove_feed(url)
        return f"Feed {'removed' if removed else 'not found'}: {url}"

    if action == "poll":
        try:
            count = _run_async(ingester.poll_all())
            return f"Polled all podcast feeds — processed {count} new episode(s).\n\n" + ingester.get_recent_insights(n=5)
        except Exception as exc:
            return f"podcast poll error: {exc}"

    if action == "check":
        return ingester.check_dependencies()

    return f"Unknown action: {action!r}. Use show, wisdom, feeds, add, remove, poll, or check."


_PODCAST_SCHEMA = {
    "name": "podcast",
    "description": (
        "Manage the podcast ingester and query Fabric-extracted wisdom from podcast episodes. "
        "Requires: feedparser, yt-dlp, and either whisper.cpp or openai-whisper. "
        "Use action='check' to verify dependencies are installed. "
        "Use action='feeds' to see configured podcasts. "
        "Use action='add' with url= to subscribe to a podcast RSS feed. "
        "Use action='remove' with url= to unsubscribe. "
        "Use action='poll' to download and transcribe new episodes immediately. "
        "Use action='wisdom' to read extracted insights from transcribed episodes. "
        "Use action='show' to see recently processed episodes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["show", "wisdom", "feeds", "add", "remove", "poll", "check"],
                "description": "Operation to perform (default: show).",
                "default": "show",
            },
            "url": {
                "type": "string",
                "description": "Podcast RSS feed URL — required for action='add' or action='remove'.",
            },
        },
        "required": ["action"],
    },
}


# ─── Fabric Tool Handler ──────────────────────────────────────────────────────

# 200 most useful Fabric patterns — shown in the tool description so the LLM
# can pick the right one without needing to call list_patterns first.
_FABRIC_NOTABLE_PATTERNS = (
    "summarize, extract_wisdom, extract_insights, extract_ideas, extract_quotes, "
    "extract_patterns, extract_recommendations, extract_references, "
    "analyze_paper, analyze_claims, analyze_debate, analyze_logs, analyze_malware, "
    "analyze_threat_report, analyze_answers, analyze_bill, "
    "create_summary, create_design_document, create_academic_paper, create_slides, "
    "create_mermaid_visualization, create_sigma_rules, "
    "improve_writing, improve_prompt, humanize, translate, convert_to_markdown, "
    "review_code, rate_content, find_hidden_message, "
    "write_essay, write_semgrep_rule, write_nuclei_template_rule, "
    "t_find_blindspots, t_year_in_review, label_and_rate, "
    "explain_code, explain_docs, get_wow_per_minute"
)


def handle_fabric(
    action: str = "apply",
    pattern: str = "summarize",
    text: str = "",
    url: str = "",
    vendor: str = "",
    model: str = "",
    **_,
) -> str:
    """
    Apply any Fabric pattern to text or a YouTube URL.

    action="apply"     — apply pattern to text
    action="youtube"   — extract YouTube transcript then apply pattern
    action="list"      — list all available patterns
    action="check"     — check if Fabric is installed/reachable
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.fabric_client import (
            apply_pattern, list_patterns, youtube_and_pattern, is_available
        )
    except ImportError as exc:
        return f"fabric: import error — {exc}"

    action = (action or "apply").strip().lower()

    if action == "check":
        ok = is_available()
        if ok:
            return "Fabric is available."
        return (
            "Fabric binary not found. Install with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/danielmiessler/fabric/"
            "main/scripts/installer/install.sh | bash\n"
            "  fabric --setup"
        )

    if action == "list":
        try:
            patterns = _run_async(list_patterns())
            if not patterns:
                return "No patterns found — is Fabric installed?"
            return f"Available patterns ({len(patterns)}):\n" + "\n".join(f"  {p}" for p in patterns)
        except Exception as exc:
            return f"fabric list error: {exc}"

    if action == "apply":
        if not text:
            return "error: text required for action='apply'"
        if not pattern:
            return "error: pattern required"
        try:
            return _run_async(apply_pattern(pattern, text, vendor, model))
        except Exception as exc:
            return f"fabric apply error: {exc}"

    if action == "youtube":
        if not url:
            return "error: url required for action='youtube'"
        pat = pattern or "extract_wisdom"
        try:
            return _run_async(youtube_and_pattern(url, pat))
        except Exception as exc:
            return f"fabric youtube error: {exc}"

    return f"Unknown action: {action!r}. Use apply, youtube, list, or check."


_FABRIC_SCHEMA = {
    "name": "fabric",
    "description": (
        "Apply any of Fabric's 252+ curated AI patterns to text or a YouTube video. "
        "Patterns are expert-crafted system prompts for specific tasks. "
        f"Notable patterns: {_FABRIC_NOTABLE_PATTERNS}. "
        "Use action='apply' with text to run a pattern. "
        "Use action='youtube' with a YouTube URL to extract the transcript and apply a pattern. "
        "Use action='list' to see all available patterns. "
        "Requires Fabric to be installed (fabric --setup)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["apply", "youtube", "list", "check"],
                "description": "Operation: apply a pattern, process a YouTube URL, list patterns, or check install status.",
                "default": "apply",
            },
            "pattern": {
                "type": "string",
                "description": "Fabric pattern name (e.g. 'summarize', 'extract_wisdom', 'analyze_paper'). Default: summarize.",
                "default": "summarize",
            },
            "text": {
                "type": "string",
                "description": "Input text to process (required for action='apply').",
            },
            "url": {
                "type": "string",
                "description": "YouTube URL (required for action='youtube').",
            },
            "vendor": {
                "type": "string",
                "description": "AI provider override (e.g. 'ollama', 'openai'). Omit to use Fabric's default.",
            },
            "model": {
                "type": "string",
                "description": "Model override (e.g. 'llama3.1:8b'). Omit to use Fabric's default.",
            },
        },
        "required": ["action"],
    },
}


# ─── Gmail Tool Handler ───────────────────────────────────────────────────────

def handle_check_email(
    action: str = "search",
    query: str = "ALL",
    folder: str = "INBOX",
    limit: int = 10,
    unread_only: bool = False,
    uid: str = "",
    **_,
) -> str:
    """
    Read Gmail via IMAP.

    action="search"        — search emails (returns list with body previews)
    action="read"          — fetch full body of one email by uid
    action="unread_count"  — return number of unread messages
    action="list_folders"  — list available mailbox folders
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.gmail_reader import GmailReader
    except ImportError as exc:
        return f"check_email: import error — {exc}"

    try:
        reader = GmailReader()
    except ValueError as exc:
        return (
            f"check_email: {exc}\n\n"
            "Add to .env:\n"
            "  GMAIL_ADDRESS=you@gmail.com\n"
            "  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx\n"
            "(Generate at: https://myaccount.google.com/apppasswords)"
        )
    except Exception as exc:
        return f"check_email: connection error — {exc}"

    action = (action or "search").strip().lower()

    if action == "search":
        try:
            emails = reader.search(
                query=query,
                folder=folder,
                limit=int(limit),
                unread_only=bool(unread_only),
            )
            return reader.format_for_llm(emails)
        except Exception as exc:
            return f"check_email search error: {exc}"

    if action == "read":
        if not uid:
            return "check_email: uid required for action='read'"
        try:
            result = reader.read(uid=uid, folder=folder)
            if not result:
                return f"No email found with uid={uid} in {folder}"
            return reader.format_for_llm([result])
        except Exception as exc:
            return f"check_email read error: {exc}"

    if action == "unread_count":
        try:
            count = reader.get_unread_count(folder=folder)
            return f"Unread messages in {folder}: {count}"
        except Exception as exc:
            return f"check_email unread_count error: {exc}"

    if action == "list_folders":
        try:
            folders = reader.list_folders()
            return "Available folders:\n" + "\n".join(f"  {f}" for f in folders)
        except Exception as exc:
            return f"check_email list_folders error: {exc}"

    return f"Unknown action: {action!r}. Use search, read, unread_count, or list_folders."


def handle_email_profile(action: str = "show", **_) -> str:
    """
    Show or refresh the accumulated interest profile built from emails.

    action="show"    — print the current interest profile
    action="refresh" — run an immediate ingest pass then show the profile
    action="insights"— show the most recently processed email summaries
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from integrations.email_ingester import EmailIngester
    except ImportError as exc:
        return f"email_profile: import error — {exc}"

    action = (action or "show").strip().lower()

    if action == "show":
        return EmailIngester.get_profile_summary()

    if action == "insights":
        return EmailIngester.get_recent_insights(n=10)

    if action == "wisdom":
        return EmailIngester.get_wisdom(n=5)

    if action == "refresh":
        ingester = EmailIngester()
        try:
            count = _run_async(ingester.ingest_once())
            result = f"Processed {count} new email(s).\n\n"
        except Exception as exc:
            result = f"Ingest error: {exc}\n\n"
        return result + EmailIngester.get_profile_summary()

    return f"Unknown action: {action!r}. Use show, refresh, or insights."


_EMAIL_PROFILE_SCHEMA = {
    "name": "email_profile",
    "description": (
        "Show or refresh the interest profile built from analysed emails. "
        "Use action='show' to see accumulated topics, likes, dislikes, project ideas, and key people. "
        "Use action='refresh' to trigger an immediate email ingest pass. "
        "Use action='insights' to see the most recently processed email summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["show", "refresh", "insights", "wisdom"],
                "description": (
                    "Operation: 'show' = interest profile, 'wisdom' = raw Fabric extractions, "
                    "'insights' = recent email summaries, 'refresh' = ingest now."
                ),
                "default": "show",
            },
        },
        "required": [],
    },
}


_CHECK_EMAIL_SCHEMA = {
    "name": "check_email",
    "description": (
        "Read Gmail messages via IMAP. "
        "Use action='search' to find emails by sender, subject, or keyword. "
        "Use action='read' to get the full body of a specific email by uid. "
        "Use action='unread_count' to see how many unread messages are waiting. "
        "Use action='list_folders' to see available mailboxes. "
        "Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "read", "unread_count", "list_folders"],
                "description": "Operation to perform (default: search).",
                "default": "search",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search query. Supports: 'unread', 'from:email@example.com', "
                    "'subject:invoice', 'body:keyword', or raw IMAP search string. "
                    "Default: ALL."
                ),
                "default": "ALL",
            },
            "folder": {
                "type": "string",
                "description": "Mailbox folder to search (default: INBOX).",
                "default": "INBOX",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of emails to return (default: 10, newest first).",
                "default": 10,
            },
            "unread_only": {
                "type": "boolean",
                "description": "If true, only return unread messages (default: false).",
                "default": False,
            },
            "uid": {
                "type": "string",
                "description": "Email UID — required for action='read'.",
            },
        },
        "required": ["action"],
    },
}


# ─── OpenClaw Filesystem Tool Handlers ───────────────────────────────────────

def handle_read_file(path: str = "", offset: int = 0, limit: int = 0, **_) -> str:
    """Read a file, optionally from a starting line with a max line count."""
    if not path:
        return "error: path required"
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"error: file not found: {path}"
        if not p.is_file():
            return f"error: not a file: {path}"
        lines = p.read_text(errors="replace").splitlines()
        start = max(0, int(offset))
        end = int(limit) if limit > 0 else len(lines)
        chunk = lines[start : start + end]
        numbered = "\n".join(f"{start + i + 1:>6}\t{line}" for i, line in enumerate(chunk))
        total = len(lines)
        header = f"[{p}  lines {start + 1}–{start + len(chunk)} of {total}]\n"
        return header + numbered
    except Exception as exc:
        return f"read_file error: {exc}"


def handle_write_file(path: str = "", content: str = "", **_) -> str:
    """Write content to a file, creating parent directories as needed."""
    if not path:
        return "error: path required"
    if content is None:
        return "error: content required"
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + (1 if content else 0)
        return f"Written {lines} lines to {p}"
    except Exception as exc:
        return f"write_file error: {exc}"


def handle_edit_file(
    path: str = "",
    old_string: str = "",
    new_string: str = "",
    replace_all: bool = False,
    **_,
) -> str:
    """Replace an exact string in a file. Fails if old_string is not found or is ambiguous."""
    if not path:
        return "error: path required"
    if old_string == "":
        return "error: old_string required"
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"error: file not found: {path}"
        original = p.read_text(errors="replace")
        count = original.count(old_string)
        if count == 0:
            return f"error: old_string not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"error: old_string appears {count} times — provide more context to make it "
                "unique, or set replace_all=true to replace every occurrence"
            )
        updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
        p.write_text(updated)
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {p}"
    except Exception as exc:
        return f"edit_file error: {exc}"


def handle_list_dir(path: str = ".", **_) -> str:
    """List the contents of a directory."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"error: path not found: {path}"
        if not p.is_dir():
            return f"error: not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        if not entries:
            return f"(empty directory: {p})"
        lines = [f"[{p}]"]
        for e in entries:
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                size = e.stat().st_size
                size_str = f"{size:,} B" if size < 1024 else f"{size // 1024:,} KB"
                lines.append(f"  {e.name}  ({size_str})")
        return "\n".join(lines)
    except Exception as exc:
        return f"list_dir error: {exc}"


def handle_glob_files(pattern: str = "", path: str = "", **_) -> str:
    """Find files matching a glob pattern, sorted by modification time."""
    if not pattern:
        return "error: pattern required"
    try:
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        full_pattern = str(base / pattern)
        matches = _glob.glob(full_pattern, recursive=True)
        if not matches:
            return f"No files matched: {pattern}"
        matches.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
        lines = [f"[{len(matches)} match(es) for {pattern!r}]"]
        lines += matches[:200]  # cap at 200 to protect context
        if len(matches) > 200:
            lines.append(f"... and {len(matches) - 200} more")
        return "\n".join(lines)
    except Exception as exc:
        return f"glob_files error: {exc}"


def handle_grep_files(
    pattern: str = "",
    path: str = "",
    file_glob: str = "",
    context: int = 0,
    ignore_case: bool = False,
    **_,
) -> str:
    """Search file contents by regex. Returns file:line matches."""
    if not pattern:
        return "error: pattern required"
    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        search_root = Path(path).expanduser().resolve() if path else Path.cwd()

        # Collect candidate files
        if search_root.is_file():
            candidates = [search_root]
        elif file_glob:
            candidates = [Path(m) for m in _glob.glob(str(search_root / "**" / file_glob), recursive=True)]
        else:
            candidates = [p for p in search_root.rglob("*") if p.is_file()]

        results: list[str] = []
        total_matches = 0
        for fpath in sorted(candidates):
            try:
                file_lines = fpath.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(file_lines):
                if regex.search(line):
                    total_matches += 1
                    if total_matches > 500:
                        continue  # count but don't add to output
                    ctx_start = max(0, i - context)
                    ctx_end = min(len(file_lines), i + context + 1)
                    for j in range(ctx_start, ctx_end):
                        prefix = f"{fpath}:{j + 1}"
                        marker = ">" if j == i else " "
                        results.append(f"{marker} {prefix}: {file_lines[j]}")
                    if context:
                        results.append("---")

        if not results and total_matches == 0:
            return f"No matches for {pattern!r}"
        header = f"[{total_matches} match(es) for {pattern!r}]"
        if total_matches > 500:
            header += f"  (showing first 500)"
        return header + "\n" + "\n".join(results)
    except re.error as exc:
        return f"grep_files: invalid regex — {exc}"
    except Exception as exc:
        return f"grep_files error: {exc}"


# ─── OpenClaw Schemas ─────────────────────────────────────────────────────────

_READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": (
        "Read the contents of a file. Returns lines with line numbers. "
        "Use offset+limit to read a specific range instead of the whole file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or ~ path to the file."},
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-indexed, default 0).",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (0 = all, default 0).",
                "default": 0,
            },
        },
        "required": ["path"],
    },
}

_WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": (
        "Write content to a file, overwriting it if it exists. "
        "Creates parent directories automatically. "
        "Use edit_file for targeted changes to existing files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or ~ path to the file."},
            "content": {"type": "string", "description": "Full content to write."},
        },
        "required": ["path", "content"],
    },
}

_EDIT_FILE_SCHEMA = {
    "name": "edit_file",
    "description": (
        "Replace an exact string in a file. Fails if old_string is not found. "
        "Fails if old_string is ambiguous (appears more than once) unless replace_all=true. "
        "Provide enough surrounding context in old_string to make it unique."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or ~ path to the file."},
            "old_string": {"type": "string", "description": "Exact text to find and replace."},
            "new_string": {"type": "string", "description": "Text to replace it with."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of just the first (default false).",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
}

_LIST_DIR_SCHEMA = {
    "name": "list_dir",
    "description": (
        "List the contents of a directory. Shows files (with size) and subdirectories. "
        "Use before read_file to explore structure without loading every file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current working directory).",
                "default": ".",
            },
        },
        "required": [],
    },
}

_GLOB_FILES_SCHEMA = {
    "name": "glob_files",
    "description": (
        "Find files matching a glob pattern. Supports ** for recursive matching. "
        "Results sorted by modification time (newest first). "
        "Examples: '**/*.py', 'src/**/*.ts', '*.json'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'config/*.yaml').",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search from (default: current working directory).",
                "default": "",
            },
        },
        "required": ["pattern"],
    },
}

_GREP_FILES_SCHEMA = {
    "name": "grep_files",
    "description": (
        "Search file contents by regular expression. "
        "Returns matching lines with file path and line number. "
        "Use file_glob to limit which files are searched (e.g. '*.py'). "
        "Use context to show lines before/after each match."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression pattern."},
            "path": {
                "type": "string",
                "description": "File or directory to search (default: current working directory).",
                "default": "",
            },
            "file_glob": {
                "type": "string",
                "description": "Limit search to files matching this glob (e.g. '*.py', '*.ts').",
                "default": "",
            },
            "context": {
                "type": "integer",
                "description": "Lines of context before and after each match (default 0).",
                "default": 0,
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive matching (default false).",
                "default": False,
            },
        },
        "required": ["pattern"],
    },
}


# ─── Registration ─────────────────────────────────────────────────────────────

def register_agentic_tools() -> bool:
    """
    Register memory, research, and task_execution into the hermes tool registry.
    Safe to call multiple times — re-registration is a no-op.
    Returns True on success, False if hermes registry is unavailable.
    """
    hermes_dir = Path(__file__).parent.parent.parent / "integrations" / "hermes-agent"
    if str(hermes_dir) not in sys.path:
        sys.path.insert(0, str(hermes_dir))

    try:
        from tools.registry import registry  # type: ignore[import]
    except ImportError:
        logger.warning("[agentic_tools] hermes registry unavailable — tools not registered")
        return False

    # Registry calls handler(args_dict, **kwargs) — unpack via lambdas.
    # Note: "memory" is intercepted by hermes agent loop; we use "remember" instead.
    tools = [
        (
            "remember", AGENTIC_TOOLSET, _REMEMBER_SCHEMA,
            lambda args, **kw: handle_memory(
                action=args.get("action", ""),
                key=args.get("key", ""),
                value=args.get("value", ""),
            ),
        ),
        (
            "research", AGENTIC_TOOLSET, _RESEARCH_SCHEMA,
            lambda args, **kw: handle_research(
                query=args.get("query", ""),
                max_results=args.get("max_results", 5),
            ),
        ),
        (
            "task_execution", AGENTIC_TOOLSET, _TASK_EXECUTION_SCHEMA,
            lambda args, **kw: handle_task_execution(
                command=args.get("command", ""),
                timeout=args.get("timeout", 30),
            ),
        ),
        # ── Phase 2 RLM tools ────────────────────────────────────────────────
        (
            "python_repl", AGENTIC_TOOLSET, _PYTHON_REPL_SCHEMA,
            lambda args, **kw: handle_python_repl(
                code=args.get("code", ""),
                session_id=args.get("session_id", "default"),
                timeout=args.get("timeout", 30),
            ),
        ),
        # ── DLAM / OpenClaw computer-use ─────────────────────────────────────
        (
            "dlam", AGENTIC_TOOLSET, _DLAM_SCHEMA,
            lambda args, **kw: handle_dlam(
                action=args.get("action", "status"),
                url=args.get("url", ""),
                task=args.get("task", ""),
                query=args.get("query", ""),
                num_results=args.get("num_results", 5),
                timeout=args.get("timeout", 120),
            ),
        ),
        (
            "web_read", AGENTIC_TOOLSET, _WEB_READ_SCHEMA,
            lambda args, **kw: handle_web_read(url=args.get("url", "")),
        ),
        (
            "codebase_search", AGENTIC_TOOLSET, _CODEBASE_SEARCH_SCHEMA,
            lambda args, **kw: handle_codebase_search(
                query=args.get("query", ""),
                top_k=args.get("top_k", 5),
            ),
        ),
        (
            "gguf_infer", AGENTIC_TOOLSET, _GGUF_INFER_SCHEMA,
            lambda args, **kw: handle_gguf_infer(
                model_path=args.get("model_path", ""),
                prompt=args.get("prompt", ""),
                max_tokens=args.get("max_tokens", 256),
                temperature=args.get("temperature", 0.7),
            ),
        ),
        # ── Semantic memory ───────────────────────────────────────────────────
        (
            "search_memory", AGENTIC_TOOLSET, _SEARCH_MEMORY_SCHEMA,
            lambda args, **kw: handle_search_memory(
                query=args.get("query", ""),
                limit=args.get("limit", 8),
                source=args.get("source", ""),
            ),
        ),
        (
            "memory_stats", AGENTIC_TOOLSET, _MEMORY_STATS_SCHEMA,
            lambda args, **kw: handle_memory_stats(),
        ),
        # ── YouTube ───────────────────────────────────────────────────────────
        (
            "youtube_profile", AGENTIC_TOOLSET, _YOUTUBE_PROFILE_SCHEMA,
            lambda args, **kw: handle_youtube_profile(
                action=args.get("action", "show"),
                url=args.get("url", ""),
                patterns=args.get("patterns", ""),
            ),
        ),
        # ── RSS feeds ─────────────────────────────────────────────────────────
        (
            "rss_feed", AGENTIC_TOOLSET, _RSS_FEED_SCHEMA,
            lambda args, **kw: handle_rss_feed(
                action=args.get("action", "show"),
                url=args.get("url", ""),
            ),
        ),
        # ── Podcast ───────────────────────────────────────────────────────────
        (
            "podcast", AGENTIC_TOOLSET, _PODCAST_SCHEMA,
            lambda args, **kw: handle_podcast(
                action=args.get("action", "show"),
                url=args.get("url", ""),
            ),
        ),
        # ── Fabric ────────────────────────────────────────────────────────────
        (
            "fabric", AGENTIC_TOOLSET, _FABRIC_SCHEMA,
            lambda args, **kw: handle_fabric(
                action=args.get("action", "apply"),
                pattern=args.get("pattern", "summarize"),
                text=args.get("text", ""),
                url=args.get("url", ""),
                vendor=args.get("vendor", ""),
                model=args.get("model", ""),
            ),
        ),
        # ── Gmail ─────────────────────────────────────────────────────────────
        (
            "email_profile", AGENTIC_TOOLSET, _EMAIL_PROFILE_SCHEMA,
            lambda args, **kw: handle_email_profile(action=args.get("action", "show")),
        ),
        (
            "check_email", AGENTIC_TOOLSET, _CHECK_EMAIL_SCHEMA,
            lambda args, **kw: handle_check_email(
                action=args.get("action", "search"),
                query=args.get("query", "ALL"),
                folder=args.get("folder", "INBOX"),
                limit=args.get("limit", 10),
                unread_only=args.get("unread_only", False),
                uid=args.get("uid", ""),
            ),
        ),
        # ── OpenClaw filesystem suite ─────────────────────────────────────────
        (
            "read_file", AGENTIC_TOOLSET, _READ_FILE_SCHEMA,
            lambda args, **kw: handle_read_file(
                path=args.get("path", ""),
                offset=args.get("offset", 0),
                limit=args.get("limit", 0),
            ),
        ),
        (
            "write_file", AGENTIC_TOOLSET, _WRITE_FILE_SCHEMA,
            lambda args, **kw: handle_write_file(
                path=args.get("path", ""),
                content=args.get("content", ""),
            ),
        ),
        (
            "edit_file", AGENTIC_TOOLSET, _EDIT_FILE_SCHEMA,
            lambda args, **kw: handle_edit_file(
                path=args.get("path", ""),
                old_string=args.get("old_string", ""),
                new_string=args.get("new_string", ""),
                replace_all=args.get("replace_all", False),
            ),
        ),
        (
            "list_dir", AGENTIC_TOOLSET, _LIST_DIR_SCHEMA,
            lambda args, **kw: handle_list_dir(path=args.get("path", ".")),
        ),
        (
            "glob_files", AGENTIC_TOOLSET, _GLOB_FILES_SCHEMA,
            lambda args, **kw: handle_glob_files(
                pattern=args.get("pattern", ""),
                path=args.get("path", ""),
            ),
        ),
        (
            "grep_files", AGENTIC_TOOLSET, _GREP_FILES_SCHEMA,
            lambda args, **kw: handle_grep_files(
                pattern=args.get("pattern", ""),
                path=args.get("path", ""),
                file_glob=args.get("file_glob", ""),
                context=args.get("context", 0),
                ignore_case=args.get("ignore_case", False),
            ),
        ),
    ]

    for name, toolset, schema, handler in tools:
        # Always register (overwrite) so our handlers win over any built-in
        # tool with the same name that hermes may have registered earlier.
        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            description=schema["description"],
        )
        logger.debug("[agentic_tools] registered: %s", name)

    # Register the toolset availability check (always available)
    if AGENTIC_TOOLSET not in registry._toolset_checks:
        registry._toolset_checks[AGENTIC_TOOLSET] = lambda: True

    logger.info(
        "[agentic_tools] registered: remember / research / task_execution / "
        "python_repl / web_read / codebase_search / gguf_infer / "
        "read_file / write_file / edit_file / list_dir / glob_files / grep_files / "
        "check_email / email_profile / fabric / youtube_profile / "
        "rss_feed / podcast / search_memory / memory_stats / dlam"
    )
    return True
