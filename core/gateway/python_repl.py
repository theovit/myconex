"""
MYCONEX Persistent Python REPL + Web/Codebase Tools
=====================================================
Phase 2 RLM tools:

  PersistentPythonREPL   — exec Python in a persistent namespace across calls
  REPLPool               — session-keyed pool of REPLs
  WebPageReader          — DOM-as-text web scraper (no screenshots needed)
  CodebaseIndex          — semantic keyword index over the MYCONEX source tree
  GGUFBackend            — llama-cpp-python interface for local GGUF models

All tools are designed to be called from agentic_tools.py handlers and from
RLMAgent.  Every public method is async where I/O is involved.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── REPL Result ──────────────────────────────────────────────────────────────

@dataclass
class REPLResult:
    """Result of a single REPL execution."""
    success: bool
    output: str = ""          # captured stdout
    error: Optional[str] = None
    return_value: Any = None  # value of the last expression, if any
    duration_ms: float = 0.0
    variables: dict = field(default_factory=dict)  # snapshot of namespace (scalars)

    def __str__(self) -> str:
        parts = []
        if self.output:
            parts.append(self.output.strip())
        if self.return_value is not None:
            parts.append(f"→ {self.return_value!r}")
        if self.error:
            parts.append(f"ERROR: {self.error}")
        return "\n".join(parts) if parts else "(no output)"


# ─── Persistent Python REPL ───────────────────────────────────────────────────

class PersistentPythonREPL:
    """
    Execute Python code in a persistent namespace.

    State (variables, imports, functions) survives across multiple execute()
    calls within the same session, mirroring how a human uses a notebook REPL.

    Execution runs in a thread-pool executor to avoid blocking the event loop.
    Stdout/stderr are captured per call.

    Safety: no sandboxing by default — this runs on the agent's host.
    The caller is responsible for scoping what code may be executed.

    Usage:
        repl = PersistentPythonREPL()
        r = await repl.execute("import json; data = json.loads('[1,2,3]')")
        r = await repl.execute("total = sum(data); print(total)")  # state persists
    """

    def __init__(self, session_id: str = "default", timeout: float = 30.0) -> None:
        self.session_id = session_id
        self.timeout = timeout
        self._namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self._lock = asyncio.Lock()
        self._exec_count = 0
        self._created_at = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute(self, code: str, timeout: Optional[float] = None) -> REPLResult:
        """
        Execute a Python code string in the persistent namespace.

        Args:
            code:    Arbitrary Python source.
            timeout: Per-call timeout override (seconds).

        Returns:
            REPLResult with stdout, return_value, error, and namespace snapshot.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self._exec_sync, code),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                result = REPLResult(
                    success=False,
                    error=f"Execution timed out after {effective_timeout}s",
                )
            self._exec_count += 1
            return result

    async def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the persistent namespace."""
        return self._namespace.get(name)

    async def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the persistent namespace."""
        self._namespace[name] = value

    async def reset(self) -> None:
        """Clear namespace, preserving __builtins__."""
        async with self._lock:
            self._namespace.clear()
            self._namespace["__builtins__"] = __builtins__
            self._exec_count = 0

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of scalar namespace values."""
        out = {}
        for k, v in self._namespace.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)   # test serializability
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)[:120]
        return out

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "exec_count": self._exec_count,
            "variables": len([k for k in self._namespace if not k.startswith("_")]),
            "age_s": round(time.time() - self._created_at, 1),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _exec_sync(self, code: str) -> REPLResult:
        """Synchronous execution (runs in thread executor)."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        return_value = None
        start = time.monotonic()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                # Try to eval last expression to capture return value
                try:
                    compiled = compile(code, "<repl>", "eval")
                    return_value = eval(compiled, self._namespace)  # noqa: S307
                except SyntaxError:
                    exec(compile(code, "<repl>", "exec"), self._namespace)  # noqa: S102

            return REPLResult(
                success=True,
                output=stdout_buf.getvalue(),
                error=stderr_buf.getvalue() or None,
                return_value=return_value,
                duration_ms=(time.monotonic() - start) * 1000,
                variables=self.snapshot(),
            )
        except Exception:
            tb = traceback.format_exc()
            return REPLResult(
                success=False,
                output=stdout_buf.getvalue(),
                error=tb.strip(),
                duration_ms=(time.monotonic() - start) * 1000,
            )


# ─── REPL Pool ────────────────────────────────────────────────────────────────

class REPLPool:
    """
    Session-keyed pool of PersistentPythonREPLs.

    Each agent session or task chain gets its own isolated REPL namespace.
    Sessions are lazily created and optionally evicted after a TTL.

    Usage:
        pool = REPLPool()
        result = await pool.execute("session-123", "x = 42")
        result = await pool.execute("session-123", "print(x)")  # → 42
    """

    def __init__(self, session_ttl_s: float = 3600.0) -> None:
        self._sessions: dict[str, PersistentPythonREPL] = {}
        self._last_used: dict[str, float] = {}
        self.session_ttl_s = session_ttl_s

    def get_or_create(self, session_id: str) -> PersistentPythonREPL:
        """Return existing REPL for session_id, or create a new one."""
        self._evict_stale()
        if session_id not in self._sessions:
            self._sessions[session_id] = PersistentPythonREPL(session_id=session_id)
            logger.debug("[repl_pool] created session %s", session_id)
        self._last_used[session_id] = time.time()
        return self._sessions[session_id]

    async def execute(self, session_id: str, code: str, **kwargs) -> REPLResult:
        """Execute code in the given session's REPL."""
        return await self.get_or_create(session_id).execute(code, **kwargs)

    async def reset_session(self, session_id: str) -> None:
        """Clear a session's namespace."""
        if session_id in self._sessions:
            await self._sessions[session_id].reset()

    def drop_session(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._sessions.pop(session_id, None)
        self._last_used.pop(session_id, None)

    def status(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": [s.status() for s in self._sessions.values()],
        }

    def _evict_stale(self) -> None:
        cutoff = time.time() - self.session_ttl_s
        stale = [sid for sid, t in self._last_used.items() if t < cutoff]
        for sid in stale:
            self.drop_session(sid)
            logger.debug("[repl_pool] evicted stale session %s", sid)


# ─── Shared singleton pool ────────────────────────────────────────────────────

REPL_POOL = REPLPool()


# ─── Web Page Reader (DOM-as-text) ────────────────────────────────────────────

@dataclass
class WebPageResult:
    url: str
    title: str = ""
    text: str = ""
    links: list[dict] = field(default_factory=list)  # [{"text": ..., "href": ...}]
    headings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = True


class WebPageReader:
    """
    Fetch a web page and return its structure as plain text.

    No screenshots, no browser — uses httpx + html.parser to extract:
      - Page title
      - Headings (h1-h3)
      - Main text content (stripped of scripts/styles)
      - Links with their anchor text

    Inspired by page-agent.js approach: treat the DOM as a text document,
    not a visual rendering.  Works well for LLM consumption.
    """

    _USER_AGENT = (
        "Mozilla/5.0 (compatible; MYCONEX/1.0; +https://github.com/myconex)"
    )

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    async def read(self, url: str) -> WebPageResult:
        """
        Fetch a URL and extract its text structure.

        Args:
            url: Fully-qualified HTTP/HTTPS URL.

        Returns:
            WebPageResult with title, text, links, and headings.
        """
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self._USER_AGENT},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return self._parse(url, resp.text)
        except Exception as exc:
            logger.warning("[web_reader] failed to fetch %s: %s", url, exc)
            return WebPageResult(url=url, error=str(exc), success=False)

    def _parse(self, url: str, html: str) -> WebPageResult:
        from html.parser import HTMLParser

        class _Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.title = ""
                self.headings: list[str] = []
                self.links: list[dict] = []
                self.text_parts: list[str] = []
                self._in_skip = False
                self._skip_tags = {"script", "style", "nav", "footer", "head"}
                self._current_tag = ""
                self._current_href = ""
                self._current_text_parts: list[str] = []

            def handle_starttag(self, tag, attrs):
                self._current_tag = tag
                if tag in self._skip_tags:
                    self._in_skip = True
                if tag == "a":
                    attrs_d = dict(attrs)
                    self._current_href = attrs_d.get("href", "")
                    self._current_text_parts = []

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._in_skip = False
                if tag == "a" and self._current_href:
                    link_text = "".join(self._current_text_parts).strip()
                    if link_text:
                        self.links.append({"text": link_text, "href": self._current_href})
                    self._current_href = ""

            def handle_data(self, data):
                if self._in_skip:
                    return
                stripped = data.strip()
                if not stripped:
                    return
                if self._current_tag == "title":
                    self.title = stripped
                elif self._current_tag in ("h1", "h2", "h3"):
                    self.headings.append(f"{'#' * int(self._current_tag[1])} {stripped}")
                if self._current_href:
                    self._current_text_parts.append(stripped)
                self.text_parts.append(stripped)

        parser = _Parser()
        parser.feed(html)

        raw_text = " ".join(parser.text_parts)
        # Collapse whitespace and truncate
        clean_text = re.sub(r"\s+", " ", raw_text).strip()[:8000]

        return WebPageResult(
            url=url,
            title=parser.title,
            text=clean_text,
            links=parser.links[:50],   # cap at 50 links
            headings=parser.headings[:30],
            success=True,
        )

    def format_for_llm(self, result: WebPageResult) -> str:
        """Format a WebPageResult as compact LLM-readable text."""
        if not result.success:
            return f"[web_read error: {result.error}]"
        parts = [f"URL: {result.url}"]
        if result.title:
            parts.append(f"Title: {result.title}")
        if result.headings:
            parts.append("Headings:\n" + "\n".join(result.headings[:10]))
        if result.text:
            parts.append(f"Content:\n{result.text[:3000]}")
        if result.links:
            link_strs = [f"  [{l['text'][:60]}]({l['href']})" for l in result.links[:15]]
            parts.append("Links:\n" + "\n".join(link_strs))
        return "\n\n".join(parts)


# ─── Codebase Index ───────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    content: str
    keywords: set[str] = field(default_factory=set)


class CodebaseIndex:
    """
    Semantic keyword index over the MYCONEX source tree.

    Gives the RLMAgent self-awareness: it can search its own codebase for
    relevant code chunks without needing an external vector database.

    Inspired by SemanticCode: build an in-memory inverted index of keywords
    → code chunks, then rank by keyword overlap for a query.

    Usage:
        idx = CodebaseIndex("/home/techno-shaman/myconex")
        idx.build()
        results = idx.search("delegate complexity routing")
    """

    _SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache"}
    _EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".toml", ".json"}
    _CHUNK_LINES = 40     # lines per chunk
    _CHUNK_OVERLAP = 5    # overlap between consecutive chunks

    def __init__(self, root: str | Path = "/home/techno-shaman/myconex") -> None:
        self.root = Path(root)
        self._chunks: list[CodeChunk] = []
        self._index: dict[str, list[int]] = {}   # keyword → chunk indices
        self._built = False

    def build(self) -> int:
        """
        Walk root and index all matching files.

        Returns:
            Number of chunks indexed.
        """
        self._chunks.clear()
        self._index.clear()

        for path in self._iter_files():
            try:
                text = path.read_text(errors="replace")
                self._index_file(str(path.relative_to(self.root)), text)
            except Exception as exc:
                logger.debug("[codebase_index] skip %s: %s", path, exc)

        self._built = True
        logger.info(
            "[codebase_index] built: %d chunks, %d keywords from %s",
            len(self._chunks), len(self._index), self.root,
        )
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Keyword search over indexed chunks.

        Args:
            query: Free-text query string.
            top_k: Maximum results to return.

        Returns:
            List of dicts with file_path, start_line, content, score.
        """
        if not self._built:
            self.build()

        query_kw = self._tokenize(query)
        scores: dict[int, int] = {}
        for kw in query_kw:
            for idx in self._index.get(kw, []):
                scores[idx] = scores.get(idx, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_idx, score in ranked:
            chunk = self._chunks[chunk_idx]
            results.append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "content": chunk.content,
                "score": score,
            })
        return results

    def status(self) -> dict:
        return {
            "built": self._built,
            "chunks": len(self._chunks),
            "keywords": len(self._index),
            "root": str(self.root),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _iter_files(self):
        for path in self.root.rglob("*"):
            if any(part in self._SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix in self._EXTENSIONS:
                yield path

    def _index_file(self, rel_path: str, text: str) -> None:
        lines = text.splitlines()
        step = self._CHUNK_LINES - self._CHUNK_OVERLAP
        for start in range(0, max(1, len(lines)), step):
            chunk_lines = lines[start: start + self._CHUNK_LINES]
            content = "\n".join(chunk_lines)
            keywords = self._tokenize(content)
            chunk = CodeChunk(
                file_path=rel_path,
                start_line=start + 1,
                content=content,
                keywords=keywords,
            )
            idx = len(self._chunks)
            self._chunks.append(chunk)
            for kw in keywords:
                self._index.setdefault(kw, []).append(idx)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
        stop = {"def", "class", "import", "from", "return", "self", "for", "the",
                "and", "not", "with", "None", "True", "False", "async", "await"}
        return {w.lower() for w in words if w not in stop and len(w) < 40}


# ─── Shared codebase index singleton ─────────────────────────────────────────

_CODEBASE_INDEX: Optional[CodebaseIndex] = None


def get_codebase_index(root: str = "/home/techno-shaman/myconex") -> CodebaseIndex:
    global _CODEBASE_INDEX
    if _CODEBASE_INDEX is None:
        _CODEBASE_INDEX = CodebaseIndex(root)
    return _CODEBASE_INDEX


# ─── GGUF Backend (llama-cpp-python) ─────────────────────────────────────────

class GGUFBackend:
    """
    Local GGUF model inference via llama-cpp-python.

    Loads a GGUF file into memory and runs generation.  The model is loaded
    lazily on first call.  If llama-cpp-python is not installed this class
    returns helpful error messages rather than raising at import time.

    Usage:
        backend = GGUFBackend("/path/to/model.gguf")
        result = await backend.generate("What is a mesh network?")
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,    # -1 = offload all layers to GPU
        n_threads: Optional[int] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count() or 4
        self._llm: Any = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """True when the model file exists and llama-cpp-python is importable."""
        if not self.model_path.is_file():
            return False
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a completion for prompt.

        Args:
            prompt:     Raw text prompt (not chat messages).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop:       Optional stop sequences.

        Returns:
            Generated text string.

        Raises:
            RuntimeError if llama-cpp-python unavailable or model not found.
        """
        if not self.available:
            if not self.model_path.is_file():
                raise RuntimeError(f"GGUF model not found: {self.model_path}")
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._generate_sync, prompt, max_tokens, temperature, stop or []
            )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Chat completion via llama-cpp-python (requires a chat-tuned model)."""
        if not self.available:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._chat_sync, messages, max_tokens, temperature
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama  # type: ignore[import]
        logger.info("[gguf] loading model from %s", self.model_path)
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False,
        )
        logger.info("[gguf] model loaded: %s", self.model_path.name)

    def _generate_sync(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str]
    ) -> str:
        self._load_model()
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
        )
        return out["choices"][0]["text"].strip()

    def _chat_sync(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        self._load_model()
        out = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()


# ─── BitNet 1-bit Backend ─────────────────────────────────────────────────────

class BitNet1BitBackend:
    """
    Ultra-low-memory inference via Microsoft BitNet (1-bit weights).

    BitNet models quantise weights to ±1 (1.58-bit effective), achieving
    ~8× memory reduction versus fp16 — ideal for edge nodes (T3/T4).

    Two execution paths:
      1. bitnet-cpp CLI binary  — official Microsoft release; fastest
      2. llama-cpp-python       — if built with BitNet support (--bitnet flag)

    Build bitnet-cpp:
        git clone https://github.com/microsoft/BitNet
        cd BitNet && cmake -B build -DBITNET=ON && cmake --build build -j

    Usage:
        backend = BitNet1BitBackend(
            model_path="/models/bitnet-b1.58-3B.gguf",
            binary_path="/opt/bitnet-cpp/build/bin/bitnet-cli",
        )
        response = await backend.generate("What is distributed AI?")
    """

    # Default install paths
    _DEFAULT_BINARY = Path("/opt/bitnet-cpp/build/bin/bitnet-cli")
    _ALT_BINARY = Path.home() / ".local" / "bin" / "bitnet-cli"

    def __init__(
        self,
        model_path: str | Path,
        binary_path: Optional[str | Path] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_llamacpp: bool = False,   # prefer llama-cpp-python if True
        timeout: float = 60.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.binary_path: Optional[Path] = (
            Path(binary_path) if binary_path
            else self._find_binary()
        )
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self.use_llamacpp = use_llamacpp
        self.timeout = timeout
        self._llm: Any = None   # llama-cpp-python Llama instance (lazy)
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """True when the model file exists and at least one execution path works."""
        if not self.model_path.is_file():
            return False
        if self.binary_path and self.binary_path.is_file():
            return True
        # Check llama-cpp-python
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def backend_type(self) -> str:
        """Which execution path will be used."""
        if not self.use_llamacpp and self.binary_path and self.binary_path.is_file():
            return "bitnet-cpp"
        try:
            import llama_cpp  # noqa: F401
            return "llama-cpp-python"
        except ImportError:
            return "unavailable"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text from a raw prompt.

        Args:
            prompt:      Text prompt.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature.
            stop:        Optional stop sequences (supported by llama-cpp path).

        Returns:
            Generated text string.

        Raises:
            RuntimeError if no execution path is available.
        """
        if not self.available:
            if not self.model_path.is_file():
                raise RuntimeError(f"BitNet model not found: {self.model_path}")
            raise RuntimeError(
                "No BitNet execution path available.\n"
                "Option 1: Build bitnet-cpp: https://github.com/microsoft/BitNet\n"
                "Option 2: Install llama-cpp-python with BitNet support:\n"
                "  CMAKE_ARGS='-DBITNET=ON' pip install llama-cpp-python"
            )

        backend = self.backend_type
        if backend == "bitnet-cpp":
            return await self._generate_cli(prompt, max_tokens, temperature)
        elif backend == "llama-cpp-python":
            async with self._lock:
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._generate_llamacpp, prompt, max_tokens, temperature, stop or []
                    ),
                    timeout=self.timeout,
                )
        raise RuntimeError(f"No available backend (backend_type={backend!r})")

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion using ChatML format.

        Converts messages to ChatML prompt string and calls generate().
        """
        prompt = self._messages_to_chatml(messages)
        result = await self.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                     stop=["<|im_end|>", "<|im_start|>"])
        return result.split("<|im_end|>")[0].strip()

    def status(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_file(),
            "binary_path": str(self.binary_path) if self.binary_path else None,
            "binary_exists": self.binary_path.is_file() if self.binary_path else False,
            "available": self.available,
            "backend": self.backend_type,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_binary(self) -> Optional[Path]:
        for candidate in (self._DEFAULT_BINARY, self._ALT_BINARY):
            if candidate.is_file():
                return candidate
        # Check PATH
        import shutil
        found = shutil.which("bitnet-cli") or shutil.which("bitnet")
        return Path(found) if found else None

    async def _generate_cli(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run inference via the bitnet-cpp CLI binary."""
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-t", str(self.n_threads),
            "--ctx-size", str(self.n_ctx),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"BitNet CLI timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"bitnet-cli exited {proc.returncode}: {stderr.decode()[:300]}"
            )
        return stdout.decode().strip()

    def _generate_llamacpp(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str]
    ) -> str:
        """Run inference via llama-cpp-python (must be built with BitNet support)."""
        if self._llm is None:
            from llama_cpp import Llama  # type: ignore[import]
            logger.info("[bitnet] loading model via llama-cpp-python: %s", self.model_path)
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
        )
        return out["choices"][0]["text"].strip()

    @staticmethod
    def _messages_to_chatml(messages: list[dict]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)
