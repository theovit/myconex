"""
MYCONEX Multi-Source Intelligence Aggregator
=============================================
Inspired by Crucix.

Aggregates data from multiple heterogeneous sources in parallel and synthesises
a unified intelligence report.  Designed for agent use: a single call collects
web pages, API responses, local files, RSS feeds, and database queries, then
merges them into a structured IntelReport.

Supported source types:
  web      — fetch URL and extract text (reuses WebPageReader)
  api      — HTTP GET/POST to a JSON API endpoint
  file     — read a local file (text, JSON, CSV, markdown)
  rss      — parse RSS/Atom feed
  search   — DuckDuckGo text search
  db       — SQLite query (lightweight, no server needed)
  shell    — run a shell command and capture stdout

Usage:
    agg = IntelAggregator()
    report = await agg.gather([
        IntelSource("hacker-news-rss", "rss", {"url": "https://news.ycombinator.com/rss"}),
        IntelSource("local-logs", "file", {"path": "/var/log/myconex.log", "tail": 200}),
        IntelSource("mesh-status", "api", {"url": "http://localhost:8765/status"}),
    ])
    print(report.to_markdown())
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ─── Source Definition ────────────────────────────────────────────────────────

@dataclass
class IntelSource:
    """
    Configuration for a single data source.

    Args:
        name:    Human-readable label for this source.
        type:    Source type: "web"|"api"|"file"|"rss"|"search"|"db"|"shell"
        config:  Type-specific configuration dict (see individual gatherers).
        timeout: Per-source timeout in seconds.
        weight:  Relevance weight for synthesis (0.0–1.0).
    """
    name: str
    type: str
    config: dict = field(default_factory=dict)
    timeout: float = 20.0
    weight: float = 1.0


# ─── Source Result ────────────────────────────────────────────────────────────

@dataclass
class SourceResult:
    """Result from a single source gather."""
    source_name: str
    source_type: str
    success: bool
    content: str = ""
    structured: Any = None    # parsed JSON/list when available
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def excerpt(self, max_chars: int = 500) -> str:
        return self.content[:max_chars] + ("…" if len(self.content) > max_chars else "")


# ─── Intelligence Report ──────────────────────────────────────────────────────

@dataclass
class IntelReport:
    """Aggregated intelligence from multiple sources."""
    query: str = ""
    sources_requested: int = 0
    sources_succeeded: int = 0
    results: list[SourceResult] = field(default_factory=list)
    synthesis: str = ""            # LLM synthesis (populated by RLMAgent)
    generated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = []
        if self.query:
            lines.append(f"# Intelligence Report: {self.query}\n")
        lines.append(
            f"*{self.sources_succeeded}/{self.sources_requested} sources succeeded "
            f"— {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.generated_at))}*\n"
        )
        if self.synthesis:
            lines.append(f"## Synthesis\n\n{self.synthesis}\n")
        lines.append("## Source Details\n")
        for r in self.results:
            status = "✅" if r.success else "❌"
            lines.append(f"### {status} {r.source_name} ({r.source_type})")
            if r.success:
                lines.append(r.excerpt(1000))
            else:
                lines.append(f"*Error: {r.error}*")
            lines.append("")
        return "\n".join(lines).strip()

    def to_agent_payload(self, max_chars: int = 12000) -> dict:
        """Compact payload suitable for agent context injection."""
        combined = "\n\n---\n\n".join(
            f"[{r.source_name}]\n{r.excerpt(2000)}"
            for r in self.results if r.success
        )
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n[...truncated]"
        return {
            "query": self.query,
            "sources": self.sources_succeeded,
            "content": combined,
            "synthesis": self.synthesis,
        }

    def failed_sources(self) -> list[str]:
        return [r.source_name for r in self.results if not r.success]


# ─── Individual Gatherers ─────────────────────────────────────────────────────

class WebGatherer:
    """Fetch a URL and return text content."""

    async def gather(self, source: IntelSource) -> SourceResult:
        from core.gateway.python_repl import WebPageReader
        url = source.config.get("url", "")
        if not url:
            return SourceResult(source.name, "web", False, error="No url in config")
        start = time.monotonic()
        try:
            reader = WebPageReader(timeout=source.timeout)
            result = await reader.read(url)
            content = reader.format_for_llm(result)
            return SourceResult(
                source_name=source.name, source_type="web",
                success=result.success,
                content=content,
                error=result.error,
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"title": result.title, "url": url},
            )
        except Exception as exc:
            return SourceResult(source.name, "web", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)


class APIGatherer:
    """Make an HTTP API call and return JSON or text response."""

    async def gather(self, source: IntelSource) -> SourceResult:
        cfg = source.config
        url = cfg.get("url", "")
        method = cfg.get("method", "GET").upper()
        headers = cfg.get("headers", {})
        body = cfg.get("body") or cfg.get("json")
        params = cfg.get("params", {})

        if not url:
            return SourceResult(source.name, "api", False, error="No url in config")

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=source.timeout) as client:
                if method == "POST":
                    resp = await client.post(url, json=body, headers=headers, params=params)
                else:
                    resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()

                # Try JSON first, fall back to text
                try:
                    data = resp.json()
                    content = json.dumps(data, indent=2)[:8000]
                    structured = data
                except Exception:
                    content = resp.text[:8000]
                    structured = None

                return SourceResult(
                    source_name=source.name, source_type="api",
                    success=True, content=content, structured=structured,
                    duration_ms=(time.monotonic() - start) * 1000,
                    metadata={"status_code": resp.status_code, "url": url},
                )
        except Exception as exc:
            return SourceResult(source.name, "api", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)


class FileGatherer:
    """Read a local file (text, JSON, CSV, markdown)."""

    async def gather(self, source: IntelSource) -> SourceResult:
        cfg = source.config
        path = Path(cfg.get("path", ""))
        tail_lines = cfg.get("tail", 0)
        head_lines = cfg.get("head", 0)

        start = time.monotonic()
        try:
            if not path.is_file():
                return SourceResult(source.name, "file", False,
                                    error=f"File not found: {path}")

            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, path.read_text, "utf-8")

            # Apply head/tail filters
            if tail_lines:
                lines = content.splitlines()
                content = "\n".join(lines[-tail_lines:])
            elif head_lines:
                lines = content.splitlines()
                content = "\n".join(lines[:head_lines])

            # Parse JSON if applicable
            structured = None
            if path.suffix == ".json":
                try:
                    structured = json.loads(content)
                except Exception:
                    pass

            return SourceResult(
                source_name=source.name, source_type="file",
                success=True, content=content[:10000], structured=structured,
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"path": str(path), "size": path.stat().st_size},
            )
        except Exception as exc:
            return SourceResult(source.name, "file", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)


class RSSGatherer:
    """Parse an RSS or Atom feed."""

    async def gather(self, source: IntelSource) -> SourceResult:
        url = source.config.get("url", "")
        max_items = source.config.get("max_items", 10)

        if not url:
            return SourceResult(source.name, "rss", False, error="No url in config")

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=source.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                xml = resp.text

            items = self._parse_feed(xml, max_items)
            content = "\n\n".join(
                f"**{i.get('title', 'No title')}**\n{i.get('link', '')}\n{i.get('summary', '')[:300]}"
                for i in items
            )
            return SourceResult(
                source_name=source.name, source_type="rss",
                success=True, content=content, structured=items,
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"items": len(items), "url": url},
            )
        except Exception as exc:
            return SourceResult(source.name, "rss", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)

    @staticmethod
    def _parse_feed(xml: str, max_items: int) -> list[dict]:
        import xml.etree.ElementTree as ET
        items: list[dict] = []
        try:
            root = ET.fromstring(xml)
            # Handle both RSS and Atom
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            # RSS
            for item in root.iter("item"):
                items.append({
                    "title": (item.findtext("title") or "").strip(),
                    "link": (item.findtext("link") or "").strip(),
                    "summary": (item.findtext("description") or "").strip()[:500],
                    "pubdate": item.findtext("pubDate") or "",
                })
                if len(items) >= max_items:
                    break
            # Atom (if no RSS items found)
            if not items:
                for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                    items.append({
                        "title": (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip(),
                        "link": link_el.get("href", "") if link_el is not None else "",
                        "summary": (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()[:500],
                    })
                    if len(items) >= max_items:
                        break
        except ET.ParseError:
            pass
        return items[:max_items]


class SearchGatherer:
    """DuckDuckGo text search."""

    async def gather(self, source: IntelSource) -> SourceResult:
        query = source.config.get("query", "")
        max_results = source.config.get("max_results", 8)

        if not query:
            return SourceResult(source.name, "search", False, error="No query in config")

        start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self._ddg_search, query, max_results)
            content = "\n\n".join(
                f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')[:300]}"
                for r in results
            )
            return SourceResult(
                source_name=source.name, source_type="search",
                success=True, content=content, structured=results,
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"query": query, "results": len(results)},
            )
        except Exception as exc:
            return SourceResult(source.name, "search", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)

    @staticmethod
    def _ddg_search(query: str, max_results: int) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))


class DBGatherer:
    """Run a SQLite query."""

    async def gather(self, source: IntelSource) -> SourceResult:
        db_path = source.config.get("db", ":memory:")
        query = source.config.get("query", "")
        params = source.config.get("params", [])

        if not query:
            return SourceResult(source.name, "db", False, error="No query in config")

        start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            rows, columns = await loop.run_in_executor(
                None, self._run_query, db_path, query, params
            )
            if not columns:
                return SourceResult(source.name, "db", True,
                                    content="Query returned no columns.",
                                    duration_ms=(time.monotonic() - start) * 1000)

            # Format as markdown table
            header = " | ".join(columns)
            sep = " | ".join(["---"] * len(columns))
            data_rows = [" | ".join(str(v) for v in row) for row in rows[:100]]
            content = "\n".join([header, sep] + data_rows)
            return SourceResult(
                source_name=source.name, source_type="db",
                success=True, content=content,
                structured={"columns": columns, "rows": [list(r) for r in rows[:100]]},
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"db": db_path, "rows_returned": len(rows)},
            )
        except Exception as exc:
            return SourceResult(source.name, "db", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)

    @staticmethod
    def _run_query(db_path: str, query: str, params: list) -> tuple[list, list]:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(query, params)
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
            return rows, columns
        finally:
            conn.close()


class ShellGatherer:
    """Run a shell command and capture stdout."""

    async def gather(self, source: IntelSource) -> SourceResult:
        command = source.config.get("command", "")
        if not command:
            return SourceResult(source.name, "shell", False, error="No command in config")

        start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            stdout, stderr, rc = await loop.run_in_executor(
                None, self._run, command, int(source.timeout)
            )
            content = stdout[:8000]
            if stderr.strip():
                content += f"\nstderr: {stderr[:500]}"
            return SourceResult(
                source_name=source.name, source_type="shell",
                success=(rc == 0),
                content=content,
                error=f"exit {rc}" if rc != 0 else None,
                duration_ms=(time.monotonic() - start) * 1000,
                metadata={"command": command, "return_code": rc},
            )
        except Exception as exc:
            return SourceResult(source.name, "shell", False,
                                error=str(exc),
                                duration_ms=(time.monotonic() - start) * 1000)

    @staticmethod
    def _run(command: str, timeout: int) -> tuple[str, str, int]:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode


# ─── Intelligence Aggregator ──────────────────────────────────────────────────

class IntelAggregator:
    """
    Parallel multi-source intelligence aggregator.

    Dispatches all sources concurrently, collects results, and optionally
    synthesises them into a narrative report via an injected LLM callable.

    Usage:
        agg = IntelAggregator()
        report = await agg.gather(sources, query="quarterly threat summary")
        report.synthesis = await llm_call(report.to_agent_payload()["content"])
    """

    _GATHERERS: dict[str, Any] = {}

    def __init__(self) -> None:
        self._gatherers = {
            "web":    WebGatherer(),
            "api":    APIGatherer(),
            "file":   FileGatherer(),
            "rss":    RSSGatherer(),
            "search": SearchGatherer(),
            "db":     DBGatherer(),
            "shell":  ShellGatherer(),
        }

    async def gather(
        self,
        sources: list[IntelSource],
        query: str = "",
        deduplicate: bool = True,
    ) -> IntelReport:
        """
        Gather intelligence from all sources in parallel.

        Args:
            sources:      List of IntelSource definitions.
            query:        Optional query label for the report.
            deduplicate:  Remove near-duplicate content across sources.

        Returns:
            IntelReport with all gathered results.
        """
        if not sources:
            return IntelReport(query=query, sources_requested=0)

        tasks = [self._gather_one(src) for src in sources]
        results: list[SourceResult] = await asyncio.gather(*tasks, return_exceptions=False)

        if deduplicate:
            results = self._deduplicate(results)

        succeeded = sum(1 for r in results if r.success)
        return IntelReport(
            query=query,
            sources_requested=len(sources),
            sources_succeeded=succeeded,
            results=results,
        )

    async def gather_with_synthesis(
        self,
        sources: list[IntelSource],
        query: str,
        llm_fn,    # async callable(prompt: str) -> str
    ) -> IntelReport:
        """
        Gather + synthesise.  llm_fn is called with the aggregated content.

        Args:
            sources: Data sources to gather from.
            query:   Research question or topic.
            llm_fn:  Async function that takes a prompt and returns a string.

        Returns:
            IntelReport with populated .synthesis field.
        """
        report = await self.gather(sources, query=query)
        if report.sources_succeeded == 0:
            report.synthesis = "No sources returned data."
            return report

        payload = report.to_agent_payload()
        synthesis_prompt = (
            f"Research question: {query}\n\n"
            f"Data from {report.sources_succeeded} sources:\n\n"
            f"{payload['content'][:6000]}\n\n"
            "Synthesise the above into a concise intelligence report. "
            "Highlight key findings, discrepancies between sources, and actionable insights. "
            "Use markdown formatting."
        )
        try:
            report.synthesis = await llm_fn(synthesis_prompt)
        except Exception as exc:
            logger.warning("[intel_agg] synthesis failed: %s", exc)
            report.synthesis = f"Synthesis unavailable: {exc}"

        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _gather_one(self, source: IntelSource) -> SourceResult:
        gatherer = self._gatherers.get(source.type)
        if gatherer is None:
            return SourceResult(
                source.name, source.type, False,
                error=f"Unknown source type: {source.type!r}. "
                      f"Supported: {list(self._gatherers)}",
            )
        try:
            return await asyncio.wait_for(
                gatherer.gather(source), timeout=source.timeout + 5
            )
        except asyncio.TimeoutError:
            return SourceResult(
                source.name, source.type, False,
                error=f"Source timed out after {source.timeout}s",
            )
        except Exception as exc:
            return SourceResult(source.name, source.type, False, error=str(exc))

    @staticmethod
    def _deduplicate(results: list[SourceResult]) -> list[SourceResult]:
        """Remove results with >80% content overlap (simple fingerprint check)."""
        seen_hashes: set[str] = set()
        unique: list[SourceResult] = []
        for r in results:
            if not r.success or not r.content:
                unique.append(r)
                continue
            # Fingerprint: first 200 + last 200 chars
            fp_text = r.content[:200] + r.content[-200:]
            fp = re.sub(r"\s+", " ", fp_text).strip()[:300]
            import hashlib
            h = hashlib.md5(fp.encode()).hexdigest()[:8]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(r)
            else:
                logger.debug("[intel_agg] deduplicated source: %s", r.source_name)
        return unique
