"""
MYCONEX Search Provider
------------------------
Unified web-search layer for the agent tool pipeline.

Priority order:
  1. SearXNG  — self-hosted, no key needed  (SEARXNG_URL set)
  2. Brave    — privacy-respecting, API key (BRAVE_API_KEY set)
  3. DuckDuckGo HTML scrape — no key, light fallback

Env vars:
  SEARXNG_URL       — base URL of your SearXNG instance, e.g. http://localhost:8888
  BRAVE_API_KEY     — Brave Search API key
  SEARCH_MAX_RESULTS — max results to return (default: 8)
  SEARCH_TIMEOUT    — request timeout in seconds (default: 10)

Usage:
    from integrations.search_provider import web_search
    results = await web_search("latest AI research 2025")
    # returns list of {"title": ..., "url": ..., "snippet": ...}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_SEARXNG_URL  = os.getenv("SEARXNG_URL", "").rstrip("/")
_BRAVE_KEY    = os.getenv("BRAVE_API_KEY", "")
_MAX_RESULTS  = int(os.getenv("SEARCH_MAX_RESULTS", "8"))
_TIMEOUT      = int(os.getenv("SEARCH_TIMEOUT", "10"))


# ── Public entry point ────────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = _MAX_RESULTS) -> list[dict[str, str]]:
    """
    Search the web and return a list of result dicts.
    Falls through providers in priority order.
    """
    if _SEARXNG_URL:
        try:
            return await _searxng(query, max_results)
        except Exception as exc:
            logger.warning("[search] SearXNG failed (%s), trying next provider", exc)

    if _BRAVE_KEY:
        try:
            return await _brave(query, max_results)
        except Exception as exc:
            logger.warning("[search] Brave failed (%s), falling back to DDG", exc)

    try:
        return await _ddg(query, max_results)
    except Exception as exc:
        logger.warning("[search] DDG fallback failed: %s", exc)
        return []


def format_results(results: list[dict[str, str]], max_chars: int = 2000) -> str:
    """Format search results as a compact string for injection into prompts."""
    if not results:
        return "No search results found."
    lines = []
    total = 0
    for i, r in enumerate(results, 1):
        title   = r.get("title", "")[:80]
        url     = r.get("url", "")
        snippet = r.get("snippet", "")[:200]
        line    = f"{i}. **{title}**\n   {url}\n   {snippet}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


# ── SearXNG ───────────────────────────────────────────────────────────────────

async def _searxng(query: str, n: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({
        "q":      query,
        "format": "json",
        "categories": "general",
    })
    url = f"{_SEARXNG_URL}/search?{params}"

    def _fetch():
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MYCONEX/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read())

    data = await asyncio.get_event_loop().run_in_executor(None, _fetch)
    results = []
    for item in (data.get("results") or [])[:n]:
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("content", ""),
        })
    logger.debug("[search] SearXNG returned %d results", len(results))
    return results


# ── Brave Search ──────────────────────────────────────────────────────────────

async def _brave(query: str, n: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query, "count": str(min(n, 20))})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"

    def _fetch():
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": _BRAVE_KEY,
            },
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            raw = r.read()
            # Brave may gzip-encode even without Accept-Encoding negotiation
            try:
                import gzip
                raw = gzip.decompress(raw)
            except Exception:
                pass
            return json.loads(raw)

    data = await asyncio.get_event_loop().run_in_executor(None, _fetch)
    results = []
    for item in (data.get("web", {}).get("results") or [])[:n]:
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    logger.debug("[search] Brave returned %d results", len(results))
    return results


# ── DuckDuckGo HTML scrape (no-JS endpoint) ───────────────────────────────────

async def _ddg(query: str, n: int) -> list[dict[str, str]]:
    """Scrape DuckDuckGo's lite HTML endpoint — no API key required."""
    import html
    import re

    params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    url = f"https://lite.duckduckgo.com/lite/?{params}"

    def _fetch():
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MYCONEX/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.read().decode("utf-8", errors="replace")

    body = await asyncio.get_event_loop().run_in_executor(None, _fetch)

    # Extract result links + snippets from simple HTML
    results = []
    links   = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', body)
    snips   = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', body, re.DOTALL)

    for i, (url_enc, title) in enumerate(links[:n]):
        # DDG lite wraps URLs
        try:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url_enc).query)
            real_url = qs.get("uddg", [url_enc])[0]
        except Exception:
            real_url = url_enc
        snippet = html.unescape(re.sub(r"<[^>]+>", "", snips[i] if i < len(snips) else "")).strip()
        results.append({
            "title":   html.unescape(title.strip()),
            "url":     real_url,
            "snippet": snippet[:300],
        })

    logger.debug("[search] DDG returned %d results", len(results))
    return results
