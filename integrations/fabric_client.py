"""
MYCONEX Fabric Integration
---------------------------
Connects myconex/Buzlock to Daniel Miessler's Fabric framework.
https://github.com/danielmiessler/Fabric

Fabric provides 252+ curated AI patterns (system prompts) that can be applied
to any text, URL, or YouTube video.  This module exposes them as a tool so
Buzlock can call any pattern on demand.

Two backends (tried in order):
  1. REST API  — if `fabric --serve` is running (FABRIC_URL in .env)
  2. Subprocess — calls the `fabric` binary directly (no server needed)

Setup:
  curl -fsSL https://raw.githubusercontent.com/danielmiessler/fabric/main/scripts/installer/install.sh | bash
  fabric --setup      # configure your preferred LLM provider
  # Optional: run as server
  fabric --serve --address :8088 &
  # Add to .env:
  #   FABRIC_URL=http://localhost:8088   (omit to use subprocess mode)
  #   FABRIC_API_KEY=your-key            (only needed if you started with --api-key)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_FABRIC_URL     = os.getenv("FABRIC_URL", "").rstrip("/")
_FABRIC_API_KEY = os.getenv("FABRIC_API_KEY", "")
_FABRIC_TIMEOUT = int(os.getenv("FABRIC_TIMEOUT", "120"))
_FABRIC_VENDOR  = os.getenv("FABRIC_VENDOR", "")   # leave empty to use fabric's default
_FABRIC_MODEL   = os.getenv("FABRIC_MODEL", "")    # leave empty to use fabric's default


# ── REST API client ────────────────────────────────────────────────────────────

async def _rest_apply(
    pattern: str,
    text: str,
    vendor: str = "",
    model: str = "",
) -> str:
    """Call POST /chat on a running Fabric server."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed — pip install aiohttp")

    headers = {"Content-Type": "application/json"}
    if _FABRIC_API_KEY:
        headers["X-API-Key"] = _FABRIC_API_KEY

    payload: dict[str, Any] = {
        "pattern": pattern,
        "message": text,
    }
    if vendor or _FABRIC_VENDOR:
        payload["vendor"] = vendor or _FABRIC_VENDOR
    if model or _FABRIC_MODEL:
        payload["model"] = model or _FABRIC_MODEL

    collected = []
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_FABRIC_URL}/chat",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=_FABRIC_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Fabric server returned {resp.status}: {body[:300]}")
            # SSE stream
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        collected.append(json.loads(chunk))
                    except json.JSONDecodeError:
                        collected.append(chunk)

    # Extract text from SSE chunks (OpenAI delta format)
    parts = []
    for chunk in collected:
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            for c in choices:
                delta = c.get("delta", {})
                if "content" in delta and delta["content"]:
                    parts.append(delta["content"])
        elif isinstance(chunk, str):
            parts.append(chunk)

    return "".join(parts).strip()


async def _rest_list_patterns() -> list[str]:
    """GET /patterns/names from the Fabric server."""
    try:
        import aiohttp
    except ImportError:
        return []

    headers = {}
    if _FABRIC_API_KEY:
        headers["X-API-Key"] = _FABRIC_API_KEY

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_FABRIC_URL}/patterns/names",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return []
            return await resp.json()


async def _rest_youtube(url: str, with_timestamps: bool = False) -> str:
    """POST /youtube/transcript via Fabric server."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed")

    headers = {"Content-Type": "application/json"}
    if _FABRIC_API_KEY:
        headers["X-API-Key"] = _FABRIC_API_KEY

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_FABRIC_URL}/youtube/transcript",
            headers=headers,
            json={"url": url, "with_timestamps": with_timestamps},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Fabric YouTube error {resp.status}: {body[:300]}")
            data = await resp.json()
            return data.get("transcript", str(data))


# ── Subprocess backend ─────────────────────────────────────────────────────────

def _fabric_binary() -> str | None:
    """Return path to the fabric binary, or None if not installed."""
    # Common install locations
    candidates = [
        shutil.which("fabric"),
        os.path.expanduser("~/.local/bin/fabric"),
        os.path.expanduser("~/go/bin/fabric"),
        "/usr/local/bin/fabric",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _sub_apply(pattern: str, text: str) -> str:
    """Run fabric --pattern <pattern> with text piped via stdin."""
    binary = _fabric_binary()
    if not binary:
        raise RuntimeError(
            "fabric binary not found. Install with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/danielmiessler/fabric/"
            "main/scripts/installer/install.sh | bash"
        )
    cmd = [binary, "--pattern", pattern]
    result = subprocess.run(
        cmd,
        input=text,
        capture_output=True,
        text=True,
        timeout=_FABRIC_TIMEOUT,
    )
    if result.returncode != 0 and result.stderr.strip():
        raise RuntimeError(f"fabric subprocess error: {result.stderr.strip()[:400]}")
    return (result.stdout or result.stderr).strip()


def _sub_list_patterns() -> list[str]:
    """Run fabric --list-patterns."""
    binary = _fabric_binary()
    if not binary:
        return []
    try:
        result = subprocess.run(
            [binary, "--list-patterns"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _sub_youtube(url: str, with_timestamps: bool = False) -> str:
    """Run fabric -y <url> to get transcript."""
    binary = _fabric_binary()
    if not binary:
        raise RuntimeError("fabric binary not found")
    cmd = [binary, "-y", url]
    if with_timestamps:
        cmd.append("--transcript-with-timestamps")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    out = result.stdout.strip()
    if not out:
        raise RuntimeError(result.stderr.strip()[:300] or "No transcript returned")
    return out


# ── Public API ────────────────────────────────────────────────────────────────

async def apply_pattern(
    pattern: str,
    text: str,
    vendor: str = "",
    model: str = "",
) -> str:
    """
    Apply a Fabric pattern to text.
    Tries REST API first (if FABRIC_URL set), falls back to subprocess.
    """
    if _FABRIC_URL:
        try:
            return await _rest_apply(pattern, text, vendor, model)
        except Exception as exc:
            logger.warning("Fabric REST failed, falling back to subprocess: %s", exc)
    return _sub_apply(pattern, text)


async def list_patterns() -> list[str]:
    """Return sorted list of available pattern names."""
    if _FABRIC_URL:
        try:
            return sorted(await _rest_list_patterns())
        except Exception:
            pass
    return sorted(_sub_list_patterns())


async def youtube_transcript(url: str, with_timestamps: bool = False) -> str:
    """Extract a YouTube transcript via Fabric."""
    if _FABRIC_URL:
        try:
            return await _rest_youtube(url, with_timestamps)
        except Exception as exc:
            logger.warning("Fabric REST youtube failed, trying subprocess: %s", exc)
    return _sub_youtube(url, with_timestamps)


async def youtube_and_pattern(url: str, pattern: str) -> str:
    """Extract a YouTube transcript then apply a pattern to it."""
    transcript = await youtube_transcript(url)
    if not transcript:
        return "Could not extract transcript."
    return await apply_pattern(pattern, transcript)


def is_available() -> bool:
    """True if fabric is usable (binary found or server reachable)."""
    if _FABRIC_URL:
        return True
    return _fabric_binary() is not None
