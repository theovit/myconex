"""
MYCONEX DLAM Client
--------------------
Connects to the DLAM webpage (https://dlam.rabbit.tech/) running in your
Brave browser session via Playwright CDP, types tasks into the text bar,
and returns the response.

This module is fully self-healing:
  - If Brave is not running, it launches it with --remote-debugging-port=9222
  - If the DLAM tab is not open, it opens it
  - If the R1 device is disconnected, it clicks the bottom icons to connect
    and handles the Brave screen-share permission dialog automatically

Public API:
    from integrations.dlam_client import dlam_browse, dlam_search, dlam_task, dlam_status

    result = await dlam_task("search arxiv for 1-bit LLM papers from 2025")
    result = await dlam_browse("https://arxiv.org/abs/...", "summarise the abstract")
    result = await dlam_search("latest BitNet quantization benchmarks")
    info   = await dlam_status()

Fallback:
    If Brave fails to start or CDP is unreachable, dlam_task/dlam_browse
    falls back to a plain HTTP fetch + text extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_CDP_URL            = "http://localhost:9222"
_DLAM_URL           = "https://dlam.rabbit.tech/"
_TASK_DIR           = Path.home() / ".buzlock" / "cua_tasks"
_PENDING            = _TASK_DIR / "pending"
_DEFAULT_TIMEOUT    = 120     # seconds to wait for DLAM response
_RESPONSE_POLL_MS   = 500     # ms between DOM polls while waiting for response
_RECONNECT_DELAY    = 3.0

# Brave binary and launch config
_BRAVE_BIN          = "/opt/brave.com/brave/brave"
_BRAVE_LAUNCH_WAIT  = 8.0    # max seconds to wait for CDP to become ready after launch

# R1 connection flow (screen coordinates, determined empirically)
_ICON_COORDS        = [(367, 921), (438, 921), (510, 921)]  # keyboard / mic / monitor
_DIALOG_FOCUS_XY    = (250, 200)   # click here to focus the screen-share dialog
_DIALOG_TAB_N       = 3            # Tab presses to reach the Share button
_CONNECT_TIMEOUT    = 20.0         # seconds to wait for "r1 connected" after clicking icons
_CONNECT_POLL       = 0.5

# Exact selector for the DLAM text bar (from live DOM inspection)
_INPUT_SELECTOR = "textarea.user-messages-scroll"
# Fallback selectors if the class name ever changes
_INPUT_SELECTORS_FALLBACK = [
    "textarea",
    "[contenteditable='true']",
    "[role='textbox']",
]

# Module-level lock so concurrent callers don't double-click connect icons
_CONNECT_LOCK: asyncio.Lock | None = None


def _get_connect_lock() -> asyncio.Lock:
    global _CONNECT_LOCK
    if _CONNECT_LOCK is None:
        _CONNECT_LOCK = asyncio.Lock()
    return _CONNECT_LOCK


# ── Brave launcher ─────────────────────────────────────────────────────────────

def _launch_brave_sync() -> None:
    """
    Launch Brave with --remote-debugging-port=9222 in the background.
    Blocks until the CDP port responds (up to _BRAVE_LAUNCH_WAIT seconds).
    """
    import urllib.request as _ur

    env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}
    logger.info("[dlam] launching Brave: %s", _BRAVE_BIN)
    subprocess.Popen(
        [_BRAVE_BIN, "--remote-debugging-port=9222", "--restore-last-session"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )

    deadline = time.monotonic() + _BRAVE_LAUNCH_WAIT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            with _ur.urlopen(f"{_CDP_URL}/json/version", timeout=1) as r:
                if r.status == 200:
                    logger.info("[dlam] Brave CDP ready")
                    return
        except Exception:
            pass
    logger.warning("[dlam] Brave did not become ready within %.0fs", _BRAVE_LAUNCH_WAIT)


async def _ensure_brave_running() -> None:
    """Launch Brave (with CDP) if it's not already running."""
    if await _cdp_available():
        logger.debug("[dlam] Brave already running with CDP")
        return
    logger.info("[dlam] Brave not running — launching now")
    await asyncio.to_thread(_launch_brave_sync)


# ── Connection helpers ──────────────────────────────────────────────────────────

def _dismiss_screen_share_dialog_sync() -> None:
    """
    Handle the Brave native screen-share picker that appears when DLAM
    requests screen access.  Uses pyautogui to:
      1. Click the dialog body to give it focus
      2. Tab to the Share button
      3. Press Enter to confirm

    This is always attempted after clicking the connect icons — it's a no-op
    if no dialog is present.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        logger.info("[dlam] attempting to dismiss screen-share dialog")
        pyautogui.click(_DIALOG_FOCUS_XY[0], _DIALOG_FOCUS_XY[1])
        time.sleep(0.5)
        for _ in range(_DIALOG_TAB_N):
            pyautogui.press("tab")
            time.sleep(0.08)
        time.sleep(0.2)
        pyautogui.press("enter")
        logger.info("[dlam] screen-share dialog dismissed (Tab x%d → Enter)", _DIALOG_TAB_N)
    except ImportError:
        logger.warning("[dlam] pyautogui not available — cannot dismiss screen-share dialog")
    except Exception as exc:
        logger.warning("[dlam] screen-share dialog dismiss error: %s", exc)


def _click_connect_icons_sync() -> None:
    """Click the three bottom icons on the DLAM page to initiate R1 connection."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        for x, y in _ICON_COORDS:
            logger.info("[dlam] clicking connect icon at screen (%d, %d)", x, y)
            pyautogui.click(x, y)
            time.sleep(0.4)
    except ImportError:
        logger.warning("[dlam] pyautogui not available — cannot click connect icons")
    except Exception as exc:
        logger.warning("[dlam] connect icon click error: %s", exc)


# ── Core helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in [_PENDING, _TASK_DIR / "processing",
              _TASK_DIR / "completed", _TASK_DIR / "failed"]:
        d.mkdir(parents=True, exist_ok=True)


def _write_task(task: dict) -> None:
    _ensure_dirs()
    tid = task.setdefault("id", str(uuid.uuid4())[:8])
    (_PENDING / f"{tid}.json").write_text(json.dumps(task, indent=2))


async def _cdp_available() -> bool:
    """Return True if Brave is running with remote debugging on port 9222."""
    import urllib.request as _ur
    try:
        with _ur.urlopen(f"{_CDP_URL}/json/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


async def _get_dlam_page(playwright):
    """
    Attach to the running Brave instance via CDP and return the DLAM page.
    Opens the DLAM tab if it's not already present.
    """
    browser = await playwright.chromium.connect_over_cdp(_CDP_URL)
    for context in browser.contexts:
        for page in context.pages:
            if "dlam.rabbit.tech" in page.url:
                logger.debug("[dlam] found existing DLAM tab")
                return browser, page

    logger.info("[dlam] DLAM tab not found — opening %s", _DLAM_URL)
    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    page = await context.new_page()
    await page.goto(_DLAM_URL, timeout=20000)
    await page.wait_for_load_state("domcontentloaded", timeout=15000)
    logger.info("[dlam] DLAM tab opened")
    return browser, page


async def _check_connected(page) -> bool:
    """
    Return True if DLAM is usable (R1 device is present).
    'voice only' (no screen share) is acceptable for text tasks.
    Only 'disconnected' means the R1 is not plugged in.
    """
    try:
        text = await page.evaluate("() => document.body.innerText")
        return "disconnected" not in text.lower()
    except Exception:
        return False


async def _attempt_connect(page) -> bool:
    """
    Full R1 connection flow:
      1. No-op if already connected
      2. Click the three bottom icons via pyautogui (screen coords)
      3. Wait briefly, then dismiss any screen-share dialog via pyautogui
      4. Poll for "r1 connected" status up to _CONNECT_TIMEOUT seconds

    Returns True if connected, False if timed out.
    """
    async with _get_connect_lock():
        if await _check_connected(page):
            logger.info("[dlam] R1 already connected — skipping connect flow")
            return True

        logger.info("[dlam] R1 disconnected — starting auto-connect flow")

        # Step 1: click the three bottom icons (screen-space coordinates)
        await asyncio.to_thread(_click_connect_icons_sync)

        # Step 2: wait for screen-share dialog, then dismiss it
        logger.info("[dlam] waiting 2s for screen-share dialog to appear")
        await asyncio.sleep(2.0)
        await asyncio.to_thread(_dismiss_screen_share_dialog_sync)

        # Step 3: poll for connection
        logger.info("[dlam] waiting up to %.0fs for R1 to connect", _CONNECT_TIMEOUT)
        deadline = time.monotonic() + _CONNECT_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(_CONNECT_POLL)
            if await _check_connected(page):
                logger.info("[dlam] R1 connected successfully ✓")
                return True

        logger.warning("[dlam] R1 did not connect within %.0fs", _CONNECT_TIMEOUT)
        return False


async def _find_input(page) -> Any | None:
    """Find the DLAM text input — try exact selector first, then fallbacks."""
    try:
        loc = page.locator(_INPUT_SELECTOR).first
        if await loc.is_visible(timeout=1500):
            return loc
    except Exception:
        pass
    for sel in _INPUT_SELECTORS_FALLBACK:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=1000):
                return loc
        except Exception:
            continue
    return None


def _extract_dlam_message(dom_text: str) -> str:
    """
    Extract DLAM's response message from the page innerText.
    DLAM renders: '<status>\n<message>\n<footer>'
    We strip the status line and footer to return just the message body.
    """
    _FOOTER = "rabbit inc."
    lines = dom_text.splitlines()
    result = []
    for line in lines:
        if _FOOTER in line.lower():
            break
        result.append(line.strip())
    _STATUS_PREFIXES = ("r1 connected", "voice only", "disconnected", "connecting")
    while result and result[0].lower() in _STATUS_PREFIXES:
        result.pop(0)
    return "\n".join(l for l in result if l)


async def _get_response_text(page, snapshot_before: str, timeout: float = 120.0) -> str:
    """
    Poll the page DOM until DLAM's response CHANGES from what was there before
    submitting.  DLAM replaces its single message in-place, so we compare content
    (not length).  Once the message body differs from the pre-submit message AND
    has been stable for SETTLE_SECS, return it.
    """
    deadline = time.monotonic() + timeout

    _BUSY_PHRASES = (
        "we're creating", "we're working", "we're searching", "we're gathering",
        "we're looking", "i'll check", "let me check", "checking in",
        "let me look", "searching for", "looking that up", "one moment",
        "we'll proceed", "pending", "i'll keep you", "keep you updated",
        "will update", "summarizing", "summarising", "just a moment",
        "just a sec", "give me a", "compiling", "preparing", "fetching",
    )
    _SETTLE_QUICK = 2.0
    _SETTLE_BUSY  = 15.0

    msg_before = _extract_dlam_message(snapshot_before)
    last_msg   = msg_before
    last_change: float | None = None

    while time.monotonic() < deadline:
        await asyncio.sleep(_RESPONSE_POLL_MS / 1000)
        try:
            current = await page.evaluate("() => document.body.innerText")
        except Exception:
            continue

        msg = _extract_dlam_message(current)

        if msg != last_msg:
            logger.debug("[dlam] response updated (%d chars)", len(msg))
            last_msg = msg
            last_change = time.monotonic()

        if last_change is None or last_msg == msg_before or not last_msg.strip():
            continue

        still_busy = any(ph in last_msg.lower() for ph in _BUSY_PHRASES)
        settle = _SETTLE_BUSY if still_busy else _SETTLE_QUICK

        if (time.monotonic() - last_change) >= settle:
            logger.info("[dlam] response stable after %.1fs settle — returning",
                        settle)
            return last_msg

    return "[dlam] no response received within timeout"


async def _send_to_dlam(task_text: str, timeout: float = _DEFAULT_TIMEOUT) -> str | None:
    """
    Core function: ensure Brave is running, open DLAM tab, connect R1,
    type task_text, wait for and return the response.
    Returns None only if Brave cannot be started (triggers HTTP fallback).
    """
    # Ensure Brave is running (launch if needed)
    await _ensure_brave_running()

    if not await _cdp_available():
        logger.error("[dlam] Brave could not be started — falling back")
        return None

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return "[dlam] playwright not installed — run: pip install playwright"

    for attempt in range(1, 3):
        try:
            async with async_playwright() as p:
                browser, page = await _get_dlam_page(p)

                # Wait for page to be interactive
                await page.wait_for_load_state("domcontentloaded", timeout=10000)

                # Check R1 connection — full auto-connect if needed
                if not await _check_connected(page):
                    logger.info("[dlam] R1 not connected — running auto-connect")
                    connected = await _attempt_connect(page)
                    if not connected:
                        return (
                            "[dlam] R1 device did not connect after auto-connect attempt. "
                            "Check that the R1 is plugged in via USB-C and try again."
                        )

                # Snapshot current DOM so we can detect new response
                snapshot = await page.evaluate("() => document.body.innerText")
                logger.info("[dlam] snapshot taken (%d chars), submitting task", len(snapshot))

                # Find the input
                inp = await _find_input(page)
                if inp is None:
                    return (
                        "[dlam] could not find the text input on the DLAM page. "
                        "Make sure https://dlam.rabbit.tech/ is open and loaded."
                    )

                # Clear, type, submit
                await inp.click()
                await page.keyboard.press("Control+a")
                await inp.type(task_text, delay=30)
                await page.keyboard.press("Enter")
                logger.info("[dlam] task submitted (%d chars) — waiting for response", len(task_text))

                # Wait for response
                response = await _get_response_text(page, snapshot, timeout=timeout)
                logger.info("[dlam] response received (%d chars)", len(response))
                return response

        except Exception as exc:
            if attempt < 2:
                logger.warning("[dlam] attempt %d failed: %s — retrying in %.0fs",
                               attempt, exc, _RECONNECT_DELAY)
                await asyncio.sleep(_RECONNECT_DELAY)
            else:
                return f"[dlam] error: {exc}"

    return "[dlam] failed after retries"


async def _fallback_fetch(url: str, hint: str = "") -> str:
    """Plain HTTP fallback when Brave/CDP is not available."""
    import re
    import urllib.request as _ur
    try:
        req = _ur.Request(url, headers={"User-Agent": "MYCONEX-DLAM/1.0"})
        with _ur.urlopen(req, timeout=20) as r:
            raw = r.read(300_000).decode("utf-8", errors="ignore")
        m = re.search(r"<title[^>]*>([^<]{1,120})</title>", raw, re.I)
        title = m.group(1).strip() if m else url
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()[:3000]
        note = "[fallback — Brave/DLAM not available, plain HTTP fetch used]\n"
        return f"{note}Title: {title}\n\n{text}"
    except Exception as exc:
        return f"[fallback fetch failed: {exc}]"


# ── Public API ──────────────────────────────────────────────────────────────────

async def dlam_task(description: str, url: str = "", timeout: int = _DEFAULT_TIMEOUT) -> str:
    """
    Send a free-form task to DLAM via the text bar.

    Automatically launches Brave, opens the DLAM tab, and connects the R1
    device if any of those steps are not already done.

    Args:
        description: Plain-English task (e.g. "search for BitNet b1.58 benchmarks
                     and summarise the top 3 results").
        url:         Optional URL to include in the task prompt.
        timeout:     Seconds to wait for DLAM to respond.
    """
    _write_task({
        "type": "dlam_task",
        "action": description,
        "url": url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pending",
        "source": "buzlock",
    })

    prompt = description
    if url:
        prompt = f"Go to {url} and then: {description}"

    result = await _send_to_dlam(prompt, timeout=timeout)

    if result is None:
        logger.warning("[dlam] Brave could not start — falling back to HTTP")
        if url:
            return await _fallback_fetch(url, description)
        return (
            "[dlam] Brave could not be started. "
            "Check that Brave is installed at /opt/brave.com/brave/brave."
        )

    return result


async def dlam_browse(url: str, task: str = "", timeout: int = _DEFAULT_TIMEOUT) -> str:
    """
    Navigate to a URL in DLAM and perform a task on the page.

    Args:
        url:  The URL to visit.
        task: What to do there (default: "summarise the main content").
    """
    task = task or "Summarise the main content of the page concisely."
    return await dlam_task(f"Go to {url} and: {task}", timeout=timeout)


async def dlam_search(query: str, num_results: int = 5, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """
    Web search via DLAM.

    Args:
        query:       Natural-language search query.
        num_results: Target number of results to summarise.
    """
    prompt = (
        f"Search the web for: {query}\n"
        f"Return the top {num_results} results as a numbered list with "
        f"title, URL, and a 1–2 sentence summary for each."
    )
    return await dlam_task(prompt, timeout=timeout)


async def dlam_status() -> dict[str, Any]:
    """Return status of the DLAM/Brave CDP integration."""
    _ensure_dirs()
    cdp_up = await _cdp_available()

    dlam_tab_open = False
    r1_connected = False
    if cdp_up:
        try:
            import urllib.request as _ur
            with _ur.urlopen(f"{_CDP_URL}/json", timeout=3) as r:
                tabs = json.loads(r.read())
                dlam_tab_open = any(
                    "dlam.rabbit.tech" in t.get("url", "") for t in tabs
                )
        except Exception:
            pass

        if dlam_tab_open:
            try:
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.connect_over_cdp(_CDP_URL)
                    for ctx in browser.contexts:
                        for pg in ctx.pages:
                            if "dlam.rabbit.tech" in pg.url:
                                r1_connected = await _check_connected(pg)
                                break
            except Exception:
                pass

    queue: dict[str, int] = {}
    for state in ("pending", "processing", "completed", "failed"):
        queue[state] = len(list((_TASK_DIR / state).glob("*.json")))

    return {
        "cdp_available":   cdp_up,
        "dlam_tab_open":   dlam_tab_open,
        "r1_connected":    r1_connected,
        "cdp_url":         _CDP_URL,
        "dlam_url":        _DLAM_URL,
        "queue":           queue,
        "setup_hint": (
            ""
            if (cdp_up and dlam_tab_open and r1_connected) else
            "Plug in the R1 via USB-C — auto-connect will handle the rest"
            if (cdp_up and dlam_tab_open and not r1_connected) else
            f"Navigate to {_DLAM_URL} in Brave (or call dlam_task — it opens it automatically)"
            if (cdp_up and not dlam_tab_open) else
            "Brave not running — calling dlam_task will launch it automatically"
        ),
    }
