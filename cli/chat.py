#!/usr/bin/env python3
"""
chat — Interactive multi-turn chat session with MYCONEX API gateway.

Maintains a persistent context_id across turns so the model sees the
full conversation history for each new message.

Usage
-----
    python cli/chat.py                          # start fresh session
    python cli/chat.py --context-id abc12345    # resume a saved session
    python cli/chat.py --url http://localhost:8765

Special commands (type during chat)
------------------------------------
    /reset          Clear history and start a new session
    /id             Print the current context_id
    /status         Show node/agent status
    /seed <chan_id> Seed context from a Discord channel (requires bot token)
    /exit, exit     Quit

Copilot-X / Claude CLI integration
------------------------------------
    # Pipe a prompt non-interactively (single turn):
    echo "What is mycelium?" | python cli/chat.py --context-id $CID --pipe

    # Or pass via stdin:
    python cli/chat.py --context-id $CID <<< "Explain the above"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

try:
    import readline  # noqa: F401 — enables arrow-key history on Linux/macOS
except ImportError:
    pass  # Windows — silently skip

API_DEFAULT = "http://localhost:8765"


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _send_chat(base_url: str, prompt: str, context_id: str | None) -> dict:
    payload: dict = {"prompt": prompt}
    if context_id:
        payload["context_id"] = context_id
    return post_json(f"{base_url}/chat", payload)


def _print_status(base_url: str) -> None:
    try:
        data = get_json(f"{base_url}/status")
        tier = data.get("tier", "?")
        sessions = data.get("active_sessions", 0)
        agents = data.get("agents", [])
        print(f"  tier={tier}  sessions={sessions}  agents={len(agents)}")
        for a in agents:
            print(f"    • {a.get('name')} [{a.get('state')}] model={a.get('model')}")
    except Exception as e:
        print(f"  [error fetching status: {e}]", file=sys.stderr)


def _seed_context(base_url: str, channel_id: str, context_id: str | None) -> str | None:
    """Seed a session from Discord channel history. Returns the new context_id."""
    payload: dict = {"channel_id": channel_id, "limit": 50}
    if context_id:
        payload["context_id"] = context_id
    try:
        result = post_json(f"{base_url}/session/seed", payload, timeout=30)
        new_cid = result.get("context_id")
        turns = result.get("turns_loaded", 0)
        print(f"  seeded {turns} turns from channel {channel_id} → context_id={new_cid}")
        return new_cid
    except Exception as e:
        print(f"  [seed error: {e}]", file=sys.stderr)
        return context_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with MYCONEX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--context-id", "-c",
        default=None,
        metavar="ID",
        help="Resume a saved session",
    )
    parser.add_argument(
        "--url",
        default=API_DEFAULT,
        metavar="URL",
        help=f"API gateway URL (default: {API_DEFAULT})",
    )
    parser.add_argument(
        "--pipe",
        action="store_true",
        help="Non-interactive: read one prompt from stdin and exit",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    context_id: str | None = args.context_id

    # ── Non-interactive (pipe) mode ───────────────────────────────────────────
    if args.pipe or not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if not prompt:
            sys.exit(0)
        try:
            result = _send_chat(base_url, prompt, context_id)
        except urllib.error.URLError as e:
            print(f"error: cannot reach API at {base_url} — {e}", file=sys.stderr)
            sys.exit(1)
        if result.get("success"):
            print(result.get("response", ""))
            if not context_id:
                print(f"\n[context_id: {result['context_id']}]", file=sys.stderr)
        else:
            print(f"error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print(f"MYCONEX Chat  (api={base_url})")
    if context_id:
        print(f"Resuming session: {context_id}")
    else:
        print("Starting new session. Type /id after the first message to see context_id.")
    print("Commands: /reset  /id  /status  /seed <channel_id>  /exit\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue

        # ── Built-in commands ─────────────────────────────────────────────────
        if prompt.lower() in ("/exit", "exit", "quit", "/quit"):
            print("Goodbye.")
            break

        if prompt == "/reset":
            context_id = None
            print("  Session cleared — starting fresh.\n")
            continue

        if prompt == "/id":
            print(f"  context_id: {context_id or '(not set — send a message first)'}\n")
            continue

        if prompt == "/status":
            _print_status(base_url)
            print()
            continue

        if prompt.startswith("/seed "):
            chan = prompt.split(None, 1)[1].strip()
            context_id = _seed_context(base_url, chan, context_id)
            print()
            continue

        # ── Chat turn ─────────────────────────────────────────────────────────
        try:
            result = _send_chat(base_url, prompt, context_id)
        except urllib.error.URLError as e:
            print(f"[error] Cannot reach API: {e}\n", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[error] {e}\n", file=sys.stderr)
            continue

        if result.get("success"):
            context_id = result["context_id"]
            response = result.get("response", "")
            model = result.get("model", "")
            ms = result.get("duration_ms", 0)
            print(f"MYCONEX [{model} {ms:.0f}ms]: {response}\n")
        else:
            print(f"[error] {result.get('error')}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
