#!/usr/bin/env python3
"""
seed_context — Pull Discord channel history and seed a MYCONEX session.

Fetches the last N messages from a Discord channel, converts them to an
AgentContext (user/assistant turns), stores it in the API gateway session
store, and returns a context_id you can pass to ask.py or chat.py.

Usage
-----
    python cli/seed_context.py --channel-id 1234567890
    python cli/seed_context.py --channel-id 1234567890 --limit 100
    python cli/seed_context.py --channel-id 1234567890 --context-id myctx
    DISCORD_BOT_TOKEN=xxx python cli/seed_context.py --channel-id 1234567890

After seeding, use the printed context_id::

    # One-liner pipeline:
    CID=$(python cli/seed_context.py --channel-id 1234567890 --quiet)
    python cli/ask.py "Summarize the above conversation" --context-id "$CID"
    python cli/chat.py --context-id "$CID"

Options
-------
  --channel-id   Discord channel ID to fetch history from (required).
  --limit        Max messages to fetch (default: 50, max: 100).
  --context-id   Use a specific context_id instead of auto-generating one.
  --url          API gateway URL (default: http://localhost:8765).
  --quiet, -q    Only print the context_id (machine-readable, for piping).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_DEFAULT = "http://localhost:8765"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a MYCONEX session from Discord channel history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--channel-id",
        required=True,
        metavar="ID",
        help="Discord channel ID to fetch history from",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Number of messages to fetch (default: 50)",
    )
    parser.add_argument(
        "--context-id",
        default=None,
        metavar="ID",
        help="Use a specific context_id (default: auto-generated)",
    )
    parser.add_argument(
        "--url",
        default=API_DEFAULT,
        metavar="URL",
        help=f"API gateway URL (default: {API_DEFAULT})",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print the context_id (for use in shell pipelines)",
    )
    args = parser.parse_args()

    # Pre-check: DISCORD_BOT_TOKEN must be set before we call the API,
    # since the gateway needs it to authenticate with Discord.
    if not os.getenv("DISCORD_BOT_TOKEN"):
        print(
            "error: DISCORD_BOT_TOKEN is not set.\n"
            "  Set it with: export DISCORD_BOT_TOKEN=your_token\n"
            "  Or add it to .env and start MYCONEX with: python main.py --mode api",
            file=sys.stderr,
        )
        sys.exit(1)

    payload: dict = {
        "channel_id": args.channel_id,
        "limit": min(args.limit, 100),
    }
    if args.context_id:
        payload["context_id"] = args.context_id

    base_url = args.url.rstrip("/")
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/session/seed",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body
        print(f"error {e.code}: {msg}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"error: cannot reach API at {base_url} — {e}", file=sys.stderr)
        print("Start MYCONEX with: python main.py --mode api", file=sys.stderr)
        sys.exit(1)

    context_id = result.get("context_id", "")
    turns = result.get("turns_loaded", 0)
    channel = result.get("channel_id", args.channel_id)

    if args.quiet:
        # Machine-readable: just the context_id on stdout
        print(context_id)
    else:
        print(f"seeded {turns} turns from channel {channel}")
        print(f"context_id: {context_id}")
        print()
        print("Use it:")
        print(f'  python cli/ask.py "Your prompt" --context-id {context_id}')
        print(f"  python cli/chat.py --context-id {context_id}")


if __name__ == "__main__":
    main()
