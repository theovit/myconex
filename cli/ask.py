#!/usr/bin/env python3
"""
ask — Submit a one-shot prompt to the MYCONEX API gateway.

Usage
-----
    python cli/ask.py "What is a fungal network?"
    python cli/ask.py "Follow-up question" --context-id abc12345
    python cli/ask.py "Write code for X" --task-type code
    python cli/ask.py "Summarize this" --url http://localhost:8765

Options
-------
  --context-id, -c   Reuse an existing session (context_id from a previous call).
                     A new context_id is printed to stderr on the first call so
                     you can capture it for follow-ups.
  --task-type, -t    Task type to route (default: chat). Other values: code,
                     summarize, translate, embedding — see TaskRouter routes.
  --url              API gateway base URL (default: http://localhost:8765).
  --json, -j         Print the full JSON response instead of just the text.
  --no-context-hint  Suppress the "context_id: ..." hint on stderr.

Copilot-X / Gemini integration
-------------------------------
    # Seed context from Discord first, then ask in one pipeline:
    CID=$(python cli/seed_context.py --channel-id 1234567890 2>/dev/null | grep context_id | awk '{print $2}')
    python cli/ask.py "Summarize the above conversation" --context-id "$CID"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a prompt to MYCONEX API gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", help="The prompt to send")
    parser.add_argument(
        "--context-id", "-c",
        default=None,
        metavar="ID",
        help="Reuse an existing session context_id",
    )
    parser.add_argument(
        "--task-type", "-t",
        default="chat",
        metavar="TYPE",
        help="Task type (default: chat)",
    )
    parser.add_argument(
        "--url",
        default=API_DEFAULT,
        metavar="URL",
        help=f"API gateway URL (default: {API_DEFAULT})",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="output_json",
        help="Output full JSON response",
    )
    parser.add_argument(
        "--no-context-hint",
        action="store_true",
        help="Suppress the context_id hint on stderr",
    )
    args = parser.parse_args()

    payload: dict = {
        "task_type": args.task_type,
        "prompt": args.prompt,
    }
    if args.context_id:
        payload["context_id"] = args.context_id

    try:
        result = post_json(f"{args.url}/task", payload)
    except urllib.error.URLError as e:
        print(f"error: cannot reach API at {args.url} — {e}", file=sys.stderr)
        print("Start MYCONEX with: python main.py --mode api", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        print(result.get("response", ""))
        if not args.no_context_hint:
            cid = result.get("context_id", "")
            # Only print the hint when a new context was created (not reused)
            if not args.context_id and cid:
                print(f"\n[context_id: {cid}]", file=sys.stderr)
    else:
        print(f"error: {result.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
