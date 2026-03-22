#!/usr/bin/env python3
"""
status — Show MYCONEX node, agent, and session status.

Usage
-----
    python cli/status.py
    python cli/status.py --url http://localhost:8765
    python cli/status.py --json
    python cli/status.py --session abc12345    # inspect a specific session
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

API_DEFAULT = "http://localhost:8765"


def get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show MYCONEX node status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Output raw JSON",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="CONTEXT_ID",
        help="Inspect a specific session (shows turn preview)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # ── Session inspect mode ──────────────────────────────────────────────────
    if args.session:
        try:
            data = get_json(f"{base_url}/session/{args.session}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"session '{args.session}' not found", file=sys.stderr)
            else:
                print(f"error {e.code}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"error: cannot reach API at {base_url} — {e}", file=sys.stderr)
            sys.exit(1)

        if args.output_json:
            print(json.dumps(data, indent=2))
            return

        print(f"Session: {data['context_id']}")
        print(f"  session_id:  {data.get('session_id')}")
        print(f"  turn_count:  {data.get('turn_count', 0)}")
        meta = data.get("metadata", {})
        if meta:
            for k, v in meta.items():
                print(f"  {k}: {v}")
        print()
        preview = data.get("preview", [])
        if preview:
            print("Last turns:")
            for t in preview:
                role = t.get("role", "?")
                content = t.get("content", "")
                print(f"  [{role}] {content[:120]}")
        return

    # ── Node status mode ──────────────────────────────────────────────────────
    try:
        data = get_json(f"{base_url}/status")
    except urllib.error.URLError as e:
        print(f"error: cannot reach API at {base_url} — {e}", file=sys.stderr)
        print("Start MYCONEX with: python main.py --mode api", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(json.dumps(data, indent=2))
        return

    tier = data.get("tier", "?")
    sessions = data.get("active_sessions", 0)
    agents = data.get("agents", [])
    routes = data.get("routes", [])

    print(f"MYCONEX Node")
    print(f"  Tier:     {tier}")
    print(f"  Sessions: {sessions} active")
    print(f"  Agents:   {len(agents)}")
    for a in agents:
        name = a.get("name", "?")
        atype = a.get("type", "?")
        state = a.get("state", "?")
        model = a.get("model", "?")
        total = a.get("total_tasks", 0)
        errors = a.get("total_errors", 0)
        uptime = a.get("uptime_s", 0)
        print(
            f"    • {name} ({atype}) [{state}]  "
            f"model={model}  tasks={total}  errors={errors}  uptime={uptime}s"
        )
    if routes:
        print(f"  Routes:  {', '.join(routes)}")


if __name__ == "__main__":
    main()
