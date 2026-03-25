# MYCONEX — Lessons Learned

Persistent learning log for Claude Code working on MYCONEX.
Each entry is a pattern extracted from a correction.
New entries are appended by the self-improvement loop defined in CLAUDE.md §3.

Format: `## [Category] — [Rule]` followed by context and the triggering incident.

---

## [Imports] — Never import moe_hermes_integration from base_agent.py

`moe_hermes_integration` imports from `base_agent`, so the reverse creates a
circular import. Use `TYPE_CHECKING` guard or pass dependencies via injection.

**How to apply:** If base_agent needs complexity scoring, use the local
`_estimate_complexity()` function defined in the same file.

---

## [Async] — Don't call asyncio.get_event_loop().run_until_complete() inside an async function

This raises "This event loop is already running" in async contexts (FastAPI, Discord.py, etc.).
Use `await` directly, or `asyncio.ensure_future()` for fire-and-forget.

**How to apply:** Check whether the calling context is already async before
using `run_until_complete`. In `agentic_tools.py` handlers that are called
synchronously from hermes registry, the pattern is acceptable but fragile —
prefer making handlers async and awaiting them.

---

## [Architecture] — New agents need set_router() called before delegate() works

`TaskRouter.start()` and `TaskRouter.register_agent()` call `set_router(self)`
automatically. But agents created outside a TaskRouter (e.g. in tests, scripts)
will have `_router = None` and delegate() will return an error result.

**How to apply:** Always create agents via `TaskRouter` or call `agent.set_router(router)`
manually before invoking delegation.

---

## [Files] — Verify syntax on all touched Python files before declaring done

AST parse check: `python3 -c "import ast; ast.parse(open('file.py').read())"`.
Run this on every file modified or created in the session.

---

## [REPL] — asyncio.get_event_loop() is deprecated in Python 3.10+

Use `asyncio.get_running_loop()` inside async functions.
Use `asyncio.new_event_loop()` / `asyncio.run()` in sync entry points.

---

*This file is updated automatically. Do not delete entries — mark them
`[RESOLVED]` if superseded by a better rule.*
