# MYCONEX — Claude Code Behavioral Rules

These rules govern how Claude Code works on the MYCONEX codebase.
They are enforced on every session and updated automatically via the self-improvement loop.

---

## 1. Plan Mode Default

**Enter plan mode before any non-trivial task.**

A task is non-trivial if it:
- Touches more than 2 files
- Adds a new class, agent type, or integration
- Changes routing logic, NATS subjects, or discovery behavior
- Has unclear scope or ambiguous requirements

In plan mode: read all relevant files first, identify risks, propose the approach,
and wait for confirmation before writing code.

Skip plan mode only for: single-line fixes, docstring edits, or trivial renames.

---

## 2. Subagent Strategy — Keep Main Context Clean

Use subagents (via the Agent tool) to protect the main context window from bloat.

**Delegate to subagents when:**
- Exploring an unfamiliar part of the codebase (use `Explore` agent)
- Researching external APIs, packages, or concepts (use `general-purpose` agent)
- The task requires reading more than ~5 files before acting

**Do NOT use subagents for:**
- Simple targeted reads of known file paths (use Read directly)
- Single-file edits (use Edit directly)

After a subagent returns, synthesize only the relevant findings into the main context.
Discard raw file dumps.

---

## 3. Self-Improvement Loop

**After any correction, update `lessons.md` with the pattern.**

A correction is: the user says "no", "wrong", "don't do that", "that's not right",
or reverts/rewrites something I did.

Update procedure:
1. Identify the *class* of mistake (e.g., "added unused import", "broke existing API")
2. Write a concise rule that prevents it
3. Append to `lessons.md` under the appropriate category
4. Apply the rule immediately in the current session

This loop ensures mistakes are never repeated across sessions.

---

## 4. Verification Before Done

**Never consider a task complete without verification.**

Verification checklist (apply what's relevant):
- [ ] Syntax-check all modified Python files: `python3 -c "import ast; ast.parse(open('f').read())"`
- [ ] Imports resolve: no circular imports between new modules
- [ ] Existing tests still pass (run `pytest` if tests exist)
- [ ] New code is wired up: factory functions are called, new agents are registered
- [ ] No orphaned files: new modules are imported somewhere or documented

For large changes, explicitly state which checks were performed.

---

## 5. MYCONEX-Specific Rules

### Architecture
- All new agents must subclass `BaseAgent` and implement `can_handle()` + `handle_task()`
- New agents registered via `TaskRouter.register_agent()` automatically get `set_router()` called
- `RLMAgent` is the preferred primary agent — use `create_rlm_agent()` factory
- MoE complexity threshold for delegation is `0.60` (see `rlm_agent.py:DECOMPOSE_THRESHOLD`)

### Code Style
- Async all the way: no `asyncio.run()` inside agents; use `await`
- Type hints on all public methods
- Module-level logger: `logger = logging.getLogger(__name__)`
- Constants in UPPER_CASE at module level, not hardcoded inline

### What NOT to do
- Do not hardcode API keys — use `os.getenv()`
- Do not import `moe_hermes_integration` from `base_agent.py` (circular dependency)
- Do not call `asyncio.get_event_loop().run_until_complete()` inside async context
- Do not add `__init__.py` files unless a package boundary is actually needed

### File layout
```
myconex/
  orchestration/agents/    — base_agent, rlm_agent, context_manager
  orchestration/workflows/ — task_router
  core/gateway/            — agentic_tools, python_repl, discord_gateway
  integrations/            — moe_hermes_integration, flash-moe/, hermes-agent/
  core/coordinator/        — orchestrator (mesh task lifecycle)
  core/discovery/          — mesh_discovery (mDNS)
  core/messaging/          — nats_client
```

---

## 6. Commit Discipline

- Never commit unless the user explicitly asks
- Stage specific files, not `git add -A`
- Commit messages: imperative mood, explain *why* not *what*
- Never `--no-verify` or force-push to main
