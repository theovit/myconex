"""
MYCONEX Hermes Self-Improvement System
----------------------------------------
Runs after each completed conversation to help Hermes improve over time:

  1. tool_stats    — per-tool call counts, success/failure rates, avg latency
  2. reflection    — asks the agent to analyse recent history and extract insights
  3. skills_file   — accumulated insights injected into future system prompts
  4. soul_patch    — proposes and applies additive updates to ~/.hermes/SOUL.md

All storage lives under ~/.myconex/.  SOUL.md is backed up before every write.
Reflection runs every REFLECT_EVERY_N conversations (default 10), or on demand
via the /reflect slash command.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Storage paths ─────────────────────────────────────────────────────────────
_BASE = Path.home() / ".myconex"
TOOL_STATS_FILE  = _BASE / "tool_stats.json"
SKILLS_FILE      = _BASE / "learned_skills.md"
REFLECTION_LOG   = _BASE / "reflections.jsonl"
SOUL_FILE        = Path.home() / ".hermes" / "SOUL.md"
SOUL_BACKUP_FILE = Path.home() / ".hermes" / "SOUL.md.bak"

REFLECT_EVERY_N = 10   # conversations between automatic reflections

# ─── Prompts ───────────────────────────────────────────────────────────────────

_REFLECT_SYSTEM = (
    "You are in self-reflection mode. Analyse the conversation samples provided "
    "and return ONLY a valid JSON object — no surrounding text, no code fences."
)

_REFLECT_PROMPT = """\
Review the conversation history below and output a JSON object with exactly these keys:

{{
  "key_learnings":            ["important facts or patterns you discovered"],
  "behaviors_to_reinforce":   ["things that worked well and should be repeated"],
  "behaviors_to_avoid":       ["mistakes, inefficiencies, or bad patterns to drop"],
  "tool_usage_notes":         ["observations about tool call effectiveness"],
  "soul_md_additions":        ["new lines or short paragraphs to add to your identity file"],
  "skill_summary":            "2-3 sentence summary of new skills or knowledge gained"
}}

Conversation history:
{history}"""

_SOUL_PATCH_SYSTEM = (
    "Output ONLY the complete updated SOUL.md file content — no explanation, "
    "no code fences. Preserve all existing content; only add, never remove."
)

_SOUL_PATCH_PROMPT = """\
Update the SOUL.md file below by incorporating the proposed additions naturally.
Do not duplicate existing content.  Add the new insights as additional paragraphs
or bullet points where they fit best.

=== Current SOUL.md ===
{soul_content}

=== Proposed additions ===
{additions}"""


# ─── HermesSelfImprover ────────────────────────────────────────────────────────

class HermesSelfImprover:
    """
    Tracks conversation outcomes and tool call stats; periodically triggers
    Hermes to reflect on its own performance and write improvements back to
    its skills file and SOUL.md.

    Thread-safe for reads.  Writes are serialised by the GIL (JSON files are
    small enough that we don't need explicit locking here).
    """

    def __init__(self, reflect_every: int = REFLECT_EVERY_N) -> None:
        self._reflect_every = reflect_every
        self._conversation_count = 0
        self._pending_histories: List[List[Dict[str, Any]]] = []
        self._reflecting = False          # prevents concurrent reflection runs
        _BASE.mkdir(parents=True, exist_ok=True)

    # ── Tool stats ──────────────────────────────────────────────────────────────

    def record_tool_outcome(
        self, tool_name: str, success: bool, latency_ms: float = 0.0
    ) -> None:
        """Increment call/success/failure counters for a tool."""
        stats = self._load_tool_stats()
        entry = stats.setdefault(
            tool_name,
            {"calls": 0, "successes": 0, "failures": 0, "total_ms": 0.0},
        )
        entry["calls"] += 1
        entry["total_ms"] = entry.get("total_ms", 0.0) + latency_ms
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        self._save_tool_stats(stats)

    def get_tool_stats(self) -> Dict[str, Any]:
        return self._load_tool_stats()

    def get_tool_stats_summary(self) -> str:
        """Human-readable performance table for Discord."""
        stats = self._load_tool_stats()
        if not stats:
            return "No tool usage recorded yet."
        lines = ["**Tool Performance**"]
        for name, s in sorted(stats.items()):
            calls = s.get("calls", 0)
            if calls == 0:
                continue
            rate = 100.0 * s.get("successes", 0) / calls
            avg_ms = s.get("total_ms", 0.0) / calls
            bar = "🟢" if rate >= 80 else ("🟡" if rate >= 50 else "🔴")
            lines.append(
                f"  {bar} `{name}`: {calls} calls · {rate:.0f}% success · {avg_ms:.0f} ms avg"
            )
        return "\n".join(lines)

    # ── Conversation recording ──────────────────────────────────────────────────

    def record_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """
        Call after each completed conversation.  Parses OpenAI-format message
        history to extract tool outcomes and queues history for reflection.
        """
        self._extract_tool_outcomes(messages)
        # Keep at most 20 pending histories to cap memory use
        self._pending_histories.append(messages)
        if len(self._pending_histories) > 20:
            self._pending_histories.pop(0)
        self._conversation_count += 1

    def should_reflect(self) -> bool:
        return (
            not self._reflecting
            and self._conversation_count > 0
            and self._conversation_count % self._reflect_every == 0
        )

    # ── Reflection ─────────────────────────────────────────────────────────────

    def run_reflection(self, agent: Any) -> Optional[Dict[str, Any]]:
        """
        Synchronous — run inside asyncio.to_thread().

        Asks the agent to analyse recent conversations, writes a reflection
        entry to REFLECTION_LOG, updates SKILLS_FILE, and returns the parsed
        reflection dict (or None on failure).
        """
        if self._reflecting or not self._pending_histories:
            return None

        self._reflecting = True
        try:
            history_text = self._format_histories(self._pending_histories[-5:])
            prompt = _REFLECT_PROMPT.format(history=history_text)
            result = agent.run_conversation(
                user_message=prompt,
                system_message=_REFLECT_SYSTEM,
                conversation_history=[],
            )
            raw = result.get("final_response") or ""
            reflection = self._parse_json(raw)
            if reflection:
                self._append_reflection_log(reflection)
                self._update_skills_file(reflection)
                logger.info(
                    "[self_improve] reflection complete — %d learnings, %d avoidances",
                    len(reflection.get("key_learnings", [])),
                    len(reflection.get("behaviors_to_avoid", [])),
                )
            return reflection
        except Exception as exc:
            logger.warning("[self_improve] reflection failed: %s", exc)
            return None
        finally:
            self._reflecting = False

    def patch_soul(self, agent: Any) -> bool:
        """
        Synchronous — run inside asyncio.to_thread().

        Collects soul_md_additions from recent reflection log entries, has the
        agent rewrite SOUL.md incorporating them, and writes the result.
        Returns True if SOUL.md was updated.
        """
        additions = self._collect_soul_additions()
        if not additions:
            return False

        soul_content = SOUL_FILE.read_text(encoding="utf-8") if SOUL_FILE.exists() else ""
        prompt = _SOUL_PATCH_PROMPT.format(
            soul_content=soul_content or "(empty — write a foundational identity)",
            additions="\n".join(f"- {a}" for a in additions),
        )
        try:
            result = agent.run_conversation(
                user_message=prompt,
                system_message=_SOUL_PATCH_SYSTEM,
                conversation_history=[],
            )
            new_content = (result.get("final_response") or "").strip()
            if new_content and len(new_content) > 100:
                if soul_content:
                    SOUL_BACKUP_FILE.write_text(soul_content, encoding="utf-8")
                SOUL_FILE.parent.mkdir(parents=True, exist_ok=True)
                SOUL_FILE.write_text(new_content, encoding="utf-8")
                logger.info("[self_improve] SOUL.md updated (%d chars)", len(new_content))
                return True
        except Exception as exc:
            logger.warning("[self_improve] soul patch failed: %s", exc)
        return False

    # ── Skills injection ────────────────────────────────────────────────────────

    def get_skills_injection(self) -> str:
        """
        Return a block to append to the system prompt with accumulated skills,
        or "" if the file doesn't exist or is empty.
        """
        if not SKILLS_FILE.exists():
            return ""
        try:
            content = SKILLS_FILE.read_text(encoding="utf-8").strip()
            if not content:
                return ""
            # Inject only the last 3 000 chars to stay within context budgets
            if len(content) > 3000:
                content = "…(truncated)\n" + content[-3000:]
            return f"\n\n## Learned Context & Skills\n{content}\n"
        except Exception:
            return ""

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _extract_tool_outcomes(self, messages: List[Dict[str, Any]]) -> None:
        """Parse tool_call / tool result message pairs and record outcomes."""
        # Map tool_call_id → tool_name from assistant messages
        call_names: Dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = tc.get("id", "")
                    name = (tc.get("function") or {}).get("name", "")
                    if cid and name:
                        call_names[cid] = name

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            name = call_names.get(msg.get("tool_call_id", ""), "")
            if not name:
                continue
            content = str(msg.get("content") or "").lower()
            success = not (
                content.startswith("error")
                or "error:" in content[:40]
                or content.startswith("execution error")
                or content.startswith("research error")
            )
            self.record_tool_outcome(name, success)

    def _format_histories(self, histories: List[List[Dict[str, Any]]]) -> str:
        lines: List[str] = []
        for i, history in enumerate(histories, 1):
            lines.append(f"\n--- Conversation {i} ---")
            for msg in history[-20:]:
                role = msg.get("role", "?")
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                content = str(content)
                if role == "tool":
                    lines.append(f"[TOOL RESULT]: {content[:200]}")
                elif role in ("user", "assistant"):
                    lines.append(f"[{role.upper()}]: {content[:300]}")
        return "\n".join(lines)

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            parts = raw.split("```", 2)
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Fall back to extracting the outermost { } block
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                pass
        logger.debug("[self_improve] could not parse reflection JSON: %.200s", raw)
        return None

    def _append_reflection_log(self, reflection: Dict[str, Any]) -> None:
        try:
            entry = {"ts": time.time(), "ts_human": time.strftime("%Y-%m-%d %H:%M"), **reflection}
            with REFLECTION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("[self_improve] reflection log write failed: %s", exc)

    def _update_skills_file(self, reflection: Dict[str, Any]) -> None:
        try:
            lines = [f"\n### {time.strftime('%Y-%m-%d %H:%M')}"]
            if summary := reflection.get("skill_summary", ""):
                lines.append(summary)
            for item in reflection.get("key_learnings", []):
                lines.append(f"- {item}")
            for item in reflection.get("behaviors_to_reinforce", []):
                lines.append(f"✓ {item}")
            for item in reflection.get("behaviors_to_avoid", []):
                lines.append(f"✗ {item}")
            for note in reflection.get("tool_usage_notes", []):
                lines.append(f"🔧 {note}")
            with SKILLS_FILE.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as exc:
            logger.warning("[self_improve] skills file write failed: %s", exc)

    def _collect_soul_additions(self) -> List[str]:
        """Gather all soul_md_additions from the last 30 reflection log lines."""
        additions: List[str] = []
        try:
            if not REFLECTION_LOG.exists():
                return []
            for line in REFLECTION_LOG.read_text().splitlines()[-30:]:
                try:
                    additions.extend(json.loads(line).get("soul_md_additions", []))
                except Exception:
                    pass
        except Exception:
            pass
        # Deduplicate while preserving order
        seen: set = set()
        return [a for a in additions if not (a in seen or seen.add(a))]  # type: ignore[func-returns-value]

    def _load_tool_stats(self) -> Dict[str, Any]:
        try:
            if TOOL_STATS_FILE.exists():
                return json.loads(TOOL_STATS_FILE.read_text())
        except Exception:
            pass
        return {}

    def _save_tool_stats(self, stats: Dict[str, Any]) -> None:
        try:
            TOOL_STATS_FILE.write_text(json.dumps(stats, indent=2))
        except Exception as exc:
            logger.debug("[self_improve] tool stats write failed: %s", exc)
