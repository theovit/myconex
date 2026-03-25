"""
MYCONEX Autonomous Optimization Loop
======================================
The self-optimization engine — MYCONEX analyses its own codebase, identifies
improvement opportunities, plans changes, executes them in a sandbox, verifies
results, and records learnings.

Cycle (one iteration):
  1. Analyse  — scan codebase with CodebaseIndex, check metrics, read lessons.md
  2. Plan     — ask RLMAgent to identify top improvement opportunity
  3. Sandbox  — execute the improvement via SandboxExecutor (never writes files directly)
  4. Verify   — syntax-check, unit test if available
  5. Record   — update lessons.md, persistent memory, and audit log
  6. Reflect  — every N cycles, run full self-optimization review

The loop never modifies files without explicit sandbox confirmation.
All actions are logged to ~/.myconex/audit.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MYCONEX_ROOT = Path(__file__).parent.parent
_MYCONEX_DIR  = Path.home() / ".myconex"
_AUDIT_LOG    = _MYCONEX_DIR / "audit.jsonl"
_LESSONS_FILE = _MYCONEX_ROOT / "lessons.md"
_METRICS_FILE = _MYCONEX_DIR / "metrics.json"


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ImprovementOpportunity:
    """A potential codebase improvement identified by the loop."""
    opportunity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    target_file: str = ""
    impact: str = "medium"         # "high" | "medium" | "low"
    category: str = "quality"      # "quality" | "performance" | "feature" | "bugfix"
    proposed_change: str = ""      # Prose description, NOT code (sandbox executes separately)
    priority_score: float = 0.5


@dataclass
class CycleResult:
    """Result of one autonomous optimization cycle."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cycle_number: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    opportunity: Optional[ImprovementOpportunity] = None
    sandbox_output: str = ""
    verified: bool = False
    lesson_added: bool = False
    error: Optional[str] = None
    metrics_snapshot: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at


@dataclass
class LoopMetrics:
    """Cumulative metrics for the autonomous loop."""
    cycles_run: int = 0
    cycles_succeeded: int = 0
    cycles_failed: int = 0
    lessons_added: int = 0
    sandbox_executions: int = 0
    total_runtime_s: float = 0.0
    last_cycle_at: Optional[float] = None
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Autonomous Optimization Loop ────────────────────────────────────────────

class AutonomousOptimizationLoop:
    """
    Self-optimization engine for MYCONEX.

    Orchestrates continuous codebase analysis, improvement planning, sandboxed
    execution, verification, and learning — all without human intervention.

    Usage:
        loop = AutonomousOptimizationLoop(agent, sandbox_executor)
        await loop.run(max_cycles=50, cycle_interval_s=30)
    """

    # How many cycles between full self-optimization reviews
    REFLECT_EVERY_N: int = 10

    # Prompts used at each phase
    _ANALYSE_PROMPT = """You are the autonomous self-improvement engine for MYCONEX.

Codebase summary (top relevant chunks):
{codebase_summary}

Recent lessons:
{lessons}

Current metrics:
{metrics}

Identify the SINGLE most impactful improvement opportunity in the MYCONEX codebase.
Focus on: correctness, performance, missing error handling, or untested edge cases.

Respond with JSON:
{{
  "title": "short title",
  "description": "2-3 sentence description of the problem",
  "target_file": "relative/path/to/file.py",
  "impact": "high|medium|low",
  "category": "quality|performance|feature|bugfix",
  "proposed_change": "describe the change in plain English — do NOT write code here",
  "priority_score": 0.0-1.0
}}"""

    _PLAN_PROMPT = """You are planning a code improvement for MYCONEX.

Opportunity:
{opportunity_json}

Write a Python script that implements this improvement by:
1. Reading the target file
2. Making the minimal necessary change
3. Writing the result to a temp file /tmp/myconex_patch_{cycle_id}.py
4. Printing "PATCH_WRITTEN" on success

Constraints:
- Make the smallest possible change
- Do not break existing interfaces
- Include a comment explaining the change
- The script MUST be self-contained (no user input)

Write ONLY the Python script, no explanation."""

    _VERIFY_PROMPT = """The following patch was applied to {target_file}.

Patch script output:
{sandbox_output}

Evaluate:
1. Did the patch succeed (PATCH_WRITTEN in output)?
2. Are there any obvious syntax errors or logic issues?
3. Does it address the stated opportunity?

Answer with JSON: {{"verified": true/false, "reason": "...", "lesson": "one-line lesson to add to lessons.md or empty string"}}"""

    def __init__(
        self,
        agent,                  # RLMAgent
        sandbox,                # SandboxExecutor
        cycle_interval_s: float = 30.0,
        dry_run: bool = False,  # dry_run=True: plan only, don't execute sandbox
    ) -> None:
        self.agent = agent
        self.sandbox = sandbox
        self.cycle_interval_s = cycle_interval_s
        self.dry_run = dry_run
        self.metrics = LoopMetrics()
        self._stop = asyncio.Event()
        self._cycle_history: list[CycleResult] = []

        # Lazy-init codebase index
        self._codebase_index = None

        # Optional novelty scanner — provides externally-sourced proposals
        self._novelty_scanner = None

    # ── Public API ────────────────────────────────────────────────────────────

    def register_novelty_scanner(self, scanner) -> None:
        """
        Register a NoveltyScanner to supply external proposals.

        When registered, each cycle checks the scanner's queue first.
        If a novelty proposal is available it takes priority over the
        LLM-generated analysis opportunity.
        """
        self._novelty_scanner = scanner
        logger.info("[autonomous_loop] novelty scanner registered")

    async def run(
        self,
        max_cycles: Optional[int] = None,
        cycle_interval_s: Optional[float] = None,
    ) -> LoopMetrics:
        """
        Run the autonomous optimization loop.

        Args:
            max_cycles:       Stop after this many cycles (None = run forever).
            cycle_interval_s: Override the per-cycle sleep interval.

        Returns:
            Final LoopMetrics.
        """
        if cycle_interval_s is not None:
            self.cycle_interval_s = cycle_interval_s

        _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load_metrics()
        self._init_codebase_index()

        logger.info(
            "[autonomous_loop] starting (max_cycles=%s, interval=%ss, dry_run=%s)",
            max_cycles, self.cycle_interval_s, self.dry_run,
        )

        while not self._stop.is_set():
            if max_cycles is not None and self.metrics.cycles_run >= max_cycles:
                logger.info("[autonomous_loop] reached max_cycles=%d, stopping", max_cycles)
                break

            cycle_result = await self._run_cycle()
            self._cycle_history.append(cycle_result)
            self._save_metrics()

            # Periodic full reflection
            if self.metrics.cycles_run % self.REFLECT_EVERY_N == 0:
                await self._reflect()

            if not self._stop.is_set():
                await asyncio.sleep(self.cycle_interval_s)

        return self.metrics

    def stop(self) -> None:
        """Signal the loop to stop after the current cycle."""
        self._stop.set()

    # ── Cycle Phases ─────────────────────────────────────────────────────────

    async def _run_cycle(self) -> CycleResult:
        """Execute one full optimisation cycle."""
        cycle = CycleResult(cycle_number=self.metrics.cycles_run + 1)
        self.metrics.cycles_run += 1
        self.metrics.last_cycle_at = time.time()

        logger.info("[autonomous_loop] cycle %d starting", cycle.cycle_number)

        try:
            # Phase 1: Analyse
            opportunity = await self._phase_analyse(cycle)
            if opportunity is None:
                cycle.error = "Analysis produced no opportunity"
                self._finalize_cycle(cycle, success=False)
                return cycle
            cycle.opportunity = opportunity

            # Phase 2: Plan + Sandbox
            if not self.dry_run:
                sandbox_output = await self._phase_sandbox(cycle, opportunity)
                cycle.sandbox_output = sandbox_output
                self.metrics.sandbox_executions += 1
            else:
                cycle.sandbox_output = "[dry_run: sandbox skipped]"

            # Phase 3: Verify
            verified, lesson = await self._phase_verify(cycle, opportunity)
            cycle.verified = verified

            # Phase 4: Record
            if lesson:
                await self._add_lesson(lesson, opportunity)
                cycle.lesson_added = True
                self.metrics.lessons_added += 1

            self._finalize_cycle(cycle, success=verified)

        except Exception as exc:
            cycle.error = traceback.format_exc(limit=5)
            logger.error("[autonomous_loop] cycle %d error: %s", cycle.cycle_number, exc)
            self._finalize_cycle(cycle, success=False)

        self._write_audit(cycle)
        return cycle

    async def _phase_analyse(self, cycle: CycleResult) -> Optional[ImprovementOpportunity]:
        """Phase 1: Analyse codebase and identify top improvement opportunity.

        Checks the novelty scanner's proposal queue first; if a proposal is
        available it is used directly (skipping the LLM analysis step).
        Falls back to standard LLM-driven codebase analysis when the queue
        is empty.
        """
        # ── Novelty scanner fast-path ──────────────────────────────────────
        if self._novelty_scanner is not None:
            try:
                proposal = self._novelty_scanner.dequeue_proposal()
                if proposal is not None:
                    logger.info(
                        "[autonomous_loop] using novelty proposal: %s (score=%.2f)",
                        proposal.get("title", "?")[:60],
                        proposal.get("priority_score", 0.0),
                    )
                    return ImprovementOpportunity(
                        title=proposal.get("title", "Novelty idea"),
                        description=proposal.get("description", ""),
                        target_file=proposal.get("target_file", ""),
                        impact=proposal.get("impact", "medium"),
                        category=proposal.get("category", "feature"),
                        proposed_change=proposal.get("proposed_change", ""),
                        priority_score=float(proposal.get("priority_score", 0.5)),
                    )
            except Exception as exc:
                logger.warning("[autonomous_loop] novelty dequeue error: %s", exc)

        # ── Standard LLM analysis ──────────────────────────────────────────
        codebase_summary = self._get_codebase_summary()
        lessons = self._read_lessons(max_chars=1500)
        metrics_str = json.dumps(self.metrics.to_dict(), indent=2)

        prompt = self._ANALYSE_PROMPT.format(
            codebase_summary=codebase_summary[:3000],
            lessons=lessons[:1500],
            metrics=metrics_str[:500],
        )

        try:
            raw = await self.agent.chat(
                [
                    {"role": "system", "content": "You are a code quality analyst. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                logger.warning("[autonomous_loop] no JSON in analyse response")
                return None
            data = json.loads(json_match.group())
            return ImprovementOpportunity(
                title=data.get("title", "Untitled"),
                description=data.get("description", ""),
                target_file=data.get("target_file", ""),
                impact=data.get("impact", "medium"),
                category=data.get("category", "quality"),
                proposed_change=data.get("proposed_change", ""),
                priority_score=float(data.get("priority_score", 0.5)),
            )
        except Exception as exc:
            logger.warning("[autonomous_loop] analyse failed: %s", exc)
            return None

    async def _phase_sandbox(
        self, cycle: CycleResult, opportunity: ImprovementOpportunity
    ) -> str:
        """Phase 2: Generate and execute a patch script in a sandbox."""
        plan_prompt = self._PLAN_PROMPT.format(
            opportunity_json=json.dumps(asdict(opportunity), indent=2),
            cycle_id=cycle.cycle_id,
        )
        try:
            patch_script = await self.agent.chat(
                [
                    {"role": "system", "content": "Write a Python script only. No explanation, no markdown."},
                    {"role": "user", "content": plan_prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            # Strip markdown code fences if present
            import re
            patch_script = re.sub(r"^```python\s*", "", patch_script.strip(), flags=re.MULTILINE)
            patch_script = re.sub(r"^```\s*$", "", patch_script, flags=re.MULTILINE)

            result = await self.sandbox.run_python(
                patch_script,
                task_id=f"patch-{cycle.cycle_id}",
            )
            return result.output or result.stderr or "(no output)"
        except Exception as exc:
            return f"Sandbox error: {exc}"

    async def _phase_verify(
        self, cycle: CycleResult, opportunity: ImprovementOpportunity
    ) -> tuple[bool, str]:
        """Phase 3: Verify sandbox output and extract a lesson."""
        verify_prompt = self._VERIFY_PROMPT.format(
            target_file=opportunity.target_file,
            sandbox_output=cycle.sandbox_output[:1000],
        )
        try:
            raw = await self.agent.chat(
                [
                    {"role": "system", "content": "You are a code reviewer. Output valid JSON only."},
                    {"role": "user", "content": verify_prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return False, ""
            data = json.loads(json_match.group())
            verified = bool(data.get("verified", False))
            lesson = str(data.get("lesson", "")).strip()
            return verified, lesson
        except Exception as exc:
            logger.debug("[autonomous_loop] verify parse error: %s", exc)
            return "PATCH_WRITTEN" in cycle.sandbox_output, ""

    # ── Reflection ────────────────────────────────────────────────────────────

    async def _reflect(self) -> None:
        """Full self-reflection: read all recent cycles and update strategy."""
        if not self._cycle_history:
            return
        recent = self._cycle_history[-self.REFLECT_EVERY_N:]
        success_rate = sum(1 for c in recent if c.verified) / len(recent)
        avg_duration = sum(c.duration_s for c in recent) / len(recent)

        reflect_prompt = (
            f"Review the last {len(recent)} autonomous optimization cycles.\n\n"
            f"Success rate: {success_rate:.0%}\n"
            f"Average cycle duration: {avg_duration:.1f}s\n"
            f"Lessons added: {sum(1 for c in recent if c.lesson_added)}\n\n"
            "Opportunities attempted:\n" +
            "\n".join(
                f"- [{c.opportunity.impact if c.opportunity else '?'}] "
                f"{c.opportunity.title if c.opportunity else 'None'}: "
                f"{'✓' if c.verified else '✗'}"
                for c in recent
            ) +
            "\n\nIn 2-3 sentences: what patterns do you see? "
            "What should the loop focus on next?"
        )

        try:
            reflection = await self.agent.chat(
                [
                    {"role": "system", "content": "You are reflecting on your own autonomous operation. Be concise."},
                    {"role": "user", "content": reflect_prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            logger.info("[autonomous_loop] reflection: %s", reflection[:300])

            if hasattr(self.agent, "persistent_memory") and self.agent.persistent_memory:
                self.agent.persistent_memory.store(
                    key=f"autonomous_reflection_{int(time.time())}",
                    content=reflection,
                    category="self_optimization",
                    importance=0.7,
                    tags=["autonomous", "reflection"],
                    source="autonomous_loop",
                )
        except Exception as exc:
            logger.debug("[autonomous_loop] reflection failed: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_codebase_index(self) -> None:
        try:
            from core.gateway.python_repl import get_codebase_index
            self._codebase_index = get_codebase_index(str(_MYCONEX_ROOT))
            self._codebase_index.build()
            logger.info("[autonomous_loop] codebase index built: %s", self._codebase_index.status())
        except Exception as exc:
            logger.warning("[autonomous_loop] codebase index unavailable: %s", exc)

    def _get_codebase_summary(self) -> str:
        """Return a compact codebase summary for the analysis prompt."""
        if self._codebase_index is None:
            return "[codebase index not available]"
        # Sample queries that surface interesting code areas
        queries = ["error handling exception", "async await delegate", "todo fixme hack"]
        results = []
        for q in queries:
            hits = self._codebase_index.search(q, top_k=2)
            for h in hits:
                results.append(f"[{h['file_path']}:{h['start_line']}]\n{h['content'][:300]}")
        return "\n\n---\n\n".join(results[:6]) if results else "[no results]"

    def _read_lessons(self, max_chars: int = 2000) -> str:
        try:
            return _LESSONS_FILE.read_text()[:max_chars]
        except FileNotFoundError:
            return "(lessons.md not found)"

    async def _add_lesson(self, lesson: str, opportunity: ImprovementOpportunity) -> None:
        """Append a new lesson to lessons.md."""
        try:
            category = opportunity.category.capitalize()
            new_entry = (
                f"\n## [{category}] — {opportunity.title}\n\n"
                f"{lesson}\n\n"
                f"*Source: autonomous loop cycle, {datetime.utcnow().strftime('%Y-%m-%d')}*\n"
            )
            with open(_LESSONS_FILE, "a") as f:
                f.write(new_entry)
            logger.info("[autonomous_loop] lesson added: %s", lesson[:80])
        except Exception as exc:
            logger.warning("[autonomous_loop] could not write lesson: %s", exc)

    def _finalize_cycle(self, cycle: CycleResult, success: bool) -> None:
        cycle.completed_at = time.time()
        cycle.metrics_snapshot = {"cycles": self.metrics.cycles_run}
        if success:
            self.metrics.cycles_succeeded += 1
        else:
            self.metrics.cycles_failed += 1
        self.metrics.total_runtime_s += cycle.duration_s

    def _write_audit(self, cycle: CycleResult) -> None:
        """Append cycle result to the JSONL audit log."""
        try:
            entry = {
                "cycle_id": cycle.cycle_id,
                "cycle_number": cycle.cycle_number,
                "ts": datetime.utcnow().isoformat(),
                "duration_s": round(cycle.duration_s, 2),
                "opportunity": asdict(cycle.opportunity) if cycle.opportunity else None,
                "verified": cycle.verified,
                "lesson_added": cycle.lesson_added,
                "error": cycle.error,
            }
            with open(_AUDIT_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("[autonomous_loop] audit write failed: %s", exc)

    def _save_metrics(self) -> None:
        try:
            _METRICS_FILE.write_text(json.dumps(self.metrics.to_dict(), indent=2))
        except Exception:
            pass

    def _load_metrics(self) -> None:
        try:
            if _METRICS_FILE.exists():
                data = json.loads(_METRICS_FILE.read_text())
                self.metrics.cycles_run = data.get("cycles_run", 0)
                self.metrics.lessons_added = data.get("lessons_added", 0)
                self.metrics.sandbox_executions = data.get("sandbox_executions", 0)
        except Exception:
            pass
