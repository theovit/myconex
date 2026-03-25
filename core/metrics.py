"""
MYCONEX Comprehensive Metrics System
======================================
Tracks all operational metrics across the MYCONEX mesh, computes trends,
exports JSON snapshots, and writes periodic reports to metrics_history.json.

Tracked dimensions:
  Tasks          — completed, success/failure, avg response time, by-agent
  Tokens         — usage per call, budget consumption, model distribution
  Tools          — call frequency, latency, error rate per tool
  Delegation     — depth distribution, delegation rate, sub-task counts
  Memory         — store size, hit rate, eviction count
  System         — uptime, cycles, self-optimization outcomes

Usage:
    metrics = MetricsCollector()

    # Record events
    metrics.record_task("Research quantum computing", success=True, duration_ms=1200.0,
                        agent="rlm-primary", tokens_used=840)
    metrics.record_tool_call("python_repl", duration_ms=45.0, success=True)
    metrics.record_delegation(depth=2, sub_tasks=3)

    # Query
    report = metrics.report()
    metrics.export_json()               # writes ~/.myconex/metrics.json
    metrics.write_periodic_report()     # appends to metrics_history.json
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MYCONEX_DIR     = Path.home() / ".myconex"
_METRICS_FILE    = _MYCONEX_DIR / "metrics.json"
_HISTORY_FILE    = _MYCONEX_DIR / "metrics_history.json"
_REPORT_INTERVAL = 3600.0   # seconds between periodic reports


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling Statistics Helper
# ═══════════════════════════════════════════════════════════════════════════════

class RollingStats:
    """
    Maintains a fixed-size window of numeric observations.
    Provides count, mean, min, max, p50, p95 without external deps.
    """

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._data: deque[float] = deque(maxlen=window)

    def record(self, value: float) -> None:
        self._data.append(value)

    def count(self) -> int:
        return len(self._data)

    def mean(self) -> float:
        if not self._data:
            return 0.0
        return sum(self._data) / len(self._data)

    def min(self) -> float:
        return min(self._data) if self._data else 0.0

    def max(self) -> float:
        return max(self._data) if self._data else 0.0

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) of the recorded values."""
        if not self._data:
            return 0.0
        sorted_data = sorted(self._data)
        idx = (p / 100) * (len(sorted_data) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(sorted_data) - 1)
        frac = idx - lo
        return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac

    def to_dict(self) -> dict:
        return {
            "count": self.count(),
            "mean": round(self.mean(), 2),
            "min": round(self.min(), 2),
            "max": round(self.max(), 2),
            "p50": round(self.percentile(50), 2),
            "p95": round(self.percentile(95), 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskMetricsSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    duration_ms: dict = field(default_factory=dict)    # RollingStats.to_dict()
    by_agent: dict = field(default_factory=dict)       # agent_name → count

    @property
    def success_rate(self) -> float:
        return (self.succeeded / self.total) if self.total > 0 else 0.0


@dataclass
class TokenMetricsSummary:
    total_tokens_used: int = 0
    calls: int = 0
    per_call: dict = field(default_factory=dict)       # RollingStats.to_dict()
    by_model: dict = field(default_factory=dict)       # model → token_count
    budget_used_pct: float = 0.0


@dataclass
class ToolMetricsSummary:
    by_tool: dict = field(default_factory=dict)        # tool → {calls, errors, duration}


@dataclass
class DelegationMetricsSummary:
    total_delegations: int = 0
    delegation_rate: float = 0.0      # delegations / tasks
    depth_distribution: dict = field(default_factory=dict)  # depth → count
    avg_sub_tasks: float = 0.0


@dataclass
class MemoryMetricsSummary:
    store_size: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


@dataclass
class SystemMetricsSummary:
    uptime_s: float = 0.0
    autonomous_cycles: int = 0
    autonomous_success_rate: float = 0.0
    novelty_scans: int = 0
    health_checks: int = 0
    self_healer_recoveries: int = 0
    start_time: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# Trend Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TrendAnalyser:
    """
    Compares current metric windows to historical baselines loaded from
    metrics_history.json.  Returns delta and trend direction for each dimension.
    """

    def __init__(self, history_file: Path = _HISTORY_FILE) -> None:
        self._history_file = history_file

    def load_baseline(self, lookback_entries: int = 10) -> Optional[dict]:
        """Load and average the last N snapshots as a baseline."""
        if not self._history_file.exists():
            return None
        try:
            entries = json.loads(self._history_file.read_text() or "[]")
            if not entries:
                return None
            recent = entries[-lookback_entries:] if len(entries) >= lookback_entries else entries
            if not recent:
                return None
            # Average numeric leaf values
            return self._average_snapshots(recent)
        except (json.JSONDecodeError, OSError):
            return None

    def compare(self, current: dict, baseline: dict) -> dict:
        """
        Return a flat dict of {metric_key: {current, baseline, delta_pct, trend}}.
        Only includes top-level numeric metrics.
        """
        results: dict = {}
        keys_to_compare = [
            ("tasks.success_rate",    current.get("tasks", {}).get("success_rate", 0)),
            ("tasks.total",           current.get("tasks", {}).get("total", 0)),
            ("tasks.duration_ms_p95", current.get("tasks", {}).get("duration_ms", {}).get("p95", 0)),
            ("tokens.total_tokens_used", current.get("tokens", {}).get("total_tokens_used", 0)),
            ("tokens.budget_used_pct",   current.get("tokens", {}).get("budget_used_pct", 0)),
            ("delegation.delegation_rate", current.get("delegation", {}).get("delegation_rate", 0)),
            ("memory.hit_rate",        current.get("memory", {}).get("hit_rate", 0)),
        ]

        for key, cur_val in keys_to_compare:
            parts = key.split(".")
            base_val = baseline
            for part in parts:
                if isinstance(base_val, dict):
                    base_val = base_val.get(part, 0)
                else:
                    base_val = 0
                    break

            if isinstance(base_val, (int, float)) and base_val != 0:
                delta_pct = ((cur_val - base_val) / base_val) * 100
            else:
                delta_pct = 0.0

            trend = "stable"
            if delta_pct > 5:
                trend = "improving" if key in (
                    "tasks.success_rate", "memory.hit_rate"
                ) else "increasing"
            elif delta_pct < -5:
                trend = "degrading" if key in (
                    "tasks.success_rate", "memory.hit_rate"
                ) else "decreasing"

            results[key] = {
                "current": round(cur_val, 4),
                "baseline": round(base_val, 4) if isinstance(base_val, (int, float)) else 0,
                "delta_pct": round(delta_pct, 2),
                "trend": trend,
            }
        return results

    def _average_snapshots(self, snapshots: list[dict]) -> dict:
        """Recursively average numeric values across a list of dicts."""
        if not snapshots:
            return {}
        result: dict = {}
        for key in snapshots[0]:
            values = [s[key] for s in snapshots if key in s]
            if all(isinstance(v, (int, float)) for v in values):
                result[key] = sum(values) / len(values)
            elif all(isinstance(v, dict) for v in values):
                result[key] = self._average_snapshots(values)
            else:
                result[key] = values[-1] if values else None
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Central metrics registry for MYCONEX.

    Thread-safe (GIL protects simple attribute updates in CPython).
    All recording methods are synchronous for low-overhead inline calls.

    Usage:
        metrics = MetricsCollector()
        metrics.record_task("do X", success=True, duration_ms=230, agent="rlm-1")
        metrics.record_tool_call("python_repl", duration_ms=40, success=True)
        print(metrics.report())
        metrics.export_json()
    """

    def __init__(
        self,
        token_budget: int = 16384,
        report_interval_s: float = _REPORT_INTERVAL,
    ) -> None:
        self.token_budget = token_budget
        self.report_interval_s = report_interval_s
        self._start_time = time.time()
        self._last_report_at: float = 0.0

        # ── Task counters ──────────────────────────────────────────────────
        self._task_total: int = 0
        self._task_succeeded: int = 0
        self._task_failed: int = 0
        self._task_duration = RollingStats(window=500)
        self._task_by_agent: dict[str, int] = defaultdict(int)

        # ── Token counters ─────────────────────────────────────────────────
        self._tokens_total: int = 0
        self._token_calls: int = 0
        self._tokens_per_call = RollingStats(window=500)
        self._tokens_by_model: dict[str, int] = defaultdict(int)

        # ── Tool counters ──────────────────────────────────────────────────
        self._tool_calls: dict[str, int] = defaultdict(int)
        self._tool_errors: dict[str, int] = defaultdict(int)
        self._tool_duration: dict[str, RollingStats] = defaultdict(lambda: RollingStats(200))

        # ── Delegation counters ────────────────────────────────────────────
        self._delegations: int = 0
        self._depth_distribution: dict[int, int] = defaultdict(int)
        self._sub_task_counts = RollingStats(window=200)

        # ── Memory counters ────────────────────────────────────────────────
        self._memory_store_size: int = 0
        self._memory_hits: int = 0
        self._memory_misses: int = 0
        self._memory_evictions: int = 0

        # ── System counters ────────────────────────────────────────────────
        self._autonomous_cycles: int = 0
        self._autonomous_succeeded: int = 0
        self._novelty_scans: int = 0
        self._health_checks: int = 0
        self._healer_recoveries: int = 0

        # ── Trend analyser ─────────────────────────────────────────────────
        self._trend = TrendAnalyser()

    # ── Recording API ─────────────────────────────────────────────────────────

    def record_task(
        self,
        task: str,
        *,
        success: bool,
        duration_ms: float = 0.0,
        agent: str = "unknown",
        tokens_used: int = 0,
        model: str = "unknown",
        delegation_depth: int = 0,
    ) -> None:
        """Record a completed task."""
        self._task_total += 1
        if success:
            self._task_succeeded += 1
        else:
            self._task_failed += 1
        if duration_ms > 0:
            self._task_duration.record(duration_ms)
        self._task_by_agent[agent] += 1

        if tokens_used > 0:
            self.record_tokens(tokens_used, model=model)

    def record_tokens(
        self,
        count: int,
        *,
        model: str = "unknown",
    ) -> None:
        """Record token usage for a single LLM call."""
        self._tokens_total += count
        self._token_calls += 1
        self._tokens_per_call.record(float(count))
        self._tokens_by_model[model] += count

    def record_tool_call(
        self,
        tool_name: str,
        *,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record a single tool invocation."""
        self._tool_calls[tool_name] += 1
        if not success:
            self._tool_errors[tool_name] += 1
        if duration_ms > 0:
            self._tool_duration[tool_name].record(duration_ms)

    def record_delegation(
        self,
        *,
        depth: int = 1,
        sub_tasks: int = 1,
    ) -> None:
        """Record a delegation event."""
        self._delegations += 1
        self._depth_distribution[depth] += 1
        self._sub_task_counts.record(float(sub_tasks))

    def record_memory_event(
        self,
        *,
        hit: Optional[bool] = None,
        store_delta: int = 0,
        evictions: int = 0,
    ) -> None:
        """Record a memory store event."""
        if hit is True:
            self._memory_hits += 1
        elif hit is False:
            self._memory_misses += 1
        self._memory_store_size = max(0, self._memory_store_size + store_delta)
        self._memory_evictions += evictions

    def record_autonomous_cycle(self, *, success: bool) -> None:
        """Record an autonomous optimization cycle."""
        self._autonomous_cycles += 1
        if success:
            self._autonomous_succeeded += 1

    def record_novelty_scan(self) -> None:
        self._novelty_scans += 1

    def record_health_check(self, *, recoveries: int = 0) -> None:
        self._health_checks += 1
        self._healer_recoveries += recoveries

    # ── Reporting API ─────────────────────────────────────────────────────────

    def report(self) -> dict:
        """
        Return a full metrics snapshot as a JSON-serialisable dict.
        Includes trend comparison to historical baseline.
        """
        uptime = time.time() - self._start_time

        current: dict = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "uptime_s": round(uptime, 1),
            "tasks": {
                "total": self._task_total,
                "succeeded": self._task_succeeded,
                "failed": self._task_failed,
                "success_rate": round(
                    self._task_succeeded / self._task_total, 4
                ) if self._task_total else 0.0,
                "duration_ms": self._task_duration.to_dict(),
                "by_agent": dict(self._task_by_agent),
            },
            "tokens": {
                "total_tokens_used": self._tokens_total,
                "calls": self._token_calls,
                "per_call": self._tokens_per_call.to_dict(),
                "by_model": dict(self._tokens_by_model),
                "budget_used_pct": round(
                    (self._tokens_total / self.token_budget) * 100, 2
                ) if self.token_budget else 0.0,
            },
            "tools": {
                tool: {
                    "calls": self._tool_calls[tool],
                    "errors": self._tool_errors.get(tool, 0),
                    "error_rate": round(
                        self._tool_errors.get(tool, 0) / self._tool_calls[tool], 4
                    ) if self._tool_calls[tool] else 0.0,
                    "duration_ms": self._tool_duration[tool].to_dict()
                    if tool in self._tool_duration else {},
                }
                for tool in sorted(self._tool_calls)
            },
            "delegation": {
                "total_delegations": self._delegations,
                "delegation_rate": round(
                    self._delegations / self._task_total, 4
                ) if self._task_total else 0.0,
                "depth_distribution": {
                    str(k): v for k, v in sorted(self._depth_distribution.items())
                },
                "sub_tasks": self._sub_task_counts.to_dict(),
            },
            "memory": {
                "store_size": self._memory_store_size,
                "hits": self._memory_hits,
                "misses": self._memory_misses,
                "hit_rate": round(
                    self._memory_hits / (self._memory_hits + self._memory_misses), 4
                ) if (self._memory_hits + self._memory_misses) else 0.0,
                "evictions": self._memory_evictions,
            },
            "system": {
                "uptime_s": round(uptime, 1),
                "autonomous_cycles": self._autonomous_cycles,
                "autonomous_success_rate": round(
                    self._autonomous_succeeded / self._autonomous_cycles, 4
                ) if self._autonomous_cycles else 0.0,
                "novelty_scans": self._novelty_scans,
                "health_checks": self._health_checks,
                "self_healer_recoveries": self._healer_recoveries,
            },
        }

        # Trend analysis
        baseline = self._trend.load_baseline()
        if baseline:
            current["trends"] = self._trend.compare(current, baseline)

        return current

    def export_json(self, path: Optional[Path] = None) -> Path:
        """Write the current metrics snapshot to metrics.json."""
        target = path or _METRICS_FILE
        _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(json.dumps(self.report(), indent=2))
            logger.debug("[metrics] exported to %s", target)
        except OSError as exc:
            logger.warning("[metrics] export failed: %s", exc)
        return target

    def write_periodic_report(self, force: bool = False) -> bool:
        """
        Append a snapshot to metrics_history.json if the report interval has passed.

        Args:
            force: Write immediately regardless of interval.

        Returns:
            True if a report was written.
        """
        now = time.time()
        if not force and (now - self._last_report_at) < self.report_interval_s:
            return False

        _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
        try:
            existing: list = []
            if _HISTORY_FILE.exists():
                try:
                    existing = json.loads(_HISTORY_FILE.read_text()) or []
                except (json.JSONDecodeError, OSError):
                    existing = []

            snap = self.report()
            # Keep last 500 entries
            existing.append(snap)
            if len(existing) > 500:
                existing = existing[-500:]

            _HISTORY_FILE.write_text(json.dumps(existing, indent=2))
            self._last_report_at = now
            logger.info("[metrics] periodic report written (%d history entries)", len(existing))
            return True
        except OSError as exc:
            logger.warning("[metrics] history write failed: %s", exc)
            return False

    def maybe_write_periodic(self) -> bool:
        """Write periodic report if interval has passed."""
        return self.write_periodic_report(force=False)

    def summary_line(self) -> str:
        """Return a one-line human-readable summary."""
        rate = f"{self._task_succeeded}/{self._task_total}"
        avg = f"{self._task_duration.mean():.0f}ms avg"
        tok = f"{self._tokens_total:,} tokens"
        delegations = f"{self._delegations} delegations"
        return f"tasks={rate} {avg} | {tok} | {delegations}"

    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        self.__init__(
            token_budget=self.token_budget,
            report_interval_s=self.report_interval_s,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_METRICS: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Return the global MetricsCollector singleton."""
    global _GLOBAL_METRICS
    if _GLOBAL_METRICS is None:
        try:
            from config import get_config
            cfg = get_config()
            _GLOBAL_METRICS = MetricsCollector(token_budget=cfg.tokens.context_budget)
        except Exception:
            _GLOBAL_METRICS = MetricsCollector()
    return _GLOBAL_METRICS


def reset_metrics() -> None:
    """Reset the global singleton (useful in tests)."""
    global _GLOBAL_METRICS
    _GLOBAL_METRICS = None
