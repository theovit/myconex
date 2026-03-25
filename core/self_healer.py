"""
MYCONEX Self-Healing System
============================
Monitors MYCONEX health and autonomously repairs issues before they cascade.

Responsibilities:
  - Health checks: imports, config, backend connectivity, file integrity
  - Auto-recovery:  restart failed components, clear caches, rebuild indexes
  - Error pattern detection: track recurring errors, develop permanent fixes
  - Dependency checker: verify packages, auto-install missing ones
  - Watchdog: monitor memory/CPU/disk; alert and act on thresholds

Architecture:
  HealthCheck   — a single named check returning HealthResult
  Healer        — maps HealthResult.issue_type → recovery action
  SelfHealer    — orchestrates checks, triggers healing, records history
  Watchdog      — system resource monitor running in background

Usage:
    healer = SelfHealer()
    await healer.run_checks()              # one-shot
    await healer.run(interval_s=60)        # continuous watchdog loop
    report = healer.health_report()        # current status dict
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

_MYCONEX_ROOT = Path(__file__).parent.parent
_MYCONEX_DIR  = Path.home() / ".myconex"
_HEALTH_LOG   = _MYCONEX_DIR / "health_history.jsonl"
_METRICS_FILE = _MYCONEX_DIR / "metrics.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthResult:
    """Result from a single health check."""
    check_name: str
    healthy: bool
    issue_type: str = ""        # e.g. "import_error", "backend_unreachable", "disk_full"
    message: str = ""
    details: dict = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)
    recovery_attempted: bool = False
    recovery_succeeded: Optional[bool] = None

    @property
    def status(self) -> str:
        return "OK" if self.healthy else f"FAIL({self.issue_type})"


@dataclass
class SystemResources:
    """Snapshot of host resource usage."""
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_percent: float = 0.0
    process_count: int = 0
    captured_at: float = field(default_factory=time.time)

    @property
    def memory_free_mb(self) -> float:
        return self.memory_total_mb - self.memory_used_mb


@dataclass
class HealerMetrics:
    """Cumulative self-healer statistics."""
    checks_run: int = 0
    checks_failed: int = 0
    recoveries_attempted: int = 0
    recoveries_succeeded: int = 0
    last_run_at: Optional[float] = None
    uptime_start: float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.uptime_start

    @property
    def success_rate(self) -> float:
        if self.recoveries_attempted == 0:
            return 1.0
        return self.recoveries_succeeded / self.recoveries_attempted


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Health Checks
# ═══════════════════════════════════════════════════════════════════════════════

async def check_core_imports() -> HealthResult:
    """Verify all critical MYCONEX modules can be imported."""
    critical_modules = [
        "config",
        "orchestration.agents.base_agent",
        "orchestration.agents.context_manager",
        "orchestration.workflows.task_router",
        "tools.sandbox_executor",
        "tools.document_processor",
        "tools.intel_aggregator",
        "core.autonomous_loop",
    ]
    failed: list[str] = []
    for mod in critical_modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failed.append(f"{mod}: {exc}")

    healthy = len(failed) == 0
    return HealthResult(
        check_name="core_imports",
        healthy=healthy,
        issue_type="import_error" if not healthy else "",
        message=f"{len(failed)} import(s) failed" if not healthy else "all imports OK",
        details={"failed": failed},
    )


async def check_config() -> HealthResult:
    """Verify the config loads without errors and has required fields."""
    try:
        sys.path.insert(0, str(_MYCONEX_ROOT))
        from config import load_config
        cfg = load_config(apply_env=True)
        issues: list[str] = []
        if not cfg.backend.ollama.url:
            issues.append("backend.ollama.url is empty")
        if cfg.api.port <= 0 or cfg.api.port > 65535:
            issues.append(f"api.port invalid: {cfg.api.port}")
        healthy = len(issues) == 0
        return HealthResult(
            check_name="config",
            healthy=healthy,
            issue_type="config_invalid" if not healthy else "",
            message="; ".join(issues) if issues else "config OK",
            details={"backend": cfg.backend.default, "api_port": cfg.api.port},
        )
    except Exception as exc:
        return HealthResult(
            check_name="config",
            healthy=False,
            issue_type="config_load_error",
            message=str(exc),
        )


async def check_ollama_backend(url: str = "http://localhost:11434",
                                timeout: float = 5.0) -> HealthResult:
    """Check Ollama backend reachability."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/api/tags")
            healthy = resp.status_code < 500
            return HealthResult(
                check_name="ollama_backend",
                healthy=healthy,
                issue_type="" if healthy else "backend_unreachable",
                message=f"Ollama HTTP {resp.status_code}",
                details={"url": url, "status_code": resp.status_code},
            )
    except Exception as exc:
        return HealthResult(
            check_name="ollama_backend",
            healthy=False,
            issue_type="backend_unreachable",
            message=str(exc),
            details={"url": url},
        )


async def check_file_integrity() -> HealthResult:
    """Verify critical MYCONEX files exist and are valid Python."""
    critical_files = [
        "config.py",
        "orchestration/agents/base_agent.py",
        "orchestration/agents/rlm_agent.py",
        "orchestration/workflows/task_router.py",
        "core/autonomous_loop.py",
        "core/gateway/python_repl.py",
        "core/gateway/agentic_tools.py",
        "tools/sandbox_executor.py",
    ]
    issues: list[str] = []
    for rel_path in critical_files:
        full = _MYCONEX_ROOT / rel_path
        if not full.exists():
            issues.append(f"missing: {rel_path}")
            continue
        try:
            import ast
            ast.parse(full.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            issues.append(f"syntax error in {rel_path}: {exc}")
        except Exception as exc:
            issues.append(f"unreadable {rel_path}: {exc}")

    healthy = len(issues) == 0
    return HealthResult(
        check_name="file_integrity",
        healthy=healthy,
        issue_type="syntax_error" if any("syntax" in i for i in issues) else (
            "missing_file" if issues else ""
        ),
        message=f"{len(issues)} file issue(s)" if issues else "all files OK",
        details={"issues": issues},
    )


async def check_memory_dir() -> HealthResult:
    """Verify the persistent memory directory is writable."""
    try:
        from config import load_config
        cfg = load_config(apply_env=False)
        mem_dir = Path(cfg.memory.dir)
    except Exception:
        mem_dir = Path.home() / ".myconex" / "memory"

    try:
        mem_dir.mkdir(parents=True, exist_ok=True)
        test_file = mem_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return HealthResult(
            check_name="memory_dir",
            healthy=True,
            message=f"memory dir writable: {mem_dir}",
        )
    except Exception as exc:
        return HealthResult(
            check_name="memory_dir",
            healthy=False,
            issue_type="memory_dir_error",
            message=str(exc),
            details={"path": str(mem_dir)},
        )


async def check_disk_space(threshold_gb: float = 1.0) -> HealthResult:
    """Check available disk space on the MYCONEX root partition."""
    try:
        stat = shutil.disk_usage(str(_MYCONEX_ROOT))
        free_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        used_pct = (stat.used / stat.total) * 100
        healthy = free_gb >= threshold_gb
        return HealthResult(
            check_name="disk_space",
            healthy=healthy,
            issue_type="disk_low" if not healthy else "",
            message=(
                f"disk {used_pct:.0f}% used, {free_gb:.1f}GB free"
                + (" — LOW" if not healthy else "")
            ),
            details={
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_pct": round(used_pct, 1),
            },
        )
    except Exception as exc:
        return HealthResult(
            check_name="disk_space",
            healthy=True,   # Don't fail hard if we can't check
            message=f"disk check error (non-fatal): {exc}",
        )


async def check_dependencies(required: Optional[list[str]] = None) -> HealthResult:
    """
    Verify required Python packages are importable.
    Auto-installs missing packages if possible.
    """
    if required is None:
        required = [
            "httpx", "yaml", "discord", "nats", "redis",
            "qdrant_client", "pydantic",
        ]

    missing: list[str] = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    healthy = len(missing) == 0
    return HealthResult(
        check_name="dependencies",
        healthy=healthy,
        issue_type="missing_packages" if not healthy else "",
        message=f"missing packages: {missing}" if missing else "all dependencies present",
        details={"missing": missing, "checked": required},
    )


async def check_audit_log() -> HealthResult:
    """Verify the audit log is not corrupted (valid JSONL)."""
    if not _HEALTH_LOG.parent.exists():
        _HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    audit = _MYCONEX_DIR / "audit.jsonl"
    if not audit.exists():
        return HealthResult(check_name="audit_log", healthy=True,
                            message="audit log does not exist yet (OK)")
    try:
        bad_lines = 0
        with open(audit) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                if i > 1000:
                    break   # only check first 1000 lines
        healthy = bad_lines == 0
        return HealthResult(
            check_name="audit_log",
            healthy=healthy,
            issue_type="corrupted_log" if not healthy else "",
            message=f"{bad_lines} malformed lines" if not healthy else "audit log OK",
        )
    except Exception as exc:
        return HealthResult(
            check_name="audit_log", healthy=True,
            message=f"audit log check error (non-fatal): {exc}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Recovery Actions
# ═══════════════════════════════════════════════════════════════════════════════

async def recover_install_packages(missing: list[str]) -> tuple[bool, str]:
    """Attempt to pip-install missing packages."""
    if not missing:
        return True, "nothing to install"
    logger.info("[self_healer] auto-installing: %s", missing)
    try:
        result = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--quiet", *missing,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        ok = result.returncode == 0
        msg = (stdout + stderr).decode("utf-8", errors="replace")[:500]
        if ok:
            logger.info("[self_healer] install succeeded: %s", missing)
        else:
            logger.warning("[self_healer] install failed: %s", msg[:200])
        return ok, msg
    except Exception as exc:
        return False, str(exc)


async def recover_rebuild_codebase_index() -> tuple[bool, str]:
    """Rebuild the CodebaseIndex for MYCONEX self-awareness."""
    try:
        sys.path.insert(0, str(_MYCONEX_ROOT))
        from core.gateway.python_repl import get_codebase_index
        idx = get_codebase_index(str(_MYCONEX_ROOT))
        idx.build()
        return True, f"codebase index rebuilt: {idx.status()}"
    except Exception as exc:
        return False, str(exc)


async def recover_clear_cache(cache_dirs: Optional[list[str]] = None) -> tuple[bool, str]:
    """Clear Python __pycache__ directories."""
    if cache_dirs is None:
        cache_dirs = list(str(p) for p in _MYCONEX_ROOT.rglob("__pycache__")
                          if ".venv" not in str(p) and "venv" not in str(p))
    cleared = []
    for d in cache_dirs[:20]:   # limit to 20 to avoid runaway
        try:
            shutil.rmtree(d, ignore_errors=True)
            cleared.append(d)
        except Exception:
            pass
    return True, f"cleared {len(cleared)} cache dirs"


async def recover_create_memory_dir(path: str) -> tuple[bool, str]:
    """Create the persistent memory directory."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True, f"created {path}"
    except Exception as exc:
        return False, str(exc)


async def recover_truncate_corrupted_log(log_path: Path) -> tuple[bool, str]:
    """Truncate a corrupted JSONL log, keeping only valid lines."""
    try:
        valid_lines = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines.append(line)
                except json.JSONDecodeError:
                    pass
        with open(log_path, "w") as f:
            f.write("\n".join(valid_lines) + "\n")
        return True, f"kept {len(valid_lines)} valid lines"
    except Exception as exc:
        return False, str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Resource Watchdog
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceWatchdog:
    """
    Monitor system resource usage and trigger alerts/actions at thresholds.

    Thresholds (configurable):
        memory_warn_pct:  log warning above this % memory usage
        memory_crit_pct:  log critical above this % (future: kill subprocesses)
        disk_warn_pct:    log warning above this % disk usage
        cpu_warn_pct:     log warning above this % CPU usage (sustained)
    """

    def __init__(
        self,
        memory_warn_pct: float = 80.0,
        memory_crit_pct: float = 92.0,
        disk_warn_pct: float = 85.0,
        cpu_warn_pct: float = 90.0,
    ) -> None:
        self.memory_warn_pct = memory_warn_pct
        self.memory_crit_pct = memory_crit_pct
        self.disk_warn_pct   = disk_warn_pct
        self.cpu_warn_pct    = cpu_warn_pct
        self._history: list[SystemResources] = []

    def snapshot(self) -> SystemResources:
        """Capture current system resource usage (stdlib only)."""
        snap = SystemResources()

        # Memory (Linux /proc/meminfo; fallback uses resource module)
        try:
            if platform.system() == "Linux":
                meminfo = Path("/proc/meminfo").read_text()
                def _kb(key: str) -> float:
                    m = __import__("re").search(rf"{key}:\s+(\d+)", meminfo)
                    return float(m.group(1)) if m else 0.0
                total_kb = _kb("MemTotal")
                avail_kb = _kb("MemAvailable")
                snap.memory_total_mb = total_kb / 1024
                snap.memory_used_mb  = (total_kb - avail_kb) / 1024
            else:
                import resource as _resource
                usage = _resource.getrusage(_resource.RUSAGE_SELF)
                snap.memory_used_mb = usage.ru_maxrss / 1024  # approximate
                snap.memory_total_mb = snap.memory_used_mb * 4  # rough estimate
        except Exception:
            pass

        if snap.memory_total_mb > 0:
            snap.memory_percent = (snap.memory_used_mb / snap.memory_total_mb) * 100

        # Disk
        try:
            stat = shutil.disk_usage(str(_MYCONEX_ROOT))
            snap.disk_total_gb = stat.total / (1024 ** 3)
            snap.disk_used_gb  = stat.used  / (1024 ** 3)
            snap.disk_percent  = (stat.used / stat.total) * 100
        except Exception:
            pass

        # CPU (Linux /proc/stat)
        try:
            if platform.system() == "Linux":
                stat1 = Path("/proc/stat").read_text().split("\n")[0].split()
                idle1 = float(stat1[4])
                total1 = sum(float(x) for x in stat1[1:])
                time.sleep(0.1)
                stat2 = Path("/proc/stat").read_text().split("\n")[0].split()
                idle2 = float(stat2[4])
                total2 = sum(float(x) for x in stat2[1:])
                d_total = total2 - total1
                d_idle  = idle2  - idle1
                snap.cpu_percent = ((d_total - d_idle) / d_total * 100) if d_total > 0 else 0.0
        except Exception:
            pass

        self._history.append(snap)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return snap

    def evaluate(self, snap: SystemResources) -> list[str]:
        """Return a list of alert messages for threshold violations."""
        alerts: list[str] = []
        if snap.memory_percent >= self.memory_crit_pct:
            alerts.append(
                f"CRITICAL: memory at {snap.memory_percent:.0f}% "
                f"({snap.memory_used_mb:.0f}/{snap.memory_total_mb:.0f} MB)"
            )
        elif snap.memory_percent >= self.memory_warn_pct:
            alerts.append(f"WARNING: memory at {snap.memory_percent:.0f}%")
        if snap.disk_percent >= self.disk_warn_pct:
            alerts.append(
                f"WARNING: disk at {snap.disk_percent:.0f}% "
                f"({snap.disk_used_gb:.1f}/{snap.disk_total_gb:.1f} GB)"
            )
        if snap.cpu_percent >= self.cpu_warn_pct:
            alerts.append(f"WARNING: CPU at {snap.cpu_percent:.0f}%")
        return alerts

    def trend(self) -> dict:
        """Return recent resource trend from the last 10 snapshots."""
        if not self._history:
            return {}
        recent = self._history[-10:]
        return {
            "samples": len(recent),
            "avg_memory_pct": sum(s.memory_percent for s in recent) / len(recent),
            "avg_cpu_pct": sum(s.cpu_percent for s in recent) / len(recent),
            "avg_disk_pct": sum(s.disk_percent for s in recent) / len(recent),
            "latest": asdict(recent[-1]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Error Pattern Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorPatternTracker:
    """
    Track recurring error patterns across health checks.

    When a pattern is seen more than `threshold` times, it is flagged as
    a persistent issue that warrants a permanent fix (written to lessons.md).
    """

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._counts: dict[str, int] = {}
        self._first_seen: dict[str, float] = {}
        self._escalated: set[str] = set()

    def record(self, issue_type: str, message: str) -> bool:
        """Record an issue. Returns True if it just crossed the escalation threshold."""
        key = f"{issue_type}:{message[:60]}"
        self._counts[key] = self._counts.get(key, 0) + 1
        if key not in self._first_seen:
            self._first_seen[key] = time.time()
        crossed = (
            self._counts[key] >= self.threshold
            and key not in self._escalated
        )
        if crossed:
            self._escalated.add(key)
        return crossed

    def persistent_issues(self) -> list[dict]:
        """Return issues that have crossed the escalation threshold."""
        return [
            {
                "key": k,
                "count": self._counts[k],
                "first_seen": datetime.fromtimestamp(
                    self._first_seen.get(k, 0), tz=timezone.utc
                ).isoformat(),
            }
            for k in self._escalated
        ]

    def clear(self, issue_key: str) -> None:
        """Clear a resolved issue from the tracker."""
        self._counts.pop(issue_key, None)
        self._first_seen.pop(issue_key, None)
        self._escalated.discard(issue_key)

    async def write_to_lessons(self, issue_type: str, message: str) -> None:
        """Append a persistent error lesson to lessons.md."""
        try:
            entry = (
                f"\n## [Self-Healer] Persistent issue: {issue_type}\n\n"
                f"Pattern detected {self._counts.get(f'{issue_type}:{message[:60]}', '?')}x:\n"
                f"  {message}\n\n"
                f"*Escalated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n"
            )
            with open(_LESSONS_FILE, "a") as f:
                f.write(entry)
            logger.info("[error_tracker] lesson written for persistent issue: %s", issue_type)
        except Exception as exc:
            logger.debug("[error_tracker] lessons.md write failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Healer Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealer:
    """
    Autonomous health monitoring and repair system for MYCONEX.

    Runs a suite of health checks, applies targeted recovery actions for
    known failure modes, tracks recurring patterns, and monitors system
    resources via the ResourceWatchdog.

    Usage:
        healer = SelfHealer()
        await healer.run_checks()         # single pass
        await healer.run(interval_s=60)   # continuous loop
        print(healer.health_report())
    """

    # Default interval between check passes
    DEFAULT_INTERVAL_S: float = 60.0

    # Recovery registry: issue_type → coroutine factory
    _RECOVERY_MAP: dict[str, str] = {
        "missing_packages": "_recover_missing_packages",
        "memory_dir_error": "_recover_memory_dir",
        "corrupted_log":    "_recover_corrupted_log",
        "config_load_error": "_recover_clear_cache",
        "import_error":     "_recover_clear_cache",
    }

    def __init__(
        self,
        watchdog: Optional[ResourceWatchdog] = None,
        check_ollama: bool = True,
        ollama_url: str = "http://localhost:11434",
        pattern_threshold: int = 3,
    ) -> None:
        self.watchdog = watchdog or ResourceWatchdog()
        self.check_ollama = check_ollama
        self.ollama_url = ollama_url
        self.pattern_tracker = ErrorPatternTracker(threshold=pattern_threshold)
        self.metrics = HealerMetrics()
        self._stop = asyncio.Event()
        self._last_results: list[HealthResult] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_checks(self) -> list[HealthResult]:
        """
        Execute all health checks once and apply recovery where possible.
        Returns the list of HealthResults.
        """
        checks = [
            check_core_imports(),
            check_config(),
            check_file_integrity(),
            check_memory_dir(),
            check_disk_space(),
            check_dependencies(),
            check_audit_log(),
        ]
        if self.check_ollama:
            checks.append(check_ollama_backend(self.ollama_url))

        results = await asyncio.gather(*checks, return_exceptions=True)
        health_results: list[HealthResult] = []

        for r in results:
            if isinstance(r, Exception):
                health_results.append(HealthResult(
                    check_name="unknown",
                    healthy=False,
                    issue_type="check_exception",
                    message=str(r),
                ))
            else:
                health_results.append(r)

        self.metrics.checks_run += len(health_results)
        self.metrics.last_run_at = time.time()
        self._last_results = health_results

        # Count failures and attempt recovery
        for result in health_results:
            if not result.healthy:
                self.metrics.checks_failed += 1
                escalated = self.pattern_tracker.record(result.issue_type, result.message)
                if escalated:
                    asyncio.create_task(
                        self.pattern_tracker.write_to_lessons(result.issue_type, result.message)
                    )
                await self._attempt_recovery(result)

        self._log_health(health_results)

        healthy_count = sum(1 for r in health_results if r.healthy)
        logger.info(
            "[self_healer] checks complete: %d/%d healthy",
            healthy_count, len(health_results),
        )
        return health_results

    async def run(
        self,
        interval_s: Optional[float] = None,
        max_cycles: Optional[int] = None,
    ) -> None:
        """
        Run continuous health monitoring.

        Args:
            interval_s:  Seconds between check passes (default: 60).
            max_cycles:  Stop after N passes (None = forever).
        """
        effective_interval = interval_s or self.DEFAULT_INTERVAL_S
        cycle = 0
        logger.info("[self_healer] starting (interval=%ss)", effective_interval)

        while not self._stop.is_set():
            # Resource watchdog snapshot
            snap = self.watchdog.snapshot()
            alerts = self.watchdog.evaluate(snap)
            for alert in alerts:
                logger.warning("[watchdog] %s", alert)

            # Health checks
            await self.run_checks()
            cycle += 1

            if max_cycles is not None and cycle >= max_cycles:
                logger.info("[self_healer] reached max_cycles=%d", max_cycles)
                break

            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=effective_interval)
                except asyncio.TimeoutError:
                    pass

    def stop(self) -> None:
        """Stop the monitoring loop after the current pass."""
        self._stop.set()

    def health_report(self) -> dict:
        """Return current health status as a serialisable dict."""
        overall = all(r.healthy for r in self._last_results)
        return {
            "overall": "healthy" if overall else "degraded",
            "checked_at": (
                datetime.fromtimestamp(self.metrics.last_run_at, tz=timezone.utc).isoformat()
                if self.metrics.last_run_at else None
            ),
            "checks": [
                {
                    "name": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "recovery": (
                        "succeeded" if r.recovery_succeeded
                        else "failed" if r.recovery_succeeded is False
                        else "not_attempted"
                    ),
                }
                for r in self._last_results
            ],
            "metrics": asdict(self.metrics),
            "resources": self.watchdog.trend(),
            "persistent_issues": self.pattern_tracker.persistent_issues(),
        }

    # ── Recovery Dispatch ─────────────────────────────────────────────────────

    async def _attempt_recovery(self, result: HealthResult) -> None:
        """Dispatch to the appropriate recovery handler for a failed check."""
        handler_name = self._RECOVERY_MAP.get(result.issue_type)
        if not handler_name:
            logger.debug("[self_healer] no recovery for issue_type=%s", result.issue_type)
            return

        handler = getattr(self, handler_name, None)
        if handler is None:
            return

        logger.info("[self_healer] attempting recovery: %s → %s",
                    result.issue_type, handler_name)
        self.metrics.recoveries_attempted += 1
        result.recovery_attempted = True

        try:
            ok, msg = await handler(result)
            result.recovery_succeeded = ok
            if ok:
                self.metrics.recoveries_succeeded += 1
                logger.info("[self_healer] recovery succeeded (%s): %s",
                            result.check_name, msg[:100])
            else:
                logger.warning("[self_healer] recovery failed (%s): %s",
                               result.check_name, msg[:100])
        except Exception as exc:
            result.recovery_succeeded = False
            logger.error("[self_healer] recovery exception: %s", exc)

    async def _recover_missing_packages(self, result: HealthResult) -> tuple[bool, str]:
        missing = result.details.get("missing", [])
        return await recover_install_packages(missing)

    async def _recover_memory_dir(self, result: HealthResult) -> tuple[bool, str]:
        path = result.details.get("path", str(Path.home() / ".myconex" / "memory"))
        return await recover_create_memory_dir(path)

    async def _recover_corrupted_log(self, result: HealthResult) -> tuple[bool, str]:
        return await recover_truncate_corrupted_log(_MYCONEX_DIR / "audit.jsonl")

    async def _recover_clear_cache(self, result: HealthResult) -> tuple[bool, str]:
        return await recover_clear_cache()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_health(self, results: list[HealthResult]) -> None:
        """Append a health snapshot to health_history.jsonl."""
        try:
            _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall": all(r.healthy for r in results),
                "checks": {r.check_name: r.status for r in results},
                "resources": self.watchdog.trend() if self.watchdog._history else {},
            }
            with open(_HEALTH_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.debug("[self_healer] health log write failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_self_healer(
    ollama_url: str = "http://localhost:11434",
    check_ollama: bool = True,
    pattern_threshold: int = 3,
) -> SelfHealer:
    """Create a SelfHealer with standard configuration."""
    return SelfHealer(
        check_ollama=check_ollama,
        ollama_url=ollama_url,
        pattern_threshold=pattern_threshold,
    )
