"""
MYCONEX Async Sandboxed Execution
===================================
Inspired by langchain-ai/open-swe.

Run code in isolated subprocesses with:
  - Per-execution timeout
  - Memory limits (via resource module on Linux/macOS)
  - Full stdout/stderr capture
  - Parallel execution for multiple sub-tasks
  - Support for Python, Bash, and arbitrary commands

The sandbox is not a security boundary (no container isolation) — it provides
resource limits and process isolation suitable for agent-generated code on a
trusted machine.  For stronger isolation, wrap in Docker/nsjail.

Usage:
    executor = SandboxExecutor()

    # Single execution
    result = await executor.run_python("import math; print(math.pi)")

    # Parallel batch
    results = await executor.run_parallel([
        SandboxTask("python", "print('task 1')"),
        SandboxTask("python", "print('task 2')"),
        SandboxTask("bash", "echo hello"),
    ])
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Platform check: resource limits only work on Unix
_UNIX = sys.platform != "win32"


# ─── Sandbox Config ───────────────────────────────────────────────────────────

@dataclass
class SandboxConfig:
    """
    Resource limits and execution parameters for a sandbox.

    Memory and CPU limits are applied via the UNIX resource module.
    They are silently ignored on non-UNIX platforms.
    """
    timeout_s: float = 30.0          # Wall-clock timeout
    max_memory_mb: int = 512         # RSS memory limit (MB); 0 = no limit
    max_output_bytes: int = 65536    # Capture at most this many bytes of stdout+stderr
    max_processes: int = 32          # Max subprocesses spawned by the code
    allowed_env_vars: Optional[list[str]] = None  # None = inherit all; list = whitelist
    working_dir: Optional[str] = None             # cwd for the subprocess; None = tmp dir
    python_executable: str = sys.executable       # Python binary to use
    extra_env: dict = field(default_factory=dict) # Extra environment variables


# ─── Sandbox Task ─────────────────────────────────────────────────────────────

@dataclass
class SandboxTask:
    """
    A single task to run in a sandbox.

    Args:
        lang:    "python" | "bash" | "command"
        code:    Source code (python/bash) or shell command string.
        task_id: Optional identifier for tracking.
        config:  Per-task config (overrides executor defaults).
        stdin:   Optional stdin bytes.
    """
    lang: str
    code: str
    task_id: str = field(default_factory=lambda: str(int(time.time() * 1000))[-6:])
    config: Optional[SandboxConfig] = None
    stdin: Optional[bytes] = None


# ─── Execution Result ─────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    """Result of a single sandboxed execution."""
    task_id: str
    lang: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False
    memory_exceeded: bool = False
    error: Optional[str] = None

    @property
    def output(self) -> str:
        """Combined stdout + stderr for display."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr] {self.stderr.strip()}")
        if self.timed_out:
            parts.append("[TIMEOUT]")
        if self.memory_exceeded:
            parts.append("[MEMORY LIMIT EXCEEDED]")
        return "\n".join(parts) if parts else "(no output)"

    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} [{self.task_id}] {self.lang} ({self.duration_ms:.0f}ms): {self.output[:200]}"


# ─── Sandbox Executor ─────────────────────────────────────────────────────────

class SandboxExecutor:
    """
    Async sandboxed code execution engine.

    Manages a pool of subprocess executions with configurable resource limits.
    Designed for agent-generated code that must not block or crash the host.

    Usage:
        executor = SandboxExecutor(default_config=SandboxConfig(timeout_s=10))
        result = await executor.run_python("for i in range(5): print(i)")
        results = await executor.run_parallel([...])
    """

    def __init__(
        self,
        default_config: Optional[SandboxConfig] = None,
        max_parallel: int = 8,
    ) -> None:
        self.default_config = default_config or SandboxConfig()
        self._semaphore = asyncio.Semaphore(max_parallel)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_python(
        self,
        code: str,
        task_id: Optional[str] = None,
        config: Optional[SandboxConfig] = None,
        stdin: Optional[bytes] = None,
    ) -> SandboxResult:
        """Execute Python code in a sandboxed subprocess."""
        task = SandboxTask(
            lang="python", code=code,
            task_id=task_id or self._new_id(),
            config=config, stdin=stdin,
        )
        return await self._execute(task)

    async def run_bash(
        self,
        script: str,
        task_id: Optional[str] = None,
        config: Optional[SandboxConfig] = None,
        stdin: Optional[bytes] = None,
    ) -> SandboxResult:
        """Execute a bash script in a sandboxed subprocess."""
        task = SandboxTask(
            lang="bash", code=script,
            task_id=task_id or self._new_id(),
            config=config, stdin=stdin,
        )
        return await self._execute(task)

    async def run_command(
        self,
        command: str,
        task_id: Optional[str] = None,
        config: Optional[SandboxConfig] = None,
        stdin: Optional[bytes] = None,
    ) -> SandboxResult:
        """Execute an arbitrary shell command."""
        task = SandboxTask(
            lang="command", code=command,
            task_id=task_id or self._new_id(),
            config=config, stdin=stdin,
        )
        return await self._execute(task)

    async def run_parallel(
        self,
        tasks: list[SandboxTask],
        return_exceptions: bool = False,
    ) -> list[SandboxResult]:
        """
        Execute multiple tasks in parallel, respecting the max_parallel limit.

        Args:
            tasks:             List of SandboxTask objects.
            return_exceptions: If True, exceptions are returned as SandboxResults
                               rather than propagated.

        Returns:
            List of SandboxResults in the same order as tasks.
        """
        coros = [self._execute(task) for task in tasks]
        if return_exceptions:
            raw = await asyncio.gather(*coros, return_exceptions=True)
            results = []
            for task, r in zip(tasks, raw):
                if isinstance(r, Exception):
                    results.append(SandboxResult(
                        task_id=task.task_id, lang=task.lang,
                        success=False, error=str(r),
                    ))
                else:
                    results.append(r)
            return results
        return list(await asyncio.gather(*coros))

    async def run_parallel_python(
        self,
        code_snippets: list[str],
        config: Optional[SandboxConfig] = None,
    ) -> list[SandboxResult]:
        """Convenience: execute multiple Python snippets in parallel."""
        tasks = [
            SandboxTask(lang="python", code=code,
                        task_id=f"py-{i}", config=config)
            for i, code in enumerate(code_snippets)
        ]
        return await self.run_parallel(tasks)

    # ── Core Execution ────────────────────────────────────────────────────────

    async def _execute(self, task: SandboxTask) -> SandboxResult:
        cfg = task.config or self.default_config
        async with self._semaphore:
            return await self._run_subprocess(task, cfg)

    async def _run_subprocess(
        self, task: SandboxTask, cfg: SandboxConfig
    ) -> SandboxResult:
        start = time.monotonic()
        tmp_file: Optional[str] = None
        work_dir = cfg.working_dir

        try:
            # Build command
            cmd, tmp_file = self._build_command(task, cfg)

            # Build environment
            env = self._build_env(cfg)

            # Working directory
            if work_dir is None:
                work_dir = tempfile.mkdtemp(prefix="myconex_sandbox_")
                cleanup_dir = work_dir
            else:
                cleanup_dir = None

            # Preexec function for resource limits (Unix only)
            preexec = self._make_preexec(cfg) if _UNIX else None

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if task.stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=work_dir,
                preexec_fn=preexec,
            )

            timed_out = False
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=task.stdin),
                    timeout=cfg.timeout_s,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                timed_out = True
                stdout_b, stderr_b = b"", b"[killed: timeout]"

            rc = proc.returncode if proc.returncode is not None else -1
            duration = (time.monotonic() - start) * 1000

            # Decode and truncate output
            stdout = stdout_b[:cfg.max_output_bytes].decode("utf-8", errors="replace")
            stderr = stderr_b[:cfg.max_output_bytes].decode("utf-8", errors="replace")

            # Memory exceeded heuristic: check stderr for MemoryError
            mem_exceeded = "MemoryError" in stderr or "Cannot allocate memory" in stderr

            success = (rc == 0) and not timed_out and not mem_exceeded

            return SandboxResult(
                task_id=task.task_id,
                lang=task.lang,
                success=success,
                stdout=stdout,
                stderr=stderr,
                return_code=rc,
                duration_ms=duration,
                timed_out=timed_out,
                memory_exceeded=mem_exceeded,
            )

        except Exception as exc:
            logger.exception("[sandbox] unexpected error for task %s: %s", task.task_id, exc)
            return SandboxResult(
                task_id=task.task_id, lang=task.lang,
                success=False, error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            # Clean up temp file
            if tmp_file:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
            # Clean up auto-created work dir
            if cleanup_dir:  # noqa: F821 — only set above in this branch
                import shutil
                try:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                except Exception:
                    pass

    # ── Command Builder ───────────────────────────────────────────────────────

    def _build_command(
        self, task: SandboxTask, cfg: SandboxConfig
    ) -> tuple[list[str], Optional[str]]:
        """
        Build the subprocess command list.

        For Python and bash, the code is written to a temp file to avoid
        shell-injection via the command line.

        Returns:
            (command_list, temp_file_path_or_None)
        """
        if task.lang == "python":
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(task.code)
                tmp = f.name
            return [cfg.python_executable, tmp], tmp

        if task.lang == "bash":
            with tempfile.NamedTemporaryFile(
                suffix=".sh", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
                f.write(task.code)
                tmp = f.name
            os.chmod(tmp, 0o755)
            return ["bash", tmp], tmp

        if task.lang == "command":
            # Shell command: use shell=False via explicit shell expansion
            return ["bash", "-c", task.code], None

        raise ValueError(f"Unknown sandbox lang: {task.lang!r}. Use 'python', 'bash', or 'command'.")

    def _build_env(self, cfg: SandboxConfig) -> dict:
        """Build the subprocess environment from config."""
        if cfg.allowed_env_vars is None:
            env = dict(os.environ)
        else:
            env = {k: os.environ[k] for k in cfg.allowed_env_vars if k in os.environ}

        # Always pass PATH and HOME so basic tools work
        env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
        env.setdefault("HOME", os.environ.get("HOME", "/tmp"))

        env.update(cfg.extra_env)
        return env

    def _make_preexec(self, cfg: SandboxConfig):
        """Return a preexec_fn that applies resource limits (Unix only)."""
        max_mem_bytes = cfg.max_memory_mb * 1024 * 1024 if cfg.max_memory_mb else 0
        max_procs = cfg.max_processes

        def preexec():
            # Memory limit (RSS + virtual)
            if max_mem_bytes:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
                except (ValueError, resource.error):
                    pass
            # Process limit
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))
            except (ValueError, resource.error, AttributeError):
                pass
            # Nice priority (lower priority so it doesn't starve the main process)
            try:
                os.nice(5)
            except Exception:
                pass

        return preexec

    @staticmethod
    def _new_id() -> str:
        return str(int(time.time() * 1000))[-8:]


# ─── Convenience Singleton ────────────────────────────────────────────────────

_DEFAULT_EXECUTOR: Optional[SandboxExecutor] = None


def get_sandbox_executor(config: Optional[SandboxConfig] = None) -> SandboxExecutor:
    """Return the shared default SandboxExecutor, creating it if needed."""
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = SandboxExecutor(default_config=config)
    return _DEFAULT_EXECUTOR
