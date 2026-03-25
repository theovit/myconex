"""
MYCONEX Plugin System
======================
Dynamically load, validate, and hot-reload capability plugins from the
plugins/ directory.  Plugins can contribute:
  - New agent tools (registered into agentic_tools registry)
  - Agent specialties (registered into AgentRoster)
  - New source types for IntelAggregator
  - Arbitrary startup/teardown hooks

Plugin contract (each plugin is a Python module with these attributes):
  __plugin_name__:    str            required — unique identifier
  __plugin_version__: str            required — semver string
  __plugin_tools__:   list[dict]     optional — tool schemas to register
  __plugin_deps__:    list[str]      optional — required pip packages
  __plugin_agent_specialties__: list[str]  optional — specialty tags

  async def plugin_setup(registry: PluginRegistry) -> None:   optional
  async def plugin_teardown() -> None:                         optional

Usage:
    loader = PluginLoader()
    await loader.load_all()
    await loader.enable("my-plugin")
    await loader.reload("my-plugin")

    # Marketplace
    available = await loader.list_marketplace()
    await loader.install_from_marketplace("vector-db-plugin")
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_MYCONEX_ROOT  = Path(__file__).parent.parent
_PLUGINS_DIR   = _MYCONEX_ROOT / "plugins"
_MYCONEX_DIR   = Path.home() / ".myconex"
_PLUGIN_STATE  = _MYCONEX_DIR / "plugin_state.json"

# Default marketplace registry URL (can be overridden via config)
_DEFAULT_MARKETPLACE_URL = "https://raw.githubusercontent.com/myconex/plugin-registry/main/registry.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PluginMeta:
    """Metadata parsed from a plugin module."""
    name: str
    version: str
    path: str
    enabled: bool = True
    tools: list[dict] = field(default_factory=list)
    agent_specialties: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    loaded_at: Optional[float] = None
    file_hash: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MarketplaceEntry:
    """Entry from the plugin marketplace registry."""
    name: str
    version: str
    description: str
    author: str
    download_url: str
    deps: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    stars: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Registry (shared state available to plugins during setup)
# ═══════════════════════════════════════════════════════════════════════════════

class PluginRegistry:
    """
    Shared registry passed to each plugin's plugin_setup() call.
    Plugins use this to register tools, hooks, and specialties.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}
        self._hooks: dict[str, list[Callable]] = {}
        self._specialties: dict[str, list[str]] = {}  # plugin_name → specialties

    def register_tool(self, plugin_name: str, schema: dict, handler: Callable) -> None:
        """Register a tool schema + handler contributed by a plugin."""
        tool_name = schema.get("name", "")
        if not tool_name:
            logger.warning("[plugin_registry] tool schema missing 'name', skipping")
            return
        self._tools[tool_name] = {"schema": schema, "handler": handler, "plugin": plugin_name}
        logger.debug("[plugin_registry] tool registered: %s (from %s)", tool_name, plugin_name)

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register an event hook. Events: 'startup', 'shutdown', 'task_complete'."""
        self._hooks.setdefault(event, []).append(callback)

    def register_specialty(self, plugin_name: str, specialties: list[str]) -> None:
        """Register agent specialties contributed by a plugin."""
        self._specialties[plugin_name] = specialties

    def get_tools(self) -> dict[str, dict]:
        return dict(self._tools)

    def get_hooks(self, event: str) -> list[Callable]:
        return list(self._hooks.get(event, []))

    async def fire_hooks(self, event: str, **kwargs: Any) -> None:
        """Fire all registered hooks for an event."""
        for cb in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(**kwargs)
                else:
                    cb(**kwargs)
            except Exception as exc:
                logger.warning("[plugin_registry] hook error (%s): %s", event, exc)

    def wire_into_agentic_tools(self) -> int:
        """
        Attempt to wire registered tools into the agentic_tools registry.
        Returns count of tools successfully wired.
        """
        wired = 0
        try:
            from core.gateway.agentic_tools import register_tool
            for tool_name, entry in self._tools.items():
                try:
                    register_tool(entry["schema"], entry["handler"])
                    wired += 1
                except Exception as exc:
                    logger.warning("[plugin_registry] wire failed for %s: %s", tool_name, exc)
        except ImportError:
            logger.debug("[plugin_registry] agentic_tools not available for wiring")
        return wired


# ═══════════════════════════════════════════════════════════════════════════════
# File Watcher (polling-based hot-reload trigger)
# ═══════════════════════════════════════════════════════════════════════════════

class PluginFileWatcher:
    """
    Polls the plugins/ directory for changes.
    Uses file content hash to detect modifications.
    """

    def __init__(self, plugins_dir: Path = _PLUGINS_DIR, poll_interval_s: float = 5.0) -> None:
        self._dir = plugins_dir
        self._interval = poll_interval_s
        self._hashes: dict[str, str] = {}
        self._callbacks: list[Callable[[str, str], None]] = []  # (path, event)
        self._stop = asyncio.Event()

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback(file_path, event) — event: 'added'|'modified'|'removed'."""
        self._callbacks.append(callback)

    async def watch(self) -> None:
        """Poll for changes until stopped."""
        logger.info("[plugin_watcher] watching %s (interval=%.1fs)", self._dir, self._interval)
        while not self._stop.is_set():
            await self._check()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def _check(self) -> None:
        if not self._dir.exists():
            return
        current: dict[str, str] = {}
        for py_file in self._dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                h = hashlib.md5(py_file.read_bytes()).hexdigest()
                current[str(py_file)] = h
            except OSError:
                continue

        # Detect added / modified
        for path, h in current.items():
            if path not in self._hashes:
                self._notify(path, "added")
            elif self._hashes[path] != h:
                self._notify(path, "modified")

        # Detect removed
        for path in set(self._hashes) - set(current):
            self._notify(path, "removed")

        self._hashes = current

    def _notify(self, path: str, event: str) -> None:
        for cb in self._callbacks:
            try:
                cb(path, event)
            except Exception as exc:
                logger.debug("[plugin_watcher] callback error: %s", exc)

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def hash_file(path: Path) -> str:
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_plugin_module(module: Any, path: str) -> ValidationResult:
    """
    Validate a loaded plugin module against the plugin contract.
    Returns ValidationResult with error/warning lists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Required attributes
    if not hasattr(module, "__plugin_name__") or not module.__plugin_name__:
        errors.append("missing __plugin_name__")
    if not hasattr(module, "__plugin_version__") or not module.__plugin_version__:
        errors.append("missing __plugin_version__")

    # Version format check (lenient semver)
    version = getattr(module, "__plugin_version__", "")
    if version and not all(p.isdigit() for p in version.split(".")[:3] if p):
        warnings.append(f"non-semver version: {version!r}")

    # Tools validation
    tools = getattr(module, "__plugin_tools__", [])
    if tools:
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                errors.append(f"__plugin_tools__[{i}] must be a dict")
            elif "name" not in tool:
                errors.append(f"__plugin_tools__[{i}] missing 'name'")

    # Deps validation
    deps = getattr(module, "__plugin_deps__", [])
    if deps and not isinstance(deps, list):
        errors.append("__plugin_deps__ must be a list")

    # Conflict detection (name collision with builtins)
    _RESERVED = {"python_repl", "web_read", "codebase_search", "gguf_infer",
                 "memory_store", "memory_retrieve", "delegate"}
    for tool in tools or []:
        if isinstance(tool, dict) and tool.get("name") in _RESERVED:
            warnings.append(f"tool name {tool['name']!r} conflicts with built-in tool")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


async def check_plugin_deps(deps: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which deps are installed.
    Returns (available, missing).
    """
    available: list[str] = []
    missing: list[str] = []
    for dep in deps:
        try:
            importlib.import_module(dep.replace("-", "_"))
            available.append(dep)
        except ImportError:
            missing.append(dep)
    return available, missing


async def install_plugin_deps(deps: list[str]) -> bool:
    """pip-install missing deps. Returns True on success."""
    if not deps:
        return True
    logger.info("[plugin_loader] installing deps: %s", deps)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--quiet", *deps,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            logger.warning("[plugin_loader] pip failed: %s", stderr.decode()[:200])
        return proc.returncode == 0
    except Exception as exc:
        logger.error("[plugin_loader] dep install error: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Loader
# ═══════════════════════════════════════════════════════════════════════════════

class PluginLoader:
    """
    Manages the full lifecycle of MYCONEX plugins.

    - Discovery: scans plugins/ directory for .py modules
    - Validation: syntax-check, contract enforcement, dep resolution
    - Loading: importlib.util.spec_from_file_location into isolated namespace
    - Hot-reload: file watcher → automatic reload on change
    - Marketplace: fetch registry, download and install plugins
    """

    def __init__(
        self,
        plugins_dir: Path = _PLUGINS_DIR,
        marketplace_url: str = _DEFAULT_MARKETPLACE_URL,
        auto_install_deps: bool = False,
        hot_reload: bool = True,
    ) -> None:
        self._dir = plugins_dir
        self._marketplace_url = marketplace_url
        self._auto_install_deps = auto_install_deps
        self._hot_reload = hot_reload

        self.registry = PluginRegistry()
        self._plugins: dict[str, PluginMeta] = {}
        self._modules: dict[str, Any] = {}
        self._watcher: Optional[PluginFileWatcher] = None
        self._watcher_task: Optional[asyncio.Task] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_all(self) -> dict[str, PluginMeta]:
        """
        Scan plugins/ and load all valid plugins.
        Returns {plugin_name: PluginMeta} for all loaded plugins.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_init_py()

        py_files = sorted(p for p in self._dir.glob("*.py") if not p.name.startswith("_"))
        logger.info("[plugin_loader] scanning %s: %d files", self._dir, len(py_files))

        for path in py_files:
            await self._load_file(path)

        if self._hot_reload:
            self._start_watcher()

        self._save_state()
        logger.info("[plugin_loader] loaded %d/%d plugins",
                    sum(1 for m in self._plugins.values() if m.ok), len(py_files))
        return dict(self._plugins)

    async def enable(self, plugin_name: str) -> bool:
        """Enable a loaded plugin and run its setup hook."""
        meta = self._plugins.get(plugin_name)
        if not meta:
            logger.warning("[plugin_loader] plugin not found: %s", plugin_name)
            return False
        meta.enabled = True
        await self._run_setup(plugin_name)
        self._save_state()
        return True

    async def disable(self, plugin_name: str) -> bool:
        """Disable a plugin and run its teardown hook."""
        meta = self._plugins.get(plugin_name)
        if not meta:
            return False
        meta.enabled = False
        await self._run_teardown(plugin_name)
        self._save_state()
        return True

    async def reload(self, plugin_name: str) -> Optional[PluginMeta]:
        """Hot-reload a plugin: teardown → re-import → setup."""
        meta = self._plugins.get(plugin_name)
        if not meta:
            logger.warning("[plugin_loader] cannot reload unknown plugin: %s", plugin_name)
            return None
        await self._run_teardown(plugin_name)
        path = Path(meta.path)
        if path.exists():
            return await self._load_file(path, force_reload=True)
        return None

    def list_plugins(self) -> list[dict]:
        """Return info about all known plugins."""
        return [
            {
                "name": m.name,
                "version": m.version,
                "enabled": m.enabled,
                "ok": m.ok,
                "tools": [t.get("name") for t in m.tools],
                "specialties": m.agent_specialties,
                "error": m.error,
            }
            for m in self._plugins.values()
        ]

    async def list_marketplace(
        self, marketplace_url: Optional[str] = None
    ) -> list[MarketplaceEntry]:
        """Fetch and return available plugins from the marketplace registry."""
        url = marketplace_url or self._marketplace_url
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                entries = resp.json()
        except ImportError:
            logger.warning("[plugin_loader] httpx not installed — cannot fetch marketplace")
            return []
        except Exception as exc:
            logger.warning("[plugin_loader] marketplace fetch failed: %s", exc)
            return []

        result: list[MarketplaceEntry] = []
        for entry in entries if isinstance(entries, list) else []:
            try:
                result.append(MarketplaceEntry(
                    name=entry["name"],
                    version=entry.get("version", "0.0.1"),
                    description=entry.get("description", ""),
                    author=entry.get("author", "unknown"),
                    download_url=entry.get("download_url", ""),
                    deps=entry.get("deps", []),
                    tags=entry.get("tags", []),
                    stars=entry.get("stars", 0),
                ))
            except (KeyError, TypeError):
                continue
        return result

    async def install_from_marketplace(self, plugin_name: str) -> Optional[PluginMeta]:
        """
        Download and install a plugin by name from the marketplace.
        Returns PluginMeta on success, None on failure.
        """
        entries = await self.list_marketplace()
        entry = next((e for e in entries if e.name == plugin_name), None)
        if not entry:
            logger.warning("[plugin_loader] plugin not in marketplace: %s", plugin_name)
            return None
        if not entry.download_url:
            logger.warning("[plugin_loader] no download URL for: %s", plugin_name)
            return None

        # Download plugin file
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(entry.download_url)
                resp.raise_for_status()
                plugin_code = resp.text
        except Exception as exc:
            logger.error("[plugin_loader] download failed for %s: %s", plugin_name, exc)
            return None

        # Write to plugins/
        self._dir.mkdir(parents=True, exist_ok=True)
        dest = self._dir / f"{plugin_name.replace('-', '_')}.py"
        dest.write_text(plugin_code, encoding="utf-8")
        logger.info("[plugin_loader] installed %s → %s", plugin_name, dest)

        return await self._load_file(dest)

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._watcher:
            self._watcher.stop()

    # ── Internal loading ──────────────────────────────────────────────────────

    async def _load_file(self, path: Path, force_reload: bool = False) -> Optional[PluginMeta]:
        """Load a single plugin file. Returns PluginMeta or None on failure."""
        # Syntax check first
        try:
            import ast
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            logger.warning("[plugin_loader] syntax error in %s: %s", path.name, exc)
            meta = PluginMeta(
                name=path.stem, version="?", path=str(path), error=f"SyntaxError: {exc}"
            )
            self._plugins[path.stem] = meta
            return meta

        # Import module
        try:
            module_name = f"myconex_plugin_{path.stem}"
            if module_name in sys.modules and not force_reload:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("[plugin_loader] import error in %s: %s", path.name, exc)
            meta = PluginMeta(
                name=getattr(module if 'module' in dir() else None, "__plugin_name__", path.stem),
                version="?", path=str(path), error=str(exc),
            )
            self._plugins[path.stem] = meta
            return meta

        # Validate
        validation = validate_plugin_module(module, str(path))
        if not validation.valid:
            logger.warning("[plugin_loader] validation failed for %s: %s",
                           path.name, validation.errors)
            meta = PluginMeta(
                name=getattr(module, "__plugin_name__", path.stem),
                version=getattr(module, "__plugin_version__", "?"),
                path=str(path),
                error=f"Validation: {validation.errors}",
            )
            self._plugins[path.stem] = meta
            return meta

        if validation.warnings:
            for w in validation.warnings:
                logger.warning("[plugin_loader] %s: %s", path.name, w)

        # Dependency resolution
        deps = list(getattr(module, "__plugin_deps__", []))
        if deps:
            _, missing = await check_plugin_deps(deps)
            if missing:
                if self._auto_install_deps:
                    ok = await install_plugin_deps(missing)
                    if not ok:
                        logger.warning("[plugin_loader] dep install failed for %s", path.name)
                else:
                    logger.warning("[plugin_loader] missing deps for %s: %s (set auto_install_deps=True)",
                                   path.name, missing)

        # Build meta
        name = module.__plugin_name__
        self._modules[name] = module
        meta = PluginMeta(
            name=name,
            version=module.__plugin_version__,
            path=str(path),
            tools=list(getattr(module, "__plugin_tools__", [])),
            agent_specialties=list(getattr(module, "__plugin_agent_specialties__", [])),
            deps=deps,
            loaded_at=time.time(),
            file_hash=PluginFileWatcher.hash_file(path),
        )
        self._plugins[name] = meta

        # Run setup
        await self._run_setup(name)
        logger.info("[plugin_loader] loaded plugin: %s v%s (%d tools)",
                    name, meta.version, len(meta.tools))
        return meta

    async def _run_setup(self, plugin_name: str) -> None:
        module = self._modules.get(plugin_name)
        if module is None:
            return
        setup_fn = getattr(module, "plugin_setup", None)
        if setup_fn is None:
            return
        try:
            if asyncio.iscoroutinefunction(setup_fn):
                await setup_fn(self.registry)
            else:
                setup_fn(self.registry)
        except Exception as exc:
            logger.warning("[plugin_loader] setup error for %s: %s", plugin_name, exc)
            if plugin_name in self._plugins:
                self._plugins[plugin_name].error = f"setup error: {exc}"

    async def _run_teardown(self, plugin_name: str) -> None:
        module = self._modules.get(plugin_name)
        if module is None:
            return
        teardown_fn = getattr(module, "plugin_teardown", None)
        if teardown_fn is None:
            return
        try:
            if asyncio.iscoroutinefunction(teardown_fn):
                await teardown_fn()
            else:
                teardown_fn()
        except Exception as exc:
            logger.warning("[plugin_loader] teardown error for %s: %s", plugin_name, exc)

    def _start_watcher(self) -> None:
        if self._watcher_task and not self._watcher_task.done():
            return
        self._watcher = PluginFileWatcher(self._dir)
        self._watcher.on_change(self._on_file_change)
        self._watcher_task = asyncio.create_task(self._watcher.watch())

    def _on_file_change(self, path: str, event: str) -> None:
        if event in ("added", "modified"):
            logger.info("[plugin_loader] hot-reload triggered: %s (%s)", path, event)
            asyncio.create_task(self._load_file(Path(path), force_reload=True))
        elif event == "removed":
            stem = Path(path).stem
            name = next(
                (m.name for m in self._plugins.values() if Path(m.path).stem == stem), None
            )
            if name:
                logger.info("[plugin_loader] plugin removed: %s", name)
                asyncio.create_task(self._run_teardown(name))
                self._plugins.pop(name, None)
                self._modules.pop(name, None)

    def _ensure_init_py(self) -> None:
        init = self._dir / "__init__.py"
        if not init.exists():
            init.write_text("# MYCONEX plugins package\n")

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
            state = {
                "plugins": {
                    name: {
                        "version": m.version,
                        "enabled": m.enabled,
                        "path": m.path,
                        "error": m.error,
                    }
                    for name, m in self._plugins.items()
                },
                "saved_at": time.time(),
            }
            _PLUGIN_STATE.write_text(json.dumps(state, indent=2))
        except OSError as exc:
            logger.debug("[plugin_loader] state save failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_plugin_loader(
    plugins_dir: Optional[Path] = None,
    auto_install_deps: bool = False,
    hot_reload: bool = True,
) -> PluginLoader:
    """Create a PluginLoader with standard configuration."""
    return PluginLoader(
        plugins_dir=plugins_dir or _PLUGINS_DIR,
        auto_install_deps=auto_install_deps,
        hot_reload=hot_reload,
    )
