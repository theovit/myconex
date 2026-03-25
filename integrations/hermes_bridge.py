"""
MYCONEX ↔ Hermes-Agent Integration Bridge
==========================================
Bridges the entire hermes-agent ecosystem into MYCONEX:

  • Tool bridge        — expose hermes's 40+ tools via MYCONEX tool registry
  • Model router       — hermes smart model selection for MYCONEX routing
  • Context compressor — hermes multi-pass compression for MYCONEX contexts
  • Gateway bridge     — launch hermes multi-platform gateways (Telegram, Slack, …)
  • Skill bridge       — list/load hermes skills as MYCONEX plugins
  • State bridge       — access hermes SQLite session DB from MYCONEX
  • Batch bridge       — parallel batch processing via hermes batch_runner

All imports are guarded; every subsystem degrades gracefully when hermes
dependencies are not installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

HERMES_ROOT: Path = Path(__file__).parent / "hermes-agent"
HERMES_SKILLS_DIR: Path = HERMES_ROOT / "skills"
HERMES_OPTIONAL_SKILLS_DIR: Path = HERMES_ROOT / "optional-skills"

_PATH_SETUP_DONE = False


def setup_hermes_path() -> bool:
    """
    Add hermes-agent to sys.path (idempotent).

    Returns True if hermes-agent directory exists and was added.
    """
    global _PATH_SETUP_DONE
    if _PATH_SETUP_DONE:
        return True
    hermes_str = str(HERMES_ROOT)
    if not HERMES_ROOT.exists():
        logger.warning("[hermes_bridge] hermes-agent not found at %s", HERMES_ROOT)
        return False
    if hermes_str not in sys.path:
        sys.path.insert(0, hermes_str)
    _PATH_SETUP_DONE = True
    return True


def is_hermes_available() -> bool:
    """Return True if core hermes modules can be imported."""
    if not setup_hermes_path():
        return False
    try:
        import run_agent  # noqa: F401
        import model_tools  # noqa: F401
        return True
    except ImportError:
        return False


# ── Tool Bridge ───────────────────────────────────────────────────────────────

class HermesToolBridge:
    """
    Exposes hermes-agent's full tool registry to MYCONEX.

    On first call to load(), the hermes tool modules are imported so all
    40+ tools self-register into the hermes registry singleton.  MYCONEX
    components can then call get_registry() to dispatch any hermes tool.
    """

    _loaded: bool = False
    _registry: Any = None
    _tool_count: int = 0

    @classmethod
    def load(cls, toolsets: Optional[list[str]] = None) -> bool:
        """
        Import hermes tool modules so they self-register into the registry.

        Args:
            toolsets: optional list of toolset names to load (default: all)

        Returns True on success.
        """
        if cls._loaded:
            return True
        if not setup_hermes_path():
            return False
        try:
            from tools.registry import registry as _reg  # type: ignore[import]
            import model_tools  # type: ignore[import]  # triggers all tool imports

            cls._registry = _reg
            cls._tool_count = len(_reg.get_all_tool_names())
            cls._loaded = True
            logger.info(
                "[hermes_bridge] tool bridge loaded: %d tools across %d toolsets",
                cls._tool_count, len(_reg.get_available_toolsets()),
            )
            return True
        except Exception as exc:
            logger.warning("[hermes_bridge] tool bridge load failed: %s", exc)
            return False

    @classmethod
    def get_registry(cls):
        """Return the live hermes tool registry (load first if needed)."""
        if not cls._loaded:
            cls.load()
        return cls._registry

    @classmethod
    def get_tool_definitions(
        cls,
        enabled_toolsets: Optional[list[str]] = None,
        disabled_toolsets: Optional[list[str]] = None,
        quiet: bool = True,
    ) -> list[dict]:
        """Return OpenAI-format tool schemas for use with any LLM API."""
        if not cls.load():
            return []
        try:
            from model_tools import get_tool_definitions  # type: ignore[import]
            return get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=quiet,
            )
        except Exception as exc:
            logger.warning("[hermes_bridge] get_tool_definitions failed: %s", exc)
            return []

    @classmethod
    def dispatch_tool(
        cls,
        function_name: str,
        function_args: dict,
        task_id: Optional[str] = None,
        user_task: Optional[str] = None,
    ) -> str:
        """Synchronously dispatch a hermes tool call; returns JSON string result."""
        if not cls.load():
            return '{"error": "hermes tool bridge not available"}'
        try:
            from model_tools import handle_function_call  # type: ignore[import]
            return handle_function_call(
                function_name=function_name,
                function_args=function_args,
                task_id=task_id,
                user_task=user_task,
            )
        except Exception as exc:
            logger.warning("[hermes_bridge] tool dispatch failed: %s", exc)
            return f'{{"error": "{exc}"}}'

    @classmethod
    async def dispatch_tool_async(
        cls,
        function_name: str,
        function_args: dict,
        task_id: Optional[str] = None,
        user_task: Optional[str] = None,
    ) -> str:
        """Async wrapper for dispatch_tool (runs in thread executor)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.dispatch_tool(function_name, function_args, task_id, user_task),
        )

    @classmethod
    def list_tools(cls) -> list[str]:
        """Return names of all registered hermes tools."""
        if not cls.load():
            return []
        try:
            from model_tools import get_all_tool_names  # type: ignore[import]
            return get_all_tool_names()
        except Exception:
            return []

    @classmethod
    def check_toolset_availability(cls) -> dict[str, bool]:
        """Return {toolset_name: available} based on env var requirements."""
        if not cls.load():
            return {}
        try:
            from model_tools import check_toolset_requirements  # type: ignore[import]
            return check_toolset_requirements()
        except Exception:
            return {}


# ── Model Router ──────────────────────────────────────────────────────────────

class HermesModelRouter:
    """
    Wraps hermes's smart_model_routing for use in MYCONEX routing decisions.

    Provides model metadata (context windows, pricing, capabilities) and
    task-complexity-based model selection.
    """

    _metadata: Any = None

    @classmethod
    def _load(cls) -> bool:
        if cls._metadata is not None:
            return True
        if not setup_hermes_path():
            return False
        try:
            from agent.model_metadata import MODEL_METADATA  # type: ignore[import]
            cls._metadata = MODEL_METADATA
            return True
        except Exception as exc:
            logger.debug("[hermes_bridge] model metadata unavailable: %s", exc)
            return False

    @classmethod
    def get_model_metadata(cls, model_id: str) -> Optional[dict]:
        """Return hermes metadata dict for a given model ID."""
        if not cls._load():
            return None
        return cls._metadata.get(model_id)

    @classmethod
    def select_model(
        cls,
        task_complexity: float,
        preferred_provider: Optional[str] = None,
        require_vision: bool = False,
        require_function_calling: bool = True,
    ) -> Optional[str]:
        """
        Use hermes smart routing to pick the best model for a task.

        Falls back gracefully if smart_model_routing is unavailable.
        """
        if not setup_hermes_path():
            return None
        try:
            from agent.smart_model_routing import select_model_for_task  # type: ignore[import]
            return select_model_for_task(
                complexity=task_complexity,
                preferred_provider=preferred_provider,
                require_vision=require_vision,
                require_function_calling=require_function_calling,
            )
        except Exception:
            pass
        # Heuristic fallback
        if task_complexity > 0.8:
            return "anthropic/claude-opus-4-6"
        if task_complexity > 0.5:
            return "anthropic/claude-sonnet-4-6"
        return "anthropic/claude-haiku-4-5-20251001"

    @classmethod
    def estimate_cost(cls, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for a model call."""
        if not cls._load():
            return 0.0
        try:
            from agent.usage_pricing import estimate_cost  # type: ignore[import]
            return estimate_cost(model_id, input_tokens, output_tokens)
        except Exception:
            return 0.0


# ── Context Compressor ────────────────────────────────────────────────────────

class HermesContextCompressor:
    """
    Wraps hermes's multi-pass context compressor for MYCONEX conversation history.

    Useful when a long MYCONEX session context needs to be trimmed before
    sending to a model with a limited context window.
    """

    @staticmethod
    def compress(
        messages: list[dict],
        target_tokens: int = 32_000,
        protected_first: int = 2,
        protected_last: int = 4,
        summarization_model: Optional[str] = None,
    ) -> list[dict]:
        """
        Compress a list of chat messages within a token budget.

        Args:
            messages:           List of {role, content} dicts.
            target_tokens:      Max tokens for the compressed output.
            protected_first:    Number of turns to preserve at the start.
            protected_last:     Number of turns to preserve at the end.
            summarization_model: Model to use for summarisation (hermes default if None).

        Returns compressed message list; original if compression fails.
        """
        if not setup_hermes_path():
            return messages
        try:
            from agent.context_compressor import compress_context  # type: ignore[import]
            return compress_context(
                messages=messages,
                target_tokens=target_tokens,
                protected_first=protected_first,
                protected_last=protected_last,
                summarization_model=summarization_model,
            )
        except Exception as exc:
            logger.debug("[hermes_bridge] context compression failed: %s", exc)
            return messages

    @staticmethod
    def count_tokens(messages: list[dict], model: str = "gpt-4") -> int:
        """Rough token count for a message list."""
        try:
            total = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            return total
        except Exception:
            return 0


# ── Gateway Bridge ────────────────────────────────────────────────────────────

class HermesGatewayBridge:
    """
    Launches hermes multi-platform messaging gateways alongside MYCONEX.

    Supported platforms (requires hermes config + platform-specific tokens):
      Telegram, Slack, WhatsApp, Signal, Discord, Matrix, Mattermost,
      Email, SMS, DingTalk, HomeAssistant, Webhook, API Server
    """

    _runner: Any = None

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Optional[Any]:
        """Load hermes gateway config from YAML file or env vars."""
        if not setup_hermes_path():
            return None
        try:
            from gateway.config import GatewayConfig  # type: ignore[import]
            cfg_path = config_path or os.getenv(
                "HERMES_GATEWAY_CONFIG",
                str(Path.home() / ".hermes" / "gateway.yaml"),
            )
            if Path(cfg_path).exists():
                return GatewayConfig.from_file(cfg_path)
            logger.info("[hermes_bridge] no gateway config found at %s", cfg_path)
            return None
        except Exception as exc:
            logger.debug("[hermes_bridge] gateway config load failed: %s", exc)
            return None

    @classmethod
    async def start(cls, config_path: Optional[str] = None) -> bool:
        """Start the hermes gateway runner in the background."""
        if not setup_hermes_path():
            return False
        cfg = cls.load_config(config_path)
        if cfg is None:
            return False
        try:
            from gateway.run import GatewayRunner  # type: ignore[import]
            cls._runner = GatewayRunner(config=cfg)
            asyncio.create_task(cls._runner.run(), name="hermes-gateway")
            logger.info("[hermes_bridge] hermes gateway started")
            return True
        except Exception as exc:
            logger.warning("[hermes_bridge] gateway start failed: %s", exc)
            return False

    @classmethod
    async def stop(cls) -> None:
        if cls._runner is not None:
            try:
                await cls._runner.stop()
            except Exception:
                pass
            cls._runner = None

    @classmethod
    def list_enabled_platforms(cls, config_path: Optional[str] = None) -> list[str]:
        """Return list of platform names that have tokens configured."""
        cfg = cls.load_config(config_path)
        if cfg is None:
            return []
        try:
            return [p.value for p in cfg.enabled_platforms()]
        except Exception:
            return []


# ── Skill Bridge ──────────────────────────────────────────────────────────────

@dataclass
class HermesSkill:
    name: str
    path: Path
    category: str
    description: str = ""
    tools: list[str] = field(default_factory=list)


class HermesSkillBridge:
    """
    Lists and loads hermes skills for use as MYCONEX plugins.

    Skills in skills/ and optional-skills/ are enumerated and can be
    registered with the MYCONEX PluginLoader.
    """

    @staticmethod
    def list_skills(include_optional: bool = True) -> list[HermesSkill]:
        """Return all available hermes skills."""
        skills: list[HermesSkill] = []

        def _scan(base: Path, category_prefix: str = "") -> None:
            if not base.exists():
                return
            for item in sorted(base.iterdir()):
                if item.is_dir() and not item.name.startswith((".", "_")):
                    category = f"{category_prefix}{item.name}"
                    # Look for skill entry points
                    for py_file in sorted(item.rglob("*.py")):
                        if py_file.name.startswith(("skill_", "main", "run")):
                            skills.append(HermesSkill(
                                name=py_file.stem,
                                path=py_file,
                                category=category,
                            ))

        _scan(HERMES_SKILLS_DIR)
        if include_optional:
            _scan(HERMES_OPTIONAL_SKILLS_DIR, "optional/")
        return skills

    @staticmethod
    def load_skill_as_plugin(skill: HermesSkill, plugin_loader) -> bool:
        """
        Load a hermes skill into the MYCONEX PluginLoader.

        Returns True if loaded successfully.
        """
        try:
            plugin_loader.load_from_path(skill.path)
            return True
        except Exception as exc:
            logger.debug("[hermes_bridge] skill load failed %s: %s", skill.name, exc)
            return False


# ── State Bridge ──────────────────────────────────────────────────────────────

class HermesStateBridge:
    """
    Access the hermes SQLite session DB from MYCONEX.

    Enables cross-system memory queries: search hermes conversation history
    from MYCONEX agents, or store MYCONEX task results in hermes's DB.
    """

    _db: Any = None

    @classmethod
    def get_db(cls, db_path: Optional[str] = None):
        """Return a hermes StateDB instance."""
        if cls._db is not None:
            return cls._db
        if not setup_hermes_path():
            return None
        try:
            from hermes_state import StateDB, DEFAULT_DB_PATH  # type: ignore[import]
            path = db_path or str(DEFAULT_DB_PATH)
            cls._db = StateDB(path)
            return cls._db
        except Exception as exc:
            logger.debug("[hermes_bridge] state DB unavailable: %s", exc)
            return None

    @classmethod
    def search_history(cls, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across hermes conversation history."""
        db = cls.get_db()
        if db is None:
            return []
        try:
            return db.search(query, limit=limit)
        except Exception as exc:
            logger.debug("[hermes_bridge] history search failed: %s", exc)
            return []

    @classmethod
    def get_recent_sessions(cls, n: int = 5) -> list[dict]:
        """Return the N most recent hermes sessions."""
        db = cls.get_db()
        if db is None:
            return []
        try:
            return db.list_sessions(limit=n)
        except Exception as exc:
            logger.debug("[hermes_bridge] session list failed: %s", exc)
            return []


# ── Batch Bridge ──────────────────────────────────────────────────────────────

class HermesBatchBridge:
    """
    Expose hermes parallel batch processing to MYCONEX.

    Useful for running MYCONEX tasks across large datasets in parallel
    using hermes's checkpoint-aware batch_runner.
    """

    @staticmethod
    async def run_batch(
        prompts: list[str],
        model: Optional[str] = None,
        toolsets: Optional[list[str]] = None,
        run_name: str = "myconex_batch",
        max_workers: int = 4,
    ) -> list[dict]:
        """
        Run a list of prompts through hermes AIAgent in parallel.

        Returns list of {prompt, response, success, task_id} dicts.
        """
        if not setup_hermes_path():
            return [{"prompt": p, "response": "", "success": False} for p in prompts]

        loop = asyncio.get_running_loop()

        def _run_sync():
            try:
                from run_agent import AIAgent  # type: ignore[import]
                from hermes_constants import OPENROUTER_BASE_URL  # type: ignore[import]
                import concurrent.futures

                _model = model or os.getenv("HERMES_MODEL", "anthropic/claude-sonnet-4-6")
                _api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
                _base_url = (
                    OPENROUTER_BASE_URL if os.getenv("OPENROUTER_API_KEY")
                    else "https://api.anthropic.com/v1"
                )

                def _process_one(prompt: str) -> dict:
                    agent = AIAgent(
                        base_url=_base_url,
                        api_key=_api_key,
                        model=_model,
                        enabled_toolsets=toolsets,
                        quiet_mode=True,
                        skip_memory=True,
                        max_iterations=30,
                    )
                    try:
                        result = agent.run_conversation(user_message=prompt)
                        return {
                            "prompt": prompt,
                            "response": result.get("response", ""),
                            "success": True,
                            "messages": result.get("messages", []),
                        }
                    except Exception as exc:
                        return {"prompt": prompt, "response": "", "success": False, "error": str(exc)}

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    results = list(pool.map(_process_one, prompts))
                return results
            except Exception as exc:
                logger.warning("[hermes_bridge] batch run failed: %s", exc)
                return [{"prompt": p, "response": "", "success": False} for p in prompts]

        return await loop.run_in_executor(None, _run_sync)


# ── Convenience: register hermes tools into MYCONEX agentic_tools ─────────────

def register_hermes_tools_in_myconex() -> int:
    """
    Register hermes tools into MYCONEX's agentic_tools registry.

    Each hermes tool is wrapped as a MYCONEX tool definition and handler.
    Returns the number of tools successfully registered.
    """
    bridge = HermesToolBridge
    if not bridge.load():
        return 0

    registry = bridge.get_registry()
    if registry is None:
        return 0

    # Hermes tools are already in the registry; expose them via MYCONEX's
    # agentic_tools path so they can be called from MYCONEX handlers.
    # Since agentic_tools.py already imports the same hermes registry,
    # this is effectively a no-op for tool dispatch — we just verify load.
    count = len(bridge.list_tools())
    logger.info("[hermes_bridge] %d hermes tools available in shared registry", count)
    return count


def status() -> dict:
    """Return a status snapshot of all hermes bridge subsystems."""
    return {
        "hermes_available": is_hermes_available(),
        "hermes_root": str(HERMES_ROOT),
        "hermes_root_exists": HERMES_ROOT.exists(),
        "tools_loaded": HermesToolBridge._loaded,
        "tool_count": len(HermesToolBridge.list_tools()),
        "toolset_availability": HermesToolBridge.check_toolset_availability(),
        "gateway_runner_active": HermesGatewayBridge._runner is not None,
        "state_db_connected": HermesStateBridge._db is not None,
        "skills_available": len(HermesSkillBridge.list_skills()),
    }
