"""
HermesAgent — MYCONEX BaseAgent backed by hermes-agent AIAgent
===============================================================
Wraps the full hermes-agent AIAgent as a MYCONEX BaseAgent, giving the
routing layer access to:
  - hermes's 40+ tools (web, terminal, browser, vision, code execution, …)
  - hermes's multi-model LLM support (OpenRouter, Anthropic, Nous Portal)
  - hermes's context compression, session state, and memory (Honcho)
  - hermes's streaming and callback infrastructure

The agent is registered with TaskRouter under the name "hermes-primary".
It handles task types: chat, code, search, research, web, analysis, task, tool.

Usage:
    agent = create_hermes_agent(config, hermes_config)
    router.register_agent(agent)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional

from orchestration.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentRole,
    BaseAgent,
)

logger = logging.getLogger(__name__)

# Task types that HermesAgent can handle
_HANDLED_TYPES = frozenset({
    "chat", "code", "search", "research", "web", "analysis",
    "task", "tool", "vision", "voice", "batch", "delegate",
})


# ─── HermesAgentConfig ────────────────────────────────────────────────────────

class HermesAgentConfig:
    """
    Hermes-specific runtime configuration layered on top of AgentConfig.

    Values are resolved from: explicit kwargs → env vars → defaults.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled_toolsets: Optional[list[str]] = None,
        disabled_toolsets: Optional[list[str]] = None,
        max_iterations: int = 90,
        save_trajectories: bool = False,
        skip_memory: bool = False,
        ephemeral_system_prompt: Optional[str] = None,
        platform: str = "myconex",
    ) -> None:
        self.model = (
            model
            or os.getenv("HERMES_MODEL")
            or os.getenv("OPENROUTER_DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
        )
        self.base_url = (
            base_url
            or os.getenv("HERMES_BASE_URL")
            or (
                "https://openrouter.ai/api/v1"
                if os.getenv("OPENROUTER_API_KEY")
                else "https://api.anthropic.com/v1"
            )
        )
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY", "")
        )
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        self.max_iterations = max_iterations
        self.save_trajectories = save_trajectories
        self.skip_memory = skip_memory
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform


# ─── HermesAgent ──────────────────────────────────────────────────────────────

class HermesAgent(BaseAgent):
    """
    MYCONEX BaseAgent that delegates execution to hermes-agent's AIAgent.

    Each task session gets a persistent AIAgent instance (keyed by
    AgentContext.session_id) so conversation history is maintained within
    a session.  Instances are evicted after idle_ttl_s seconds.
    """

    def __init__(
        self,
        config: AgentConfig,
        hermes_config: Optional[HermesAgentConfig] = None,
        idle_ttl_s: float = 1800.0,
    ) -> None:
        super().__init__(config)
        self.hermes_config = hermes_config or HermesAgentConfig()
        self._idle_ttl_s = idle_ttl_s
        self._sessions: dict[str, dict] = {}   # session_id → {agent, last_used}
        self._bridge_loaded: bool = False
        self._AIAgent: Any = None              # lazy-loaded class ref

    # ── Interface ─────────────────────────────────────────────────────────────

    def can_handle(self, task_type: str) -> bool:
        return task_type in _HANDLED_TYPES

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        start = time.monotonic()
        session_id = context.session_id if context else "default"
        prompt = payload.get("prompt") or payload.get("message") or payload.get("content", "")

        if not prompt:
            return AgentResult(
                task_id=task_id,
                agent_name=self.config.name,
                success=False,
                error="No prompt provided in payload",
            )

        if not self._ensure_bridge():
            return AgentResult(
                task_id=task_id,
                agent_name=self.config.name,
                success=False,
                error="hermes-agent not available — check HERMES_ROOT and dependencies",
            )

        hermes_agent = self._get_or_create_session(session_id)

        # Build conversation history from MYCONEX context
        history = self._build_history(context)

        # Run synchronous hermes conversation in thread executor
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: hermes_agent.run_conversation(
                    user_message=prompt,
                    conversation_history=history,
                    task_id=task_id,
                ),
            )
        except Exception as exc:
            logger.exception("[HermesAgent] run_conversation failed: %s", exc)
            duration = (time.monotonic() - start) * 1000
            return AgentResult(
                task_id=task_id,
                agent_name=self.config.name,
                success=False,
                error=str(exc),
                duration_ms=duration,
            )

        # Update last-used timestamp
        if session_id in self._sessions:
            self._sessions[session_id]["last_used"] = time.monotonic()

        duration = (time.monotonic() - start) * 1000
        response = result.get("response", "") if isinstance(result, dict) else str(result)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_calls = [m for m in messages if m.get("role") == "tool"]

        return AgentResult(
            task_id=task_id,
            agent_name=self.config.name,
            success=True,
            output={"response": response, "messages": messages, "task_type": task_type},
            duration_ms=duration,
            model_used=self.hermes_config.model,
            metadata={
                "session_id": session_id,
                "tool_calls": len(tool_calls),
                "hermes": True,
            },
        )

    # ── Session management ────────────────────────────────────────────────────

    def _ensure_bridge(self) -> bool:
        if self._bridge_loaded:
            return self._AIAgent is not None
        self._bridge_loaded = True
        try:
            from integrations.hermes_bridge import setup_hermes_path  # type: ignore[import]
            if not setup_hermes_path():
                return False
            from run_agent import AIAgent  # type: ignore[import]
            self._AIAgent = AIAgent
            logger.info("[HermesAgent] hermes AIAgent loaded")
            return True
        except Exception as exc:
            logger.warning("[HermesAgent] hermes load failed: %s", exc)
            return False

    def _get_or_create_session(self, session_id: str) -> Any:
        """Return a hermes AIAgent for the session, evicting stale ones first."""
        now = time.monotonic()
        # Evict stale sessions
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s["last_used"] > self._idle_ttl_s
        ]
        for sid in stale:
            del self._sessions[sid]
            logger.debug("[HermesAgent] evicted stale session %s", sid)

        if session_id not in self._sessions:
            hc = self.hermes_config
            agent = self._AIAgent(
                base_url=hc.base_url,
                api_key=hc.api_key,
                model=hc.model,
                max_iterations=hc.max_iterations,
                enabled_toolsets=hc.enabled_toolsets,
                disabled_toolsets=hc.disabled_toolsets,
                save_trajectories=hc.save_trajectories,
                skip_memory=hc.skip_memory,
                quiet_mode=True,
                session_id=session_id,
                platform=hc.platform,
                ephemeral_system_prompt=hc.ephemeral_system_prompt,
            )
            self._sessions[session_id] = {"agent": agent, "last_used": now}
            logger.debug("[HermesAgent] created new session %s", session_id)

        return self._sessions[session_id]["agent"]

    @staticmethod
    def _build_history(context: Optional[AgentContext]) -> list[dict]:
        """Convert MYCONEX AgentContext history to hermes message format."""
        if context is None:
            return []
        messages = []
        for turn in context.history:
            if hasattr(turn, "role") and hasattr(turn, "content"):
                messages.append({"role": turn.role, "content": turn.content})
        return messages

    # ── Status & introspection ─────────────────────────────────────────────────

    def status(self) -> dict:
        base = super().status() if hasattr(super(), "status") else {}
        base.update({
            "hermes_model": self.hermes_config.model,
            "hermes_base_url": self.hermes_config.base_url,
            "active_sessions": len(self._sessions),
            "handled_types": sorted(_HANDLED_TYPES),
            "bridge_loaded": self._bridge_loaded,
            "bridge_ok": self._AIAgent is not None,
        })
        return base

    def evict_session(self, session_id: str) -> bool:
        """Manually evict a session (free memory)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# ── Factory ───────────────────────────────────────────────────────────────────

def create_hermes_agent(
    hermes_config: Optional[HermesAgentConfig] = None,
    name: str = "hermes-primary",
    idle_ttl_s: float = 1800.0,
) -> HermesAgent:
    """
    Create and return a HermesAgent ready for TaskRouter registration.

    Example:
        agent = create_hermes_agent()
        router.register_agent(agent)
    """
    config = AgentConfig(
        name=name,
        agent_type="hermes",
        model=hermes_config.model if hermes_config else "hermes",
        backend="hermes",
        role=AgentRole.SPECIALIST,
    )
    return HermesAgent(config=config, hermes_config=hermes_config, idle_ttl_s=idle_ttl_s)
