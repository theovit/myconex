"""
MYCONEX Base Agent
Abstract base class for all Jarvis-style AI agents in the mesh.
Agents subscribe to task subjects, execute via Ollama/LiteLLM, and return results.

Phase 1 additions:
  - AgentRole enum for agent-as-employee hierarchy
  - AgentConfig: multi-backend support (ollama, llamacpp, lmstudio, litellm)
  - _estimate_complexity(): lightweight complexity scorer for delegation decisions
  - BaseAgent.set_router() / .delegate(): hierarchical sub-agent delegation
  - BaseAgent._chat_openai_compat(): shared path for llama.cpp + LM Studio
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import httpx

if TYPE_CHECKING:
    # Avoid circular import; TaskRouter is injected at runtime via set_router()
    from orchestration.workflows.task_router import TaskRouter

logger = logging.getLogger(__name__)


# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


# ─── Agent Role (agent-as-employee hierarchy) ─────────────────────────────────

class AgentRole(str, Enum):
    MANAGER = "manager"        # Decomposes complex tasks; delegates to workers
    WORKER = "worker"          # Handles well-scoped tasks end-to-end
    SPECIALIST = "specialist"  # Narrow-domain expert (code, search, embed, …)


# ─── Agent Config ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    name: str
    agent_type: str                          # "inference", "search", "embedding", etc.
    model: str = "llama3.2:3b"
    # ── Backend URLs ────────────────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434"
    litellm_url: str = "http://localhost:4000"
    llamacpp_url: str = "http://localhost:8080"   # llama.cpp server (OpenAI-compat)
    lmstudio_url: str = "http://localhost:1234"   # LM Studio (OpenAI-compat)
    # ── Backend selection ───────────────────────────────────────────────────
    # "ollama" | "llamacpp" | "lmstudio" | "litellm"
    # use_litellm=True is a legacy alias for backend="litellm"
    backend: str = "ollama"
    use_litellm: bool = False                # legacy; prefer backend="litellm"
    # ── Inference params ────────────────────────────────────────────────────
    system_prompt: str = "You are a helpful AI assistant in the MYCONEX mesh."
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 60.0
    max_concurrent: int = 4
    # ── Agent-as-employee ───────────────────────────────────────────────────
    role: AgentRole = AgentRole.WORKER
    manager: Optional[str] = None    # name of the supervising manager agent


# ─── Conversation Turn ────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str      # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentContext:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    history: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, role: str, content: str) -> None:
        self.history.append(Turn(role=role, content=content))

    def to_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.history]

    def trim(self, max_turns: int = 20) -> None:
        if len(self.history) > max_turns:
            system = [t for t in self.history if t.role == "system"]
            rest = [t for t in self.history if t.role != "system"]
            keep = max(0, max_turns - len(system))
            self.history = system + (rest[-keep:] if keep else [])


# ─── Agent Result ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0.0
    model_used: str = ""
    metadata: dict = field(default_factory=dict)


# ─── Complexity Estimation ────────────────────────────────────────────────────

def _estimate_complexity(messages: list[dict]) -> float:
    """
    Lightweight complexity score (0.0–1.0) used to drive delegation decisions.

    Intentionally dependency-free so BaseAgent doesn't import moe_hermes_integration.
    The HermesMoEAgent uses the fuller _score_complexity() from that module instead.

    Signals:
      - Length of user turns           (0 → 0.30)
      - Code blocks / inline code      (+0.20)
      - URL presence                   (+0.10)
      - Multi-turn depth               (0 → 0.15)
      - Delegation payload marker      (+0.05, already-delegated tasks)
    """
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    if not user_text:
        return 0.5

    score = 0.0
    score += min(0.30, len(user_text.split()) / 200.0)

    if "```" in user_text or re.search(r"`[^`\n]+`", user_text):
        score += 0.20

    if re.search(r"https?://", user_text):
        score += 0.10

    user_turns = sum(1 for m in messages if m.get("role") == "user")
    score += min(0.15, user_turns * 0.04)

    return min(1.0, score)


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all MYCONEX agents.

    Subclass and implement:
        handle_task(task_id, task_type, payload, context) -> AgentResult
        can_handle(task_type) -> bool

    Optional overrides:
        on_start(), on_stop(), on_error()

    Agent-as-employee delegation:
        Call set_router(router) after construction to enable delegate().
        Manager agents decompose tasks and call delegate() with sub-task types;
        the TaskRouter picks the best available worker/specialist.
    """

    MAX_DELEGATION_DEPTH: int = 4

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type

        self._state = AgentState.IDLE
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._http: Optional[httpx.AsyncClient] = None
        self._total_tasks = 0
        self._total_errors = 0
        self._started_at: Optional[float] = None

        # Injected by TaskRouter.start() / .register_agent()
        self._router: Optional[Any] = None

    # ─── Abstract Methods ─────────────────────────────────────────────────────

    @abstractmethod
    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Process a task and return a result."""
        ...

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Return True if this agent handles the given task type."""
        ...

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=self.config.timeout)
        self._state = AgentState.IDLE
        self._started_at = time.time()
        await self.on_start()
        logger.info(
            "[agent:%s] started (type=%s, model=%s, backend=%s, role=%s)",
            self.name, self.agent_type, self.config.model,
            self.config.backend, self.config.role.value,
        )

    async def stop(self) -> None:
        self._state = AgentState.STOPPED
        for task in self._active_tasks.values():
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        if self._http:
            await self._http.aclose()
        await self.on_stop()
        logger.info(
            "[agent:%s] stopped. tasks=%d errors=%d",
            self.name, self._total_tasks, self._total_errors,
        )

    async def on_start(self) -> None:
        """Override to perform setup after start."""

    async def on_stop(self) -> None:
        """Override to perform cleanup before stop."""

    async def on_error(self, task_id: str, error: Exception) -> None:
        """Override to handle task errors."""
        logger.error("[agent:%s] error on task %s: %s", self.name, task_id, error)

    # ─── Router Injection ─────────────────────────────────────────────────────

    def set_router(self, router: Any) -> None:
        """
        Inject the TaskRouter so this agent can delegate sub-tasks.
        Called automatically by TaskRouter.start() and TaskRouter.register_agent().
        """
        self._router = router
        logger.debug("[agent:%s] router injected", self.name)

    # ─── Agent-as-Employee Delegation ─────────────────────────────────────────

    async def delegate(
        self,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
        *,
        depth: int = 0,
        complexity: Optional[float] = None,
    ) -> AgentResult:
        """
        Delegate a sub-task to a specialised agent via the TaskRouter.

        Implements the agent-as-employee pattern: a Manager agent decomposes a
        complex request and hands off focused sub-tasks to Worker/Specialist
        agents.  The TaskRouter selects the best available handler.

        Args:
            task_type:   The sub-task type to route (e.g. "code", "search").
            payload:     Task payload forwarded to the worker agent.
            context:     Shared conversation context; fresh one created if None.
            depth:       Current delegation depth (internal; guards against loops).
            complexity:  Pre-computed complexity score; auto-computed if None.

        Returns:
            AgentResult from the worker, with delegation metadata in .metadata.
        """
        task_id = str(uuid.uuid4())[:8]

        if depth >= self.MAX_DELEGATION_DEPTH:
            logger.warning(
                "[agent:%s] delegation depth limit (%d) reached for task_type=%s",
                self.name, self.MAX_DELEGATION_DEPTH, task_type,
            )
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error=f"Max delegation depth ({self.MAX_DELEGATION_DEPTH}) reached",
                metadata={"delegation_depth": depth, "delegated_by": self.name},
            )

        if self._router is None:
            logger.error("[agent:%s] cannot delegate — no router injected", self.name)
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error="Agent has no router — ensure TaskRouter.register_agent() was called",
            )

        # Auto-compute complexity from context if not provided
        if complexity is None and context is not None:
            complexity = _estimate_complexity(context.to_messages())

        # Embed delegation metadata so the worker can inspect its call chain
        enriched_payload: dict = {
            **payload,
            "_delegation": {
                "depth": depth + 1,
                "delegated_by": self.name,
                "delegated_by_role": self.config.role.value,
                "complexity": round(complexity, 3) if complexity is not None else None,
            },
        }

        logger.info(
            "[agent:%s][role:%s] → delegate %s (depth=%d, complexity=%s)",
            self.name, self.config.role.value, task_type, depth,
            f"{complexity:.2f}" if complexity is not None else "n/a",
        )

        result = await self._router.route(
            task_type=task_type,
            payload=enriched_payload,
            context=context,
            force_local=True,
        )
        result.metadata.setdefault("delegation_depth", depth + 1)
        result.metadata.setdefault("delegated_by", self.name)
        return result

    async def delegate_parallel(
        self,
        subtasks: list[tuple[str, dict]],
        context: Optional[AgentContext] = None,
        depth: int = 0,
    ) -> list[AgentResult]:
        """
        Delegate multiple independent sub-tasks in parallel.

        Args:
            subtasks: List of (task_type, payload) pairs.
            context:  Shared context forwarded to all workers.
            depth:    Current delegation depth.

        Returns:
            List of AgentResults in the same order as subtasks.
        """
        coros = [
            self.delegate(task_type, payload, context=context, depth=depth)
            for task_type, payload in subtasks
        ]
        return list(await asyncio.gather(*coros, return_exceptions=False))

    # ─── Task Dispatch ────────────────────────────────────────────────────────

    async def dispatch(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Dispatch a task with concurrency limiting and timing."""
        if not self.can_handle(task_type):
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error=f"Agent {self.name} cannot handle task type '{task_type}'",
            )

        async with self._semaphore:
            self._state = AgentState.RUNNING
            start = time.time()
            self._total_tasks += 1

            try:
                result = await asyncio.wait_for(
                    self.handle_task(task_id, task_type, payload, context),
                    timeout=self.config.timeout,
                )
                result.duration_ms = (time.time() - start) * 1000
                return result
            except asyncio.TimeoutError:
                self._total_errors += 1
                err = f"Task {task_id} timed out after {self.config.timeout}s"
                await self.on_error(task_id, TimeoutError(err))
                return AgentResult(task_id=task_id, agent_name=self.name, success=False, error=err)
            except Exception as e:
                self._total_errors += 1
                await self.on_error(task_id, e)
                return AgentResult(task_id=task_id, agent_name=self.name, success=False, error=str(e))
            finally:
                if not self._semaphore._value == self.config.max_concurrent:
                    self._state = AgentState.IDLE

    # ─── LLM Helpers ─────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to the configured backend.

        Backend resolution order:
          1. use_litellm=True (legacy) → litellm
          2. config.backend             → ollama | llamacpp | lmstudio | litellm
        """
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        # Resolve effective backend (legacy flag wins)
        backend = "litellm" if self.config.use_litellm else self.config.backend

        if backend == "litellm":
            return await self._chat_litellm(messages, model, temperature, max_tokens)
        if backend == "llamacpp":
            return await self._chat_openai_compat(
                self.config.llamacpp_url, messages, model, temperature, max_tokens
            )
        if backend == "lmstudio":
            return await self._chat_openai_compat(
                self.config.lmstudio_url, messages, model, temperature, max_tokens
            )
        # default: ollama
        return await self._chat_ollama(messages, model, temperature, max_tokens)

    async def _chat_ollama(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int
    ) -> str:
        url = f"{self.config.ollama_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def _chat_litellm(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int
    ) -> str:
        url = f"{self.config.litellm_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _chat_openai_compat(
        self,
        base_url: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str = "local",
    ) -> str:
        """
        POST to any OpenAI-compatible /v1/chat/completions endpoint.

        Used for llama.cpp server (default :8080) and LM Studio (:1234).
        Both expose the OpenAI chat completions API with no auth by default.
        """
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat tokens from Ollama."""
        model = model or self.config.model
        url = f"{self.config.ollama_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        async with self._http.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue

    async def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """Generate embeddings via Ollama."""
        model = model or "nomic-embed-text"
        url = f"{self.config.ollama_url}/api/embed"
        payload = {"model": model, "prompt": text}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    # ─── Status ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    def status(self) -> dict:
        return {
            "name": self.name,
            "type": self.agent_type,
            "role": self.config.role.value,
            "state": self._state.value,
            "model": self.config.model,
            "backend": self.config.backend,
            "router_attached": self._router is not None,
            "total_tasks": self._total_tasks,
            "total_errors": self._total_errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
        }


# ─── InferenceAgent (concrete example) ───────────────────────────────────────

class InferenceAgent(BaseAgent):
    """
    General-purpose inference agent.
    Handles: "inference", "chat", "ask", "generate"
    """

    HANDLED_TYPES = {"inference", "chat", "ask", "generate", "completion"}

    def can_handle(self, task_type: str) -> bool:
        return task_type.lower() in self.HANDLED_TYPES

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        prompt = payload.get("prompt") or payload.get("message") or payload.get("query", "")
        if not prompt:
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error="No prompt in payload",
            )

        if context is None:
            context = AgentContext()

        if not any(t.role == "system" for t in context.history):
            context.add("system", self.config.system_prompt)
        context.add("user", prompt)
        context.trim()

        response = await self.chat(context.to_messages())
        context.add("assistant", response)

        return AgentResult(
            task_id=task_id,
            agent_name=self.name,
            success=True,
            output={"response": response, "session_id": context.session_id},
            model_used=self.config.model,
        )


# ─── EmbeddingAgent ───────────────────────────────────────────────────────────

class EmbeddingAgent(BaseAgent):
    """Generates text embeddings."""

    HANDLED_TYPES = {"embedding", "embed", "vectorize"}

    def can_handle(self, task_type: str) -> bool:
        return task_type.lower() in self.HANDLED_TYPES

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        text = payload.get("text") or payload.get("content", "")
        if not text:
            return AgentResult(
                task_id=task_id, agent_name=self.name, success=False, error="No text in payload"
            )

        model = payload.get("model", "nomic-embed-text")
        vector = await self.embed(text, model=model)

        return AgentResult(
            task_id=task_id,
            agent_name=self.name,
            success=True,
            output={"embedding": vector, "dimensions": len(vector)},
            model_used=model,
        )


# ─── Agent Factory ────────────────────────────────────────────────────────────

def create_agent(agent_type: str, name: str, config_overrides: dict = {}) -> BaseAgent:
    base_config = AgentConfig(name=name, agent_type=agent_type, **config_overrides)

    registry = {
        "inference": InferenceAgent,
        "chat": InferenceAgent,
        "embedding": EmbeddingAgent,
    }

    cls = registry.get(agent_type, InferenceAgent)
    return cls(base_config)
