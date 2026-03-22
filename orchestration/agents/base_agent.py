"""
MYCONEX Base Agent
Abstract base class for all Jarvis-style AI agents in the mesh.
Agents subscribe to task subjects, execute via Ollama/LiteLLM, and return results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


# ─── Agent Config ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    name: str
    agent_type: str                          # "inference", "search", "embedding", etc.
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    litellm_url: str = "http://localhost:4000"
    system_prompt: str = "You are a helpful AI assistant in the MYCONEX mesh."
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 60.0
    max_concurrent: int = 4
    use_litellm: bool = False                # route through LiteLLM proxy if True


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
            # Always keep the system prompt if present
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


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all MYCONEX agents.

    Subclass and implement:
        handle_task(task_id, task_type, payload, context) -> AgentResult
        can_handle(task_type) -> bool

    Optional overrides:
        on_start(), on_stop(), on_error()
    """

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
        logger.info(f"[agent:{self.name}] started (type={self.agent_type}, model={self.config.model})")

    async def stop(self) -> None:
        self._state = AgentState.STOPPED
        for task in self._active_tasks.values():
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        if self._http:
            await self._http.aclose()
        await self.on_stop()
        logger.info(f"[agent:{self.name}] stopped. tasks={self._total_tasks} errors={self._total_errors}")

    async def on_start(self) -> None:
        """Override to perform setup after start."""

    async def on_stop(self) -> None:
        """Override to perform cleanup before stop."""

    async def on_error(self, task_id: str, error: Exception) -> None:
        """Override to handle task errors."""
        logger.error(f"[agent:{self.name}] error on task {task_id}: {error}")

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
        """Send a chat completion request to Ollama or LiteLLM."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        if self.config.use_litellm:
            return await self._chat_litellm(messages, model, temperature, max_tokens)
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
        url = f"{self.config.ollama_url}/api/embeddings"
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
            "state": self._state.value,
            "model": self.config.model,
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

        # Add system prompt only once per session (first turn)
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
