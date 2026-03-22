"""
MYCONEX Task Router
Routes incoming tasks to the appropriate agent or mesh node
based on task type, tier requirements, and load balancing.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from orchestration.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
    BaseAgent,
    EmbeddingAgent,
    InferenceAgent,
)

logger = logging.getLogger(__name__)


# ─── Route Rule ───────────────────────────────────────────────────────────────

@dataclass
class RouteRule:
    """Maps task types to a preferred tier and optional role."""
    task_type: str
    preferred_tier: Optional[str] = None   # T1/T2/T3/T4 or None = any
    required_role: Optional[str] = None
    priority_boost: int = 0
    description: str = ""


# Default routing table
DEFAULT_ROUTES: list[RouteRule] = [
    RouteRule("inference",   preferred_tier="T1", required_role="large-model",   description="Large model inference"),
    RouteRule("chat",        preferred_tier="T2", required_role="inference",     description="Interactive chat"),
    RouteRule("ask",         preferred_tier="T2", description="Single-shot Q&A"),
    RouteRule("generate",    preferred_tier="T1", description="Long-form generation"),
    RouteRule("embedding",   preferred_tier="T2", required_role="embedding",     description="Vector embedding"),
    RouteRule("embed",       preferred_tier="T2", required_role="embedding"),
    RouteRule("search",      preferred_tier="T3", description="Semantic search"),
    RouteRule("classify",    preferred_tier="T3", description="Classification"),
    RouteRule("summarize",   preferred_tier="T2", description="Summarization"),
    RouteRule("translate",   preferred_tier="T2", description="Translation"),
    RouteRule("code",        preferred_tier="T1", description="Code generation"),
    RouteRule("train",       preferred_tier="T1", required_role="training",      description="Model fine-tuning"),
    RouteRule("sensor",      preferred_tier="T4", description="Edge sensor data"),
    RouteRule("relay",       preferred_tier="T4", description="Message relay"),
]


# ─── Agent Registry ───────────────────────────────────────────────────────────

class AgentRegistry:
    """Maintains the pool of local agents."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        logger.info(f"[registry] agent registered: {agent.name} (type={agent.agent_type})")

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def find(self, task_type: str) -> list[BaseAgent]:
        """Return all agents that can handle the given task type."""
        return [a for a in self._agents.values() if a.can_handle(task_type)]

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    @property
    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def status(self) -> list[dict]:
        return [a.status() for a in self._agents.values()]


# ─── Router ───────────────────────────────────────────────────────────────────

class TaskRouter:
    """
    Routes tasks to local agents or remote mesh nodes.

    Local routing: matches task_type → capable agent in the local registry.
    Remote routing: defers to the MeshOrchestrator for cross-node tasks.

    Usage:
        router = TaskRouter(node_tier="T2")
        await router.start()
        result = await router.route("chat", {"prompt": "hello"})
        await router.stop()
    """

    def __init__(
        self,
        node_tier: str = "T3",
        ollama_url: str = "http://localhost:11434",
        litellm_url: str = "http://localhost:4000",
        routing_table: Optional[list[RouteRule]] = None,
    ):
        self.node_tier = node_tier
        self.ollama_url = ollama_url
        self.litellm_url = litellm_url
        self.routing_table: dict[str, RouteRule] = {
            r.task_type: r for r in (routing_table or DEFAULT_ROUTES)
        }

        self.registry = AgentRegistry()
        self._remote_handler: Optional[Any] = None   # injected MeshOrchestrator
        self._running = False

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._provision_default_agents()

        for agent in self.registry.all:
            await agent.start()

        self._running = True
        logger.info(
            f"[router] started on tier={self.node_tier} "
            f"agents={len(self.registry.all)}"
        )

    async def stop(self) -> None:
        self._running = False
        for agent in self.registry.all:
            await agent.stop()
        logger.info("[router] stopped.")

    def set_remote_handler(self, orchestrator: Any) -> None:
        """Inject an orchestrator for remote task submission."""
        self._remote_handler = orchestrator

    # ─── Agent Provisioning ───────────────────────────────────────────────────

    def _provision_default_agents(self) -> None:
        """Create default agents appropriate for this node's tier."""
        tier_models = {
            "T1": "llama3.1:70b",
            "T2": "llama3.1:8b",
            "T3": "llama3.2:3b",
            "T4": "phi3:mini",
        }
        model = tier_models.get(self.node_tier, "phi3:mini")

        inference_config = AgentConfig(
            name="inference-primary",
            agent_type="inference",
            model=model,
            ollama_url=self.ollama_url,
            litellm_url=self.litellm_url,
        )
        self.registry.register(InferenceAgent(inference_config))

        # Only provision embedding agent on T1–T3
        if self.node_tier in ("T1", "T2", "T3"):
            embed_config = AgentConfig(
                name="embedding-primary",
                agent_type="embedding",
                model="nomic-embed-text",
                ollama_url=self.ollama_url,
            )
            self.registry.register(EmbeddingAgent(embed_config))

    # ─── Routing ──────────────────────────────────────────────────────────────

    async def route(
        self,
        task_type: str,
        payload: dict,
        task_id: Optional[str] = None,
        context: Optional[AgentContext] = None,
        force_local: bool = False,
    ) -> AgentResult:
        """
        Route a task to the best handler.

        1. Check if the task should go remote (tier mismatch, specialized role)
        2. Otherwise dispatch to a local agent
        """
        task_id = task_id or str(uuid.uuid4())[:8]

        rule = self.routing_table.get(task_type)

        # Attempt remote routing if tier doesn't match and orchestrator available
        if (
            not force_local
            and rule
            and rule.preferred_tier
            and rule.preferred_tier != self.node_tier
            and self._remote_handler
        ):
            try:
                task = await self._remote_handler.submit_task(
                    task_type=task_type,
                    payload=payload,
                    required_tier=rule.preferred_tier,
                    required_role=rule.required_role,
                )
                # Wait for result
                result = await self._wait_for_remote_result(task, timeout=60.0)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"[router] remote routing failed for {task_type}: {e}. Falling back local.")

        # Local dispatch
        return await self._dispatch_local(task_id, task_type, payload, context)

    async def _dispatch_local(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext],
    ) -> AgentResult:
        candidates = self.registry.find(task_type)

        if not candidates:
            logger.warning(f"[router] no local agent for task_type={task_type}")
            return AgentResult(
                task_id=task_id,
                agent_name="router",
                success=False,
                error=f"No agent available for task type '{task_type}'",
            )

        # Pick least busy agent (lowest active task count proxy via state)
        idle = [a for a in candidates if a.state == AgentState.IDLE]
        agent = idle[0] if idle else candidates[0]

        logger.info(f"[router] routing task {task_id} (type={task_type}) → agent={agent.name}")
        return await agent.dispatch(task_id, task_type, payload, context)

    async def _wait_for_remote_result(self, task: Any, timeout: float) -> Optional[AgentResult]:
        """Poll for a completed remote task."""
        from core.coordinator.orchestrator import TaskStatus
        deadline = time.time() + timeout
        while time.time() < deadline:
            if task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                if task.status == TaskStatus.DONE:
                    return AgentResult(
                        task_id=task.id,
                        agent_name=task.assigned_to or "remote",
                        success=True,
                        output=task.result,
                    )
                else:
                    return AgentResult(
                        task_id=task.id,
                        agent_name=task.assigned_to or "remote",
                        success=False,
                        error=task.error or "Remote task failed",
                    )
            await asyncio.sleep(0.5)
        return None

    # ─── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "tier": self.node_tier,
            "agents": self.registry.status(),
            "routes": list(self.routing_table.keys()),
        }

    def add_route(self, rule: RouteRule) -> None:
        self.routing_table[rule.task_type] = rule
        logger.info(f"[router] route added: {rule.task_type} → tier={rule.preferred_tier}")

    def register_agent(self, agent: BaseAgent) -> None:
        self.registry.register(agent)


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

async def _demo():
    import argparse

    parser = argparse.ArgumentParser(description="MYCONEX Task Router Demo")
    parser.add_argument("--tier", default="T3")
    parser.add_argument("--task", default="chat")
    parser.add_argument("--prompt", default="What is a fungal network?")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    router = TaskRouter(node_tier=args.tier, ollama_url=args.ollama_url)
    await router.start()

    print(f"\nRouting '{args.task}' task to local agent...")
    result = await router.route(
        task_type=args.task,
        payload={"prompt": args.prompt},
        force_local=True,
    )

    if result.success:
        print(f"\nResult from {result.agent_name} ({result.model_used}):")
        print(f"  {result.output}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
    else:
        print(f"\nError: {result.error}")

    await router.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
