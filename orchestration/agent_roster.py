"""
MYCONEX Agent Roster System
============================
Inspired by The Agency (147 agents across 12 divisions).

A division-based agent registry that organises agents into functional teams,
manages their lifecycle, and routes inter-division communication.

Divisions:
  ENGINEERING  — code generation, architecture review, debugging
  RESEARCH     — web research, document analysis, literature review
  SECURITY     — vulnerability scanning, threat analysis, code audit
  DATA         — data processing, analytics, embeddings, database ops
  DEVOPS       — deployment, infrastructure, monitoring, system ops
  QA           — testing, validation, quality checks

Each agent belongs to exactly one division and declares its specialties.
The roster handles:
  - Agent registration and lifecycle (start/stop)
  - Load-balanced routing by division + specialty
  - Inter-division messaging via the TaskRouter
  - Division-level status and health monitoring
  - Broadcast (fan-out) to all agents in a division or all divisions
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from orchestration.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentRole,
    AgentState,
    BaseAgent,
)

logger = logging.getLogger(__name__)


# ─── Divisions ────────────────────────────────────────────────────────────────

class Division(str, Enum):
    ENGINEERING = "engineering"
    RESEARCH    = "research"
    SECURITY    = "security"
    DATA        = "data"
    DEVOPS      = "devops"
    QA          = "qa"


# Division metadata: display name + mission brief
_DIVISION_META: dict[Division, dict] = {
    Division.ENGINEERING: {
        "name": "Engineering",
        "mission": "Code generation, architecture, debugging, and refactoring.",
        "default_task_types": {"code", "generate", "refactor", "debug", "review"},
    },
    Division.RESEARCH: {
        "name": "Research",
        "mission": "Web research, document analysis, and knowledge synthesis.",
        "default_task_types": {"search", "research", "summarize", "analyze", "ask"},
    },
    Division.SECURITY: {
        "name": "Security",
        "mission": "Threat analysis, code audit, and vulnerability assessment.",
        "default_task_types": {"audit", "scan", "threat", "security", "classify"},
    },
    Division.DATA: {
        "name": "Data",
        "mission": "Data processing, analytics, embeddings, and database operations.",
        "default_task_types": {"embedding", "embed", "vectorize", "data", "query", "transform"},
    },
    Division.DEVOPS: {
        "name": "DevOps",
        "mission": "Deployment, infrastructure monitoring, and system operations.",
        "default_task_types": {"deploy", "monitor", "sensor", "relay", "shell", "system"},
    },
    Division.QA: {
        "name": "QA",
        "mission": "Testing, validation, and quality assurance.",
        "default_task_types": {"test", "validate", "check", "verify", "qa"},
    },
}


# ─── Roster Agent ─────────────────────────────────────────────────────────────

@dataclass
class RosterAgent:
    """An agent registered in the roster with division and specialty metadata."""
    agent: BaseAgent
    division: Division
    specialties: list[str]             # fine-grained task tags, e.g. ["python", "security"]
    is_lead: bool = False              # division lead — preferred for inter-division routing
    joined_at: float = field(default_factory=time.time)
    tasks_handled: int = 0
    tasks_failed: int = 0

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def state(self) -> AgentState:
        return self.agent.state

    @property
    def is_available(self) -> bool:
        return self.agent.state == AgentState.IDLE

    def specialty_score(self, query_tags: list[str]) -> int:
        """Count how many query_tags match this agent's specialties."""
        query_set = {t.lower() for t in query_tags}
        spec_set = {s.lower() for s in self.specialties}
        return len(query_set & spec_set)


# ─── Division Registry ────────────────────────────────────────────────────────

class DivisionRegistry:
    """Manages agents within a single division."""

    def __init__(self, division: Division) -> None:
        self.division = division
        self.meta = _DIVISION_META[division]
        self._agents: dict[str, RosterAgent] = {}

    def add(self, ra: RosterAgent) -> None:
        self._agents[ra.name] = ra
        logger.info(
            "[roster:%s] agent joined: %s (lead=%s, specialties=%s)",
            self.division.value, ra.name, ra.is_lead, ra.specialties,
        )

    def remove(self, name: str) -> Optional[RosterAgent]:
        return self._agents.pop(name, None)

    def get(self, name: str) -> Optional[RosterAgent]:
        return self._agents.get(name)

    def all(self) -> list[RosterAgent]:
        return list(self._agents.values())

    def lead(self) -> Optional[RosterAgent]:
        """Return the division lead agent, or any available agent."""
        for ra in self._agents.values():
            if ra.is_lead:
                return ra
        # Fall back to any available agent
        for ra in self._agents.values():
            if ra.is_available:
                return ra
        return next(iter(self._agents.values()), None)

    def find(self, task_type: str, tags: Optional[list[str]] = None) -> Optional[RosterAgent]:
        """
        Find the best available agent for a task type + optional specialty tags.

        Selection order:
          1. Available agents that handle the task type AND match specialty tags
          2. Available agents that handle the task type (any specialty)
          3. Any agent in the division (fallback)
        """
        candidates = [ra for ra in self._agents.values() if ra.agent.can_handle(task_type)]
        if not candidates:
            # Fallback: check division default task types
            if task_type in self.meta["default_task_types"]:
                candidates = list(self._agents.values())

        if not candidates:
            return None

        # Prefer available agents
        available = [ra for ra in candidates if ra.is_available]
        pool = available if available else candidates

        if tags:
            # Score by specialty overlap
            scored = sorted(pool, key=lambda ra: ra.specialty_score(tags), reverse=True)
            return scored[0]

        # Prefer lead, then least-busy
        leads = [ra for ra in pool if ra.is_lead]
        if leads:
            return leads[0]
        return min(pool, key=lambda ra: ra.tasks_handled)

    def status(self) -> dict:
        return {
            "division": self.division.value,
            "name": self.meta["name"],
            "agents": len(self._agents),
            "available": sum(1 for ra in self._agents.values() if ra.is_available),
            "agents_detail": [
                {
                    "name": ra.name,
                    "state": ra.state.value,
                    "is_lead": ra.is_lead,
                    "specialties": ra.specialties,
                    "tasks": ra.tasks_handled,
                    "errors": ra.tasks_failed,
                }
                for ra in self._agents.values()
            ],
        }


# ─── Agent Roster ─────────────────────────────────────────────────────────────

class AgentRoster:
    """
    Division-based agent roster for MYCONEX.

    Manages agent lifecycle, load balancing, inter-division communication,
    and division-level status.

    Usage:
        roster = AgentRoster(router)

        # Register agents
        roster.add(inference_agent, Division.ENGINEERING,
                   specialties=["python", "architecture"], is_lead=True)
        roster.add(embed_agent, Division.DATA,
                   specialties=["embeddings", "vector"])

        # Route a task to best agent in a division
        result = await roster.route(Division.RESEARCH, "search",
                                    {"prompt": "quantum computing advances"})

        # Inter-division: Engineering asks Research for a summary
        result = await roster.inter_division(
            from_div=Division.ENGINEERING,
            to_div=Division.RESEARCH,
            task_type="summarize",
            payload={"prompt": "summarise this codebase"},
        )

        # Broadcast to all QA agents
        results = await roster.broadcast(Division.QA, "validate",
                                         {"prompt": "check all endpoints"})
    """

    def __init__(self, router: Optional[Any] = None) -> None:
        self._router = router
        self._divisions: dict[Division, DivisionRegistry] = {
            div: DivisionRegistry(div) for div in Division
        }
        self._agent_index: dict[str, Division] = {}  # name → division
        self._started = False

    # ── Agent Registration ────────────────────────────────────────────────────

    def add(
        self,
        agent: BaseAgent,
        division: Division,
        specialties: Optional[list[str]] = None,
        is_lead: bool = False,
    ) -> RosterAgent:
        """
        Register an agent in the specified division.

        Args:
            agent:       The BaseAgent instance.
            division:    Division to assign this agent to.
            specialties: Fine-grained skill tags (e.g. ["python", "pytorch"]).
            is_lead:     Mark as division lead (preferred for inter-division tasks).

        Returns:
            The RosterAgent wrapper.
        """
        ra = RosterAgent(
            agent=agent,
            division=division,
            specialties=specialties or [],
            is_lead=is_lead,
        )
        self._divisions[division].add(ra)
        self._agent_index[agent.name] = division
        if self._router:
            agent.set_router(self._router)
        return ra

    def remove(self, name: str) -> bool:
        """Unregister an agent from the roster."""
        division = self._agent_index.pop(name, None)
        if division:
            self._divisions[division].remove(name)
            return True
        return False

    def get(self, name: str) -> Optional[RosterAgent]:
        """Look up a roster agent by name."""
        division = self._agent_index.get(name)
        if division:
            return self._divisions[division].get(name)
        return None

    def division_agents(self, division: Division) -> list[RosterAgent]:
        return self._divisions[division].all()

    def all_agents(self) -> list[RosterAgent]:
        return [ra for div in self._divisions.values() for ra in div.all()]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start all registered agents."""
        tasks = [ra.agent.start() for ra in self.all_agents()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._started = True
        logger.info(
            "[roster] started %d agents across %d divisions",
            len(self.all_agents()), len(self._divisions),
        )

    async def stop_all(self) -> None:
        """Stop all registered agents."""
        tasks = [ra.agent.stop() for ra in self.all_agents()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._started = False

    async def start_division(self, division: Division) -> None:
        """Start all agents in a specific division."""
        tasks = [ra.agent.start() for ra in self._divisions[division].all()]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Routing ───────────────────────────────────────────────────────────────

    async def route(
        self,
        division: Division,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
        tags: Optional[list[str]] = None,
        task_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Route a task to the best available agent in the given division.

        Args:
            division:   Target division.
            task_type:  Type of task to dispatch.
            payload:    Task payload.
            context:    Optional conversation context.
            tags:       Specialty tags to prefer specific agents.
            task_id:    Optional task ID (auto-generated if None).

        Returns:
            AgentResult from the selected agent.
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        div_reg = self._divisions[division]
        ra = div_reg.find(task_type, tags)

        if ra is None:
            logger.warning(
                "[roster] no agent in %s for task_type=%s",
                division.value, task_type,
            )
            # Try TaskRouter as fallback
            if self._router:
                return await self._router.route(task_type, payload, task_id=task_id, context=context)
            return AgentResult(
                task_id=task_id,
                agent_name="roster",
                success=False,
                error=f"No agent in {division.value} can handle task_type={task_type!r}",
            )

        logger.info(
            "[roster] %s → agent=%s (div=%s, lead=%s)",
            task_type, ra.name, division.value, ra.is_lead,
        )
        result = await ra.agent.dispatch(task_id, task_type, payload, context)
        ra.tasks_handled += 1
        if not result.success:
            ra.tasks_failed += 1
        return result

    async def inter_division(
        self,
        from_div: Division,
        to_div: Division,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
        tags: Optional[list[str]] = None,
    ) -> AgentResult:
        """
        Route a task from one division to another.

        The sending division's context is preserved; the receiving division's
        lead agent handles the task.

        Args:
            from_div:   Originating division (logged only).
            to_div:     Target division.
            task_type:  Task type.
            payload:    Task payload (automatically annotated with from_div).
            context:    Shared context.
            tags:       Specialty preference tags.

        Returns:
            AgentResult from the target division.
        """
        annotated_payload = {
            **payload,
            "_inter_division": {
                "from": from_div.value,
                "to": to_div.value,
            },
        }
        logger.info(
            "[roster] inter-division: %s → %s (%s)",
            from_div.value, to_div.value, task_type,
        )
        return await self.route(to_div, task_type, annotated_payload, context, tags)

    async def broadcast(
        self,
        division: Division,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> list[AgentResult]:
        """
        Fan-out a task to ALL agents in a division in parallel.

        Useful for QA validation (everyone checks), security scans (all
        agents audit their area), or parallel research.

        Returns:
            List of AgentResults in the same order as division agent registration.
        """
        agents = self._divisions[division].all()
        if not agents:
            return []
        task_id_base = str(uuid.uuid4())[:6]
        tasks = [
            ra.agent.dispatch(f"{task_id_base}-{i}", task_type, payload, context)
            for i, ra in enumerate(agents)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for ra, result in zip(agents, results):
            ra.tasks_handled += 1
            if not result.success:
                ra.tasks_failed += 1
        return list(results)

    async def broadcast_all(
        self,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> dict[str, list[AgentResult]]:
        """
        Broadcast to ALL divisions simultaneously.

        Returns:
            Dict of division_name → list of AgentResults.
        """
        tasks = {
            div.value: self.broadcast(div, task_type, payload, context)
            for div in Division
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=False)
        return dict(zip(tasks.keys(), results))

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return full roster status across all divisions."""
        total = len(self.all_agents())
        available = sum(1 for ra in self.all_agents() if ra.is_available)
        return {
            "total_agents": total,
            "available_agents": available,
            "started": self._started,
            "divisions": {
                div.value: reg.status()
                for div, reg in self._divisions.items()
            },
        }

    def division_status(self, division: Division) -> dict:
        return self._divisions[division].status()

    def __repr__(self) -> str:
        counts = {
            div.value: len(reg.all())
            for div, reg in self._divisions.items()
            if reg.all()
        }
        return f"AgentRoster({counts})"


# ─── Factory: Default Roster ──────────────────────────────────────────────────

def build_default_roster(
    router: Any,
    ollama_url: str = "http://localhost:11434",
    tier: str = "T2",
) -> AgentRoster:
    """
    Build a sensible default roster wired to an existing TaskRouter.

    Creates one RLMAgent per active division and registers them.  All agents
    use the same Ollama backend but have division-specific system prompts.

    Args:
        router:      An already-started TaskRouter instance.
        ollama_url:  Ollama endpoint.
        tier:        Node tier (determines default model).

    Returns:
        Configured AgentRoster (agents not yet started — call roster.start_all()).
    """
    from orchestration.agents.rlm_agent import create_rlm_agent

    roster = AgentRoster(router=router)

    tier_models = {"T1": "llama3.1:70b", "T2": "llama3.1:8b", "T3": "llama3.2:3b", "T4": "phi3:mini"}
    model = tier_models.get(tier, "llama3.1:8b")

    division_configs = [
        (
            Division.ENGINEERING,
            "engineering-lead",
            "You are an expert software engineer in the MYCONEX mesh. "
            "You specialise in code generation, architecture review, debugging, and refactoring. "
            "Write clean, production-ready code with proper error handling.",
            ["python", "architecture", "code", "debug", "refactor"],
            True,   # is_lead
        ),
        (
            Division.RESEARCH,
            "research-lead",
            "You are a research specialist in the MYCONEX mesh. "
            "You excel at web research, document analysis, fact-checking, and synthesising information "
            "from multiple sources into clear, structured reports.",
            ["search", "research", "analysis", "summarize", "literature"],
            True,
        ),
        (
            Division.SECURITY,
            "security-lead",
            "You are a cybersecurity expert in the MYCONEX mesh. "
            "You specialise in threat analysis, code auditing, vulnerability assessment, "
            "and secure system design. Always think adversarially.",
            ["security", "audit", "vulnerability", "threat", "compliance"],
            True,
        ),
        (
            Division.DATA,
            "data-lead",
            "You are a data engineering specialist in the MYCONEX mesh. "
            "You handle data transformation, analytics, embedding generation, "
            "database queries, and structured data processing.",
            ["data", "sql", "analytics", "embedding", "etl", "transform"],
            True,
        ),
        (
            Division.DEVOPS,
            "devops-lead",
            "You are a DevOps and infrastructure engineer in the MYCONEX mesh. "
            "You handle deployment pipelines, system monitoring, container orchestration, "
            "and operational reliability. Focus on automation and observability.",
            ["deploy", "docker", "kubernetes", "monitoring", "infra", "linux"],
            True,
        ),
        (
            Division.QA,
            "qa-lead",
            "You are a quality assurance engineer in the MYCONEX mesh. "
            "You write and run tests, validate outputs, check for regressions, "
            "and ensure that all features meet acceptance criteria.",
            ["testing", "validation", "pytest", "qa", "regression", "coverage"],
            True,
        ),
    ]

    for division, name, system_prompt, specialties, is_lead in division_configs:
        agent = create_rlm_agent(
            name=name,
            ollama_url=ollama_url,
            ollama_model=model,
            system_prompt=system_prompt,
            memory_namespace=division.value,
        )
        roster.add(agent, division, specialties=specialties, is_lead=is_lead)

    logger.info(
        "[roster] default roster built: %d divisions, %d agents",
        len(Division), len(roster.all_agents()),
    )
    return roster
