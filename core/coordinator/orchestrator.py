"""
MYCONEX Mesh Orchestrator
Central coordinator that manages node roles, routes tasks by tier,
monitors peer health, and maintains mesh topology state.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.classifier.hardware import HardwareDetector, HardwareProfile
from core.discovery.mesh_discovery import MeshDiscovery, MeshPeer
from core.messaging.nats_client import (
    HeartbeatService,
    MeshMessage,
    MeshNATSClient,
)

logger = logging.getLogger(__name__)


# ─── Task Model ───────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    ROUTED = "routed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class MeshTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""                        # e.g. "inference", "embedding", "search"
    payload: dict = field(default_factory=dict)
    required_tier: Optional[str] = None  # T1/T2/T3/T4 or None for any
    required_role: Optional[str] = None  # e.g. "large-model"
    priority: int = TaskPriority.NORMAL
    submitter: str = ""
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout: float = 60.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "required_tier": self.required_tier,
            "required_role": self.required_role,
            "priority": self.priority,
            "submitter": self.submitter,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at,
            "timeout": self.timeout,
        }


# ─── Node State ───────────────────────────────────────────────────────────────

@dataclass
class NodeState:
    name: str
    tier: str
    roles: list[str]
    address: str
    port: int
    last_heartbeat: float = field(default_factory=time.time)
    active_tasks: int = 0
    is_healthy: bool = True
    hardware: Optional[dict] = None


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class MeshOrchestrator:
    """
    Main coordinator for the MYCONEX AI mesh.

    Responsibilities:
    - Detect local hardware, classify tier, advertise via mDNS
    - Maintain live mesh topology via heartbeats and discovery
    - Route incoming tasks to appropriate nodes by tier/role/load
    - Track task lifecycle (submit → route → result)
    - Serve as coordinator if this node is T1/T2/T3
    """

    HEARTBEAT_TIMEOUT = 30.0  # seconds before marking node unhealthy

    def __init__(self, config: dict):
        self.config = config
        self.node_name: str = (
            config.get("node", {}).get("name") or "myconex-node"
        )
        self.nats_url: str = config.get("nats", {}).get("url", "nats://localhost:4222")
        self.api_port: int = config.get("mesh", {}).get("api_port", 8765)

        self._hardware: Optional[HardwareProfile] = None
        self._discovery: Optional[MeshDiscovery] = None
        self._nats: Optional[MeshNATSClient] = None
        self._heartbeat: Optional[HeartbeatService] = None

        self._nodes: dict[str, NodeState] = {}          # all known mesh nodes
        self._tasks: dict[str, MeshTask] = {}           # active/recent tasks
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._lock = asyncio.Lock()

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info(f"[orchestrator] starting node={self.node_name}")

        # 1. Detect hardware
        self._hardware = HardwareDetector().detect()
        logger.info(
            f"[orchestrator] hardware: tier={self._hardware.tier} "
            f"gpu={self._hardware.gpu_name} ram={self._hardware.ram_total_gb}GB"
        )

        # Register self in nodes dict
        self._register_self()

        # 2. Start mDNS discovery
        self._discovery = MeshDiscovery(
            node_name=self.node_name,
            tier=self._hardware.tier,
            roles=self._hardware.roles,
            api_port=self.api_port,
            on_peer_join=self._on_peer_join,
            on_peer_leave=self._on_peer_leave,
        )
        await self._discovery.start()

        # 3. Connect to NATS
        try:
            self._nats = MeshNATSClient(
                node_name=self.node_name,
                nats_url=self.nats_url,
            )
            await self._nats.connect()
            await self._setup_subscriptions()
        except Exception as e:
            logger.warning(f"[orchestrator] NATS unavailable: {e}. Running in local mode.")

        # 4. Start heartbeat
        if self._nats and self._nats.is_connected:
            self._heartbeat = HeartbeatService(
                client=self._nats,
                node_info=self._node_info_dict(),
                interval=10.0,
            )
            await self._heartbeat.start()

        # 5. Start background workers
        self._running = True
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._task_dispatcher())

        logger.info(
            f"[orchestrator] online — tier={self._hardware.tier} "
            f"roles={self._hardware.roles}"
        )

    async def stop(self) -> None:
        self._running = False

        if self._heartbeat:
            await self._heartbeat.stop()
        if self._nats:
            await self._nats.disconnect()
        if self._discovery:
            await self._discovery.stop()

        logger.info("[orchestrator] stopped.")

    # ─── Self Registration ────────────────────────────────────────────────────

    def _register_self(self) -> None:
        hw = self._hardware
        self._nodes[self.node_name] = NodeState(
            name=self.node_name,
            tier=hw.tier,
            roles=hw.roles,
            address="localhost",
            port=self.api_port,
            is_healthy=True,
            hardware={
                "cpu_cores": hw.cpu_cores_logical,
                "ram_gb": hw.ram_total_gb,
                "gpu": hw.gpu_name,
                "vram_gb": hw.gpu_vram_gb,
            },
        )

    # ─── NATS Subscriptions ───────────────────────────────────────────────────

    async def _setup_subscriptions(self) -> None:
        await self._nats.subscribe_self(self._handle_direct_message)
        await self._nats.subscribe_broadcast(self._handle_broadcast)
        await self._nats.subscribe_heartbeats(self._handle_heartbeat)

        # Only T1/T2/T3 nodes process tasks from the shared queue
        if self._hardware.tier in ("T1", "T2", "T3"):
            await self._nats.subscribe_tasks(self._handle_task_message, queue="workers")

        logger.info("[orchestrator] NATS subscriptions active.")

    async def _handle_direct_message(self, msg: MeshMessage) -> None:
        logger.debug(f"[orchestrator] direct msg from {msg.sender}: {msg.payload}")
        # Dispatch by payload type
        payload = msg.payload or {}
        msg_type = payload.get("type", "")

        if msg_type == "task":
            await self._enqueue_task_from_payload(payload, source=msg.sender)
        elif msg_type == "query_roster":
            await self._nats.reply(msg, self._get_roster())

    async def _handle_broadcast(self, msg: MeshMessage) -> None:
        logger.debug(f"[orchestrator] broadcast from {msg.sender}: {msg.payload}")

    async def _handle_heartbeat(self, msg: MeshMessage) -> None:
        payload = msg.payload or {}
        sender = msg.sender or payload.get("name", "")
        if not sender or sender == self.node_name:
            return

        async with self._lock:
            if sender in self._nodes:
                self._nodes[sender].last_heartbeat = time.time()
                self._nodes[sender].is_healthy = True
                self._nodes[sender].active_tasks = payload.get("active_tasks", 0)

    async def _handle_task_message(self, msg: MeshMessage) -> None:
        payload = msg.payload or {}
        await self._enqueue_task_from_payload(payload, source=msg.sender)

    # ─── Task Management ──────────────────────────────────────────────────────

    async def submit_task(
        self,
        task_type: str,
        payload: dict,
        required_tier: Optional[str] = None,
        required_role: Optional[str] = None,
        priority: int = TaskPriority.NORMAL,
        timeout: float = 60.0,
    ) -> MeshTask:
        task = MeshTask(
            type=task_type,
            payload=payload,
            required_tier=required_tier,
            required_role=required_role,
            priority=priority,
            submitter=self.node_name,
            timeout=timeout,
        )

        async with self._lock:
            self._tasks[task.id] = task

        # Priority queue: lower number = higher priority
        await self._task_queue.put((-task.priority, task.id))
        logger.info(f"[orchestrator] task submitted: {task.id} type={task_type}")
        return task

    async def _enqueue_task_from_payload(self, payload: dict, source: str) -> None:
        task = MeshTask(
            id=payload.get("id", str(uuid.uuid4())),
            type=payload.get("task_type", payload.get("type", "unknown")),
            payload=payload.get("payload", payload),
            required_tier=payload.get("required_tier"),
            required_role=payload.get("required_role"),
            priority=payload.get("priority", TaskPriority.NORMAL),
            submitter=source,
            timeout=payload.get("timeout", 60.0),
        )
        async with self._lock:
            self._tasks[task.id] = task
        await self._task_queue.put((-task.priority, task.id))

    async def _task_dispatcher(self) -> None:
        """Background loop: dequeue tasks and route them."""
        while self._running:
            try:
                _, task_id = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0
                )
                async with self._lock:
                    task = self._tasks.get(task_id)

                if task and task.status == TaskStatus.PENDING:
                    await self._route_task(task)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[dispatcher] error: {e}")

    async def _route_task(self, task: MeshTask) -> None:
        target = await self._select_node(task)

        if not target:
            task.status = TaskStatus.FAILED
            task.error = "No suitable node available"
            logger.warning(f"[orchestrator] no node for task {task.id} (type={task.type})")
            return

        task.assigned_to = target.name
        task.status = TaskStatus.ROUTED
        task.started_at = time.time()

        logger.info(f"[orchestrator] routing task {task.id} → {target.name} ({target.tier})")

        if target.name == self.node_name:
            # Handle locally
            await self._execute_task_locally(task)
        elif self._nats and self._nats.is_connected:
            # Send to remote node
            await self._nats.send_to_node(target.name, {
                "type": "task",
                **task.to_dict(),
            })
        else:
            task.status = TaskStatus.FAILED
            task.error = "Cannot reach remote node (NATS unavailable)"

    async def _execute_task_locally(self, task: MeshTask) -> None:
        """Placeholder local task execution — real logic lives in agents."""
        task.status = TaskStatus.RUNNING
        logger.info(f"[orchestrator] executing task {task.id} locally")

        # Simulate execution — agents will override this
        await asyncio.sleep(0.1)
        task.status = TaskStatus.DONE
        task.result = {"status": "ok", "executed_by": self.node_name}
        task.completed_at = time.time()

        # Notify submitter if via NATS
        if self._nats and task.submitter != self.node_name:
            await self._nats.send_to_node(task.submitter, {
                "type": "task_result",
                "task_id": task.id,
                "result": task.result,
                "status": task.status.value,
            })

    # ─── Node Selection ───────────────────────────────────────────────────────

    async def _select_node(self, task: MeshTask) -> Optional[NodeState]:
        """Pick the best available node for a task."""
        async with self._lock:
            candidates = list(self._nodes.values())

        healthy = [n for n in candidates if n.is_healthy]

        if task.required_tier:
            healthy = [n for n in healthy if n.tier == task.required_tier]

        if task.required_role:
            healthy = [n for n in healthy if task.required_role in n.roles]

        if not healthy:
            return None

        # Prefer nodes with fewer active tasks, then by tier rank
        tier_rank = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        return min(healthy, key=lambda n: (n.active_tasks, tier_rank.get(n.tier, 5)))

    # ─── Peer Callbacks ───────────────────────────────────────────────────────

    def _on_peer_join(self, peer: MeshPeer) -> None:
        state = NodeState(
            name=peer.name,
            tier=peer.tier,
            roles=peer.roles,
            address=peer.address,
            port=peer.port,
        )
        self._nodes[peer.name] = state
        logger.info(f"[mesh] node joined: {peer.name} ({peer.tier})")

    def _on_peer_leave(self, name: str) -> None:
        if name in self._nodes:
            self._nodes[name].is_healthy = False
        logger.info(f"[mesh] node offline: {name}")

    # ─── Health Monitor ───────────────────────────────────────────────────────

    async def _health_monitor(self) -> None:
        """Mark nodes unhealthy if heartbeats go stale."""
        while self._running:
            now = time.time()
            async with self._lock:
                for node in self._nodes.values():
                    if node.name == self.node_name:
                        continue
                    age = now - node.last_heartbeat
                    if age > self.HEARTBEAT_TIMEOUT and node.is_healthy:
                        node.is_healthy = False
                        logger.warning(f"[health] node {node.name} timed out ({age:.0f}s)")
            await asyncio.sleep(5.0)

    # ─── Status ───────────────────────────────────────────────────────────────

    def _get_roster(self) -> dict:
        return {
            n: {
                "tier": s.tier,
                "roles": s.roles,
                "healthy": s.is_healthy,
                "active_tasks": s.active_tasks,
            }
            for n, s in self._nodes.items()
        }

    def _node_info_dict(self) -> dict:
        hw = self._hardware
        return {
            "name": self.node_name,
            "tier": hw.tier,
            "roles": hw.roles,
            "active_tasks": sum(
                1 for t in self._tasks.values()
                if t.status in (TaskStatus.RUNNING, TaskStatus.ROUTED)
                and t.assigned_to == self.node_name
            ),
        }

    def status(self) -> dict:
        return {
            "node": self.node_name,
            "tier": self._hardware.tier if self._hardware else "unknown",
            "roles": self._hardware.roles if self._hardware else [],
            "nodes_online": sum(1 for n in self._nodes.values() if n.is_healthy),
            "nodes_total": len(self._nodes),
            "tasks_active": sum(
                1 for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.ROUTED, TaskStatus.RUNNING)
            ),
            "nats_connected": self._nats.is_connected if self._nats else False,
        }
