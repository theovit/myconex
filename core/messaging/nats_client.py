"""
MYCONEX NATS Client
Async pub/sub and request/reply messaging over NATS for inter-node communication.
Supports subject routing, queue groups, and mesh broadcast patterns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Coroutine, Optional

import nats
from nats.aio.client import Client as NATSConn
from nats.aio.msg import Msg
from nats.errors import ConnectionClosedError, NoServersError, TimeoutError as NATSTimeout

logger = logging.getLogger(__name__)

# ─── Subject Conventions ──────────────────────────────────────────────────────
# mesh.broadcast          — all nodes
# mesh.node.<name>        — specific node
# mesh.tier.<T1|T2|T3|T4> — tier-specific broadcast
# mesh.task.submit        — submit task to coordinator
# mesh.task.result.<id>   — task result delivery
# mesh.heartbeat          — liveness pulses
# mesh.roster             — peer roster updates

SUBJECT_BROADCAST = "mesh.broadcast"
SUBJECT_HEARTBEAT = "mesh.heartbeat"
SUBJECT_ROSTER = "mesh.roster"
SUBJECT_TASK_SUBMIT = "mesh.task.submit"


def subject_node(node_name: str) -> str:
    return f"mesh.node.{node_name}"


def subject_tier(tier: str) -> str:
    return f"mesh.tier.{tier}"


def subject_task_result(task_id: str) -> str:
    return f"mesh.task.result.{task_id}"


# ─── Message Envelope ─────────────────────────────────────────────────────────

@dataclass
class MeshMessage:
    subject: str
    payload: Any
    sender: str = ""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def encode(self) -> bytes:
        return json.dumps({
            "id": self.msg_id,
            "sender": self.sender,
            "subject": self.subject,
            "payload": self.payload,
            "ts": self.timestamp,
        }).encode()

    @staticmethod
    def decode(data: bytes) -> "MeshMessage":
        d = json.loads(data.decode())
        return MeshMessage(
            subject=d.get("subject", ""),
            payload=d.get("payload"),
            sender=d.get("sender", ""),
            msg_id=d.get("id", ""),
            timestamp=d.get("ts", time.time()),
        )


# ─── Handler Type ─────────────────────────────────────────────────────────────

MessageHandler = Callable[[MeshMessage], Coroutine[Any, Any, None]]


# ─── NATS Client ──────────────────────────────────────────────────────────────

class MeshNATSClient:
    """
    Async NATS client for MYCONEX mesh messaging.

    Usage:
        client = MeshNATSClient(node_name="mynode", nats_url="nats://localhost:4222")
        await client.connect()
        await client.publish("mesh.broadcast", {"hello": "world"})
        await client.subscribe("mesh.node.mynode", my_handler)
        await client.disconnect()
    """

    def __init__(
        self,
        node_name: str,
        nats_url: str = "nats://localhost:4222",
        reconnect_attempts: int = 10,
        reconnect_delay: float = 2.0,
    ):
        self.node_name = node_name
        self.nats_url = nats_url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._nc: Optional[NATSConn] = None
        self._subscriptions: dict[str, Any] = {}
        self._connected = False
        self._connect_lock = asyncio.Lock()

    # ─── Connection ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._connected:
                return

            logger.info(f"[nats] connecting to {self.nats_url} as '{self.node_name}'")
            try:
                self._nc = await nats.connect(
                    self.nats_url,
                    name=self.node_name,
                    reconnect_time_wait=self.reconnect_delay,
                    max_reconnect_attempts=self.reconnect_attempts,
                    error_cb=self._on_error,
                    disconnected_cb=self._on_disconnect,
                    reconnected_cb=self._on_reconnect,
                    closed_cb=self._on_close,
                )
                self._connected = True
                logger.info(f"[nats] connected (client_id={self._nc.client_id})")
            except NoServersError as e:
                logger.error(f"[nats] no servers available: {e}")
                raise
            except Exception as e:
                logger.error(f"[nats] connection failed: {e}")
                raise

    async def disconnect(self) -> None:
        if self._nc and self._connected:
            await self._nc.drain()
            await self._nc.close()
            self._connected = False
            logger.info("[nats] disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._nc is not None and self._nc.is_connected

    # ─── Publish ──────────────────────────────────────────────────────────────

    async def publish(self, subject: str, payload: Any) -> None:
        if not self.is_connected:
            raise ConnectionClosedError("Not connected to NATS.")

        msg = MeshMessage(subject=subject, payload=payload, sender=self.node_name)
        await self._nc.publish(subject, msg.encode())
        logger.debug(f"[nats] published → {subject}")

    async def broadcast(self, payload: Any) -> None:
        await self.publish(SUBJECT_BROADCAST, payload)

    async def send_to_node(self, target_node: str, payload: Any) -> None:
        await self.publish(subject_node(target_node), payload)

    async def send_to_tier(self, tier: str, payload: Any) -> None:
        await self.publish(subject_tier(tier), payload)

    async def submit_task(self, task: dict) -> None:
        await self.publish(SUBJECT_TASK_SUBMIT, task)

    async def send_heartbeat(self, node_info: dict) -> None:
        await self.publish(SUBJECT_HEARTBEAT, node_info)

    # ─── Request / Reply ──────────────────────────────────────────────────────

    async def request(
        self, subject: str, payload: Any, timeout: float = 5.0
    ) -> Optional[MeshMessage]:
        if not self.is_connected:
            raise ConnectionClosedError("Not connected to NATS.")

        msg = MeshMessage(subject=subject, payload=payload, sender=self.node_name)
        try:
            response = await self._nc.request(subject, msg.encode(), timeout=timeout)
            return MeshMessage.decode(response.data)
        except NATSTimeout:
            logger.warning(f"[nats] request to {subject} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"[nats] request error on {subject}: {e}")
            return None

    # ─── Subscribe ────────────────────────────────────────────────────────────

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        queue: Optional[str] = None,
    ) -> None:
        if not self.is_connected:
            raise ConnectionClosedError("Not connected to NATS.")

        async def _wrapped(msg: Msg) -> None:
            try:
                mesh_msg = MeshMessage.decode(msg.data)
                mesh_msg.reply_to = msg.reply
                await handler(mesh_msg)
            except Exception as e:
                logger.error(f"[nats] handler error on {subject}: {e}")

        if subject in self._subscriptions:
            logger.warning(f"[nats] already subscribed to {subject}, skipping.")
            return

        if queue:
            sub = await self._nc.subscribe(subject, queue=queue, cb=_wrapped)
        else:
            sub = await self._nc.subscribe(subject, cb=_wrapped)

        self._subscriptions[subject] = sub
        logger.info(f"[nats] subscribed → {subject}" + (f" (queue={queue})" if queue else ""))

    async def unsubscribe(self, subject: str) -> None:
        sub = self._subscriptions.pop(subject, None)
        if sub:
            await sub.unsubscribe()
            logger.info(f"[nats] unsubscribed from {subject}")

    # ─── Convenience: subscribe to own node channel ───────────────────────────

    async def subscribe_self(self, handler: MessageHandler) -> None:
        """Subscribe to messages addressed to this node."""
        await self.subscribe(subject_node(self.node_name), handler)

    async def subscribe_broadcast(self, handler: MessageHandler) -> None:
        await self.subscribe(SUBJECT_BROADCAST, handler)

    async def subscribe_heartbeats(self, handler: MessageHandler) -> None:
        await self.subscribe(SUBJECT_HEARTBEAT, handler)

    async def subscribe_tasks(self, handler: MessageHandler, queue: str = "workers") -> None:
        """Queue-group subscription so only one node handles each task."""
        await self.subscribe(SUBJECT_TASK_SUBMIT, handler, queue=queue)

    # ─── Reply helper ─────────────────────────────────────────────────────────

    async def reply(self, original_msg: MeshMessage, payload: Any) -> None:
        if not original_msg.reply_to:
            logger.warning("[nats] reply_to missing on message, cannot reply.")
            return
        response = MeshMessage(
            subject=original_msg.reply_to,
            payload=payload,
            sender=self.node_name,
        )
        await self._nc.publish(original_msg.reply_to, response.encode())

    # ─── NATS Callbacks ───────────────────────────────────────────────────────

    async def _on_error(self, e: Exception) -> None:
        logger.error(f"[nats] error: {e}")

    async def _on_disconnect(self) -> None:
        self._connected = False
        logger.warning("[nats] disconnected from server.")

    async def _on_reconnect(self) -> None:
        self._connected = True
        logger.info("[nats] reconnected.")

    async def _on_close(self) -> None:
        self._connected = False
        logger.info("[nats] connection closed.")


# ─── Heartbeat Service ────────────────────────────────────────────────────────

class HeartbeatService:
    """Periodically broadcasts node liveness info."""

    def __init__(
        self,
        client: MeshNATSClient,
        node_info: dict,
        interval: float = 10.0,
    ):
        self.client = client
        self.node_info = node_info
        self.interval = interval
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[heartbeat] started (interval={self.interval}s)")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("[heartbeat] stopped.")

    async def _loop(self) -> None:
        while True:
            try:
                info = {**self.node_info, "ts": time.time()}
                await self.client.send_heartbeat(info)
                logger.debug(f"[heartbeat] pulse sent")
            except Exception as e:
                logger.warning(f"[heartbeat] failed to send: {e}")
            await asyncio.sleep(self.interval)


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

async def _demo():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="nats://localhost:4222")
    parser.add_argument("--name", default="demo-node")
    parser.add_argument("--mode", choices=["pub", "sub"], default="sub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    client = MeshNATSClient(node_name=args.name, nats_url=args.url)
    await client.connect()

    if args.mode == "sub":
        async def handler(msg: MeshMessage):
            print(f"[MSG] {msg.sender} → {msg.subject}: {msg.payload}")

        await client.subscribe_broadcast(handler)
        await client.subscribe_self(handler)
        print(f"Subscribed. Waiting for messages on {args.name}...")
        await asyncio.sleep(60)
    else:
        for i in range(5):
            await client.broadcast({"ping": i, "from": args.name})
            print(f"Broadcast {i} sent.")
            await asyncio.sleep(1)

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(_demo())
