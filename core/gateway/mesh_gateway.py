"""
MYCONEX Mesh Gateway
FastAPI service that bridges the mobile app to the NATS mesh.
Provides REST endpoints and WebSocket connections for real-time communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.messaging.nats_client import MeshNATSClient, MeshMessage
from core.discovery.mesh_discovery import MeshDiscovery, MeshPeer

logger = logging.getLogger(__name__)

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    type: str
    payload: Dict[str, Any]
    tier: Optional[str] = None
    role: Optional[str] = None
    timeout: Optional[float] = 60.0

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    context: Optional[List[Dict]] = None

class NodeInfo(BaseModel):
    name: str
    tier: str
    roles: List[str]
    address: str
    port: int
    status: str

# ─── WebSocket Manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

# ─── Mesh Gateway ─────────────────────────────────────────────────────────────

class MeshGateway:
    """
    FastAPI gateway service connecting mobile apps to the MYCONEX mesh.

    Features:
    - REST API for task submission and status queries
    - WebSocket connections for real-time mesh events
    - NATS integration for mesh communication
    - mDNS peer discovery integration
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        host: str = "0.0.0.0",
        port: int = 8765,
        cors_origins: List[str] = None,
    ):
        self.nats_url = nats_url
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]

        # Core services
        self.nats_client: Optional[MeshNATSClient] = None
        self.discovery: Optional[MeshDiscovery] = None
        self.ws_manager = ConnectionManager()

        # App state
        self.app = self._create_app()
        self._running = False

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with routes."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()

        app = FastAPI(
            title="MYCONEX Mesh Gateway",
            description="API gateway for MYCONEX distributed AI mesh",
            version="0.1.0",
            lifespan=lifespan,
        )

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Routes
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "services": {
                    "nats": self.nats_client.is_connected if self.nats_client else False,
                    "discovery": self.discovery._running if self.discovery else False,
                }
            }

        @app.get("/peers")
        async def get_peers():
            """Get list of discovered mesh peers."""
            if not self.discovery:
                raise HTTPException(status_code=503, detail="Discovery service not available")

            peers = await self.discovery.registry.get_online()
            return {
                "peers": [
                    {
                        "name": p.name,
                        "tier": p.tier,
                        "roles": p.roles,
                        "address": p.address,
                        "port": p.port,
                        "endpoint": p.endpoint,
                        "last_seen": p.last_seen,
                    }
                    for p in peers
                ]
            }

        @app.post("/task")
        async def submit_task(request: TaskRequest):
            """Submit a task to the mesh."""
            if not self.nats_client or not self.nats_client.is_connected:
                raise HTTPException(status_code=503, detail="NATS not connected")

            try:
                task_data = {
                    "id": f"task_{int(time.time())}_{hash(str(request.payload)) % 1000}",
                    "type": request.type,
                    "payload": request.payload,
                    "required_tier": request.tier,
                    "required_role": request.role,
                    "submitter": "gateway",
                    "timeout": request.timeout,
                    "created_at": time.time(),
                }

                await self.nats_client.submit_task(task_data)

                # Broadcast task submission to WebSocket clients
                await self.ws_manager.broadcast({
                    "type": "task_submitted",
                    "task_id": task_data["id"],
                    "task_type": request.type,
                })

                return {"task_id": task_data["id"], "status": "submitted"}

            except Exception as e:
                logger.error(f"Task submission failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/chat")
        async def chat(request: ChatRequest):
            """Send a chat message to the mesh."""
            if not self.nats_client or not self.nats_client.is_connected:
                raise HTTPException(status_code=503, detail="NATS not connected")

            try:
                chat_payload = {
                    "message": request.message,
                    "model": request.model,
                    "context": request.context or [],
                    "timestamp": time.time(),
                }

                # Route to appropriate tier (prefer T2 for chat)
                await self.nats_client.send_to_tier("T2", {
                    "type": "chat",
                    "payload": chat_payload,
                    "reply_subject": f"gateway.reply.{time.time()}",
                })

                return {"status": "sent", "message": "Chat message sent to mesh"}

            except Exception as e:
                logger.error(f"Chat failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time mesh events."""
            await self.ws_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive, events are pushed via broadcast
                    data = await websocket.receive_text()
                    # Echo back for connection health
                    await websocket.send_json({"type": "echo", "data": data})
            except WebSocketDisconnect:
                await self.ws_manager.disconnect(websocket)

        return app

    async def _startup(self):
        """Initialize gateway services."""
        logger.info("[gateway] starting up...")

        # Connect to NATS
        try:
            self.nats_client = MeshNATSClient(
                node_name="gateway",
                nats_url=self.nats_url,
            )
            await self.nats_client.connect()

            # Subscribe to mesh events for WebSocket broadcasting
            await self.nats_client.subscribe_broadcast(self._on_broadcast_message)
            await self.nats_client.subscribe_heartbeats(self._on_heartbeat)

        except Exception as e:
            logger.error(f"[gateway] NATS connection failed: {e}")
            # Continue without NATS - some endpoints will return 503

        # Start mDNS discovery (for peer info, not as a mesh node)
        try:
            # Minimal discovery for peer enumeration only
            self.discovery = MeshDiscovery(
                node_name="gateway-observer",
                tier="T3",  # Doesn't participate in mesh
                roles=[],
                api_port=self.port,
            )
            await self.discovery.start()
        except Exception as e:
            logger.error(f"[gateway] Discovery failed: {e}")

        self._running = True
        logger.info(f"[gateway] running on {self.host}:{self.port}")

    async def _shutdown(self):
        """Clean up gateway services."""
        logger.info("[gateway] shutting down...")

        if self.discovery:
            await self.discovery.stop()

        if self.nats_client:
            await self.nats_client.disconnect()

        self._running = False

    async def _on_broadcast_message(self, msg: MeshMessage):
        """Handle broadcast messages from mesh."""
        await self.ws_manager.broadcast({
            "type": "broadcast",
            "sender": msg.sender,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
        })

    async def _on_heartbeat(self, msg: MeshMessage):
        """Handle heartbeat messages."""
        await self.ws_manager.broadcast({
            "type": "heartbeat",
            "node": msg.sender,
            "info": msg.payload,
            "timestamp": msg.timestamp,
        })

    def run(self):
        """Run the gateway server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MYCONEX Mesh Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--nats-url", default="nats://localhost:4222", help="NATS server URL")

    args = parser.parse_args()

    gateway = MeshGateway(
        host=args.host,
        port=args.port,
        nats_url=args.nats_url,
    )
    gateway.run()