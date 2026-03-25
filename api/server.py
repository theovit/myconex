"""
MYCONEX HTTP + WebSocket API Server
=====================================
Lightweight aiohttp-based server providing external integration endpoints.

Endpoints:
  POST   /chat                   — send a message, get agent response
  GET    /status                 — system health and metrics
  POST   /tools/{tool_name}      — invoke a specific tool directly
  GET    /agents                 — list agents and divisions
  GET    /agents/{name}/status   — single agent status
  POST   /autonomous/start       — start the optimization loop
  POST   /autonomous/stop        — stop the optimization loop
  GET    /autonomous/status      — loop status and metrics
  GET    /metrics                — full metrics report
  GET    /novelty/status         — novelty scanner status
  POST   /novelty/scan           — trigger an immediate scan
  GET    /health                 — self-healer health report
  WS     /ws                     — real-time streaming responses

Authentication:
  Bearer token via Authorization header.  Set MYCONEX_API_KEY env var.
  When not set, authentication is disabled (localhost-only recommended).

Usage:
    server = APIServer(agent=rlm_agent)
    await server.start(host="127.0.0.1", port=8765)
    # or:
    python3 -m myconex --mode api
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _json_response(data: Any, status: int = 200) -> "aiohttp.web.Response":
    """Return an aiohttp JSON response."""
    from aiohttp import web
    return web.Response(
        text=json.dumps(data, default=str),
        content_type="application/json",
        status=status,
    )


def _error(message: str, status: int = 400) -> "aiohttp.web.Response":
    return _json_response({"error": message}, status=status)


def _require_json(request: "aiohttp.web.Request"):
    """Return parsed JSON body or raise HTTPBadRequest."""
    pass  # handled inline in each handler for clarity


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Middleware
# ═══════════════════════════════════════════════════════════════════════════════

def _make_auth_middleware(api_key: Optional[str]):
    """Returns an aiohttp middleware that enforces Bearer token auth if api_key is set."""
    from aiohttp import web

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # Skip auth for health endpoint and websocket
        if request.path in ("/health", "/status") or request.path.startswith("/ws"):
            return await handler(request)
        if api_key:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return _error("Missing Authorization header", status=401)
            token = auth_header[len("Bearer "):]
            if token != api_key:
                return _error("Invalid API key", status=403)
        return await handler(request)

    return auth_middleware


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Manager
# ═══════════════════════════════════════════════════════════════════════════════

class WebSocketManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: set["aiohttp.web.WebSocketResponse"] = set()

    def register(self, ws: "aiohttp.web.WebSocketResponse") -> None:
        self._connections.add(ws)

    def unregister(self, ws: "aiohttp.web.WebSocketResponse") -> None:
        self._connections.discard(ws)

    async def broadcast(self, event: str, data: Any) -> None:
        """Send a JSON event to all connected WebSocket clients."""
        if not self._connections:
            return
        payload = json.dumps({"event": event, "data": data, "ts": time.time()}, default=str)
        dead: list = []
        for ws in list(self._connections):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


# ═══════════════════════════════════════════════════════════════════════════════
# API Server
# ═══════════════════════════════════════════════════════════════════════════════

class APIServer:
    """
    Async HTTP + WebSocket API server for MYCONEX.

    Provides a REST interface for external systems and a real-time WebSocket
    channel for streaming responses and event notifications.

    Usage:
        server = APIServer(
            agent=rlm_agent,
            autonomous_loop=loop,
            novelty_scanner=scanner,
            self_healer=healer,
        )
        await server.start(host="127.0.0.1", port=8765)
    """

    def __init__(
        self,
        agent=None,                         # RLMAgent
        router=None,                        # TaskRouter
        autonomous_loop=None,               # AutonomousOptimizationLoop
        novelty_scanner=None,               # NoveltyScanner
        self_healer=None,                   # SelfHealer
        metrics_collector=None,             # MetricsCollector
        agent_roster=None,                  # AgentRoster
        api_key: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.router = router
        self.autonomous_loop = autonomous_loop
        self.novelty_scanner = novelty_scanner
        self.self_healer = self_healer
        self.metrics_collector = metrics_collector
        self.agent_roster = agent_roster
        self.api_key = api_key or os.environ.get("MYCONEX_API_KEY")
        self._ws_manager = WebSocketManager()
        self._app: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._autonomous_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Start the HTTP server and begin listening."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("[api_server] aiohttp not installed — run: pip install aiohttp")
            raise

        self._app = web.Application(middlewares=[_make_auth_middleware(self.api_key)])
        self._register_routes()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

        logger.info("[api_server] listening on http://%s:%d (auth=%s)",
                    host, port, "enabled" if self.api_key else "disabled")

    async def stop(self) -> None:
        """Gracefully stop the server."""
        if self._runner:
            await self._runner.cleanup()
        logger.info("[api_server] stopped")

    def _register_routes(self) -> None:
        from aiohttp import web
        app = self._app

        # Chat
        app.router.add_post("/chat", self._handle_chat)

        # Status + Health
        app.router.add_get("/status", self._handle_status)
        app.router.add_get("/health", self._handle_health)

        # Metrics
        app.router.add_get("/metrics", self._handle_metrics)

        # Tools
        app.router.add_post("/tools/{tool_name}", self._handle_tool)

        # Agents
        app.router.add_get("/agents", self._handle_list_agents)
        app.router.add_get("/agents/{name}/status", self._handle_agent_status)

        # Autonomous loop
        app.router.add_post("/autonomous/start", self._handle_autonomous_start)
        app.router.add_post("/autonomous/stop",  self._handle_autonomous_stop)
        app.router.add_get("/autonomous/status",  self._handle_autonomous_status)

        # Novelty scanner
        app.router.add_get("/novelty/status",  self._handle_novelty_status)
        app.router.add_post("/novelty/scan",   self._handle_novelty_scan)

        # WebSocket
        app.router.add_get("/ws", self._handle_websocket)

    # ── Route Handlers ────────────────────────────────────────────────────────

    async def _handle_chat(self, request: "aiohttp.web.Request"):
        """POST /chat — { "message": "...", "session_id": "..." (optional) }"""
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body")

        message = body.get("message", "").strip()
        if not message:
            return _error("'message' field required")

        session_id = body.get("session_id", "api-default")
        stream = bool(body.get("stream", False))

        if self.agent is None:
            return _error("No agent configured", status=503)

        t0 = time.time()
        try:
            if hasattr(self.agent, "handle_task"):
                response = await self.agent.handle_task(message)
            elif hasattr(self.agent, "chat"):
                response = await self.agent.chat(
                    [{"role": "user", "content": message}]
                )
            else:
                return _error("Agent does not support chat/handle_task", status=503)

            duration_ms = (time.time() - t0) * 1000

            if self.metrics_collector:
                self.metrics_collector.record_task(
                    message, success=True, duration_ms=duration_ms, agent="api"
                )

            await self._ws_manager.broadcast("chat_response", {
                "session_id": session_id,
                "message": message[:100],
                "response": response[:500] if isinstance(response, str) else str(response)[:500],
                "duration_ms": round(duration_ms, 1),
            })

            return _json_response({
                "response": response,
                "session_id": session_id,
                "duration_ms": round(duration_ms, 1),
            })

        except Exception as exc:
            logger.error("[api_server] chat error: %s", exc)
            if self.metrics_collector:
                self.metrics_collector.record_task(message, success=False, agent="api")
            return _error(f"Agent error: {exc}", status=500)

    async def _handle_status(self, request: "aiohttp.web.Request"):
        """GET /status — system overview."""
        uptime = time.time() - self._start_time
        data: dict = {
            "status": "ok",
            "uptime_s": round(uptime, 1),
            "agent": None,
            "router": None,
            "ws_connections": self._ws_manager.count,
        }

        if self.agent and hasattr(self.agent, "status"):
            try:
                data["agent"] = self.agent.status()
            except Exception:
                data["agent"] = {"error": "status unavailable"}

        if self.router and hasattr(self.router, "status"):
            try:
                data["router"] = self.router.status()
            except Exception:
                data["router"] = {"error": "status unavailable"}

        return _json_response(data)

    async def _handle_health(self, request: "aiohttp.web.Request"):
        """GET /health — self-healer health report."""
        if self.self_healer is None:
            return _json_response({"overall": "unknown", "message": "self-healer not configured"})
        try:
            report = self.self_healer.health_report()
            return _json_response(report)
        except Exception as exc:
            return _error(f"Health check error: {exc}", status=500)

    async def _handle_metrics(self, request: "aiohttp.web.Request"):
        """GET /metrics — full metrics snapshot."""
        if self.metrics_collector is None:
            try:
                from core.metrics import get_metrics
                report = get_metrics().report()
            except Exception as exc:
                return _error(f"Metrics unavailable: {exc}", status=503)
        else:
            report = self.metrics_collector.report()
        return _json_response(report)

    async def _handle_tool(self, request: "aiohttp.web.Request"):
        """POST /tools/{tool_name} — invoke a tool by name."""
        from aiohttp import web
        tool_name = request.match_info["tool_name"]
        try:
            body = await request.json()
        except Exception:
            body = {}

        try:
            from core.gateway.agentic_tools import call_tool
            t0 = time.time()
            result = await call_tool(tool_name, body)
            duration_ms = (time.time() - t0) * 1000
            if self.metrics_collector:
                self.metrics_collector.record_tool_call(tool_name, duration_ms=duration_ms)
            return _json_response({"tool": tool_name, "result": result, "duration_ms": round(duration_ms, 1)})
        except AttributeError:
            # call_tool may not exist; fall back to direct handler lookup
            try:
                from core.gateway.agentic_tools import get_tool_handler
                handler = get_tool_handler(tool_name)
                if handler is None:
                    return _error(f"Tool not found: {tool_name}", status=404)
                result = handler(body)
                return _json_response({"tool": tool_name, "result": result})
            except Exception as exc:
                return _error(f"Tool error: {exc}", status=500)
        except Exception as exc:
            logger.error("[api_server] tool error (%s): %s", tool_name, exc)
            return _error(f"Tool error: {exc}", status=500)

    async def _handle_list_agents(self, request: "aiohttp.web.Request"):
        """GET /agents — list all registered agents."""
        agents: list[dict] = []

        if self.agent_roster:
            try:
                agents = self.agent_roster.list_agents() if hasattr(self.agent_roster, "list_agents") else []
            except Exception:
                pass

        if not agents and self.router:
            try:
                agents = [
                    {"name": name, "status": "registered"}
                    for name in (self.router.list_agents() if hasattr(self.router, "list_agents") else [])
                ]
            except Exception:
                pass

        return _json_response({"agents": agents, "count": len(agents)})

    async def _handle_agent_status(self, request: "aiohttp.web.Request"):
        """GET /agents/{name}/status — single agent status."""
        name = request.match_info["name"]
        if self.router and hasattr(self.router, "get_agent"):
            agent = self.router.get_agent(name)
            if agent and hasattr(agent, "status"):
                return _json_response(agent.status())
        return _error(f"Agent not found: {name}", status=404)

    async def _handle_autonomous_start(self, request: "aiohttp.web.Request"):
        """POST /autonomous/start — start the optimization loop."""
        if self.autonomous_loop is None:
            return _error("Autonomous loop not configured", status=503)
        if self._autonomous_task and not self._autonomous_task.done():
            return _json_response({"status": "already_running"})

        try:
            body = await request.json()
        except Exception:
            body = {}

        interval_s = float(body.get("interval_s", 30.0))
        max_cycles = body.get("max_cycles")

        self._autonomous_task = asyncio.create_task(
            self.autonomous_loop.run(
                max_cycles=max_cycles,
                cycle_interval_s=interval_s,
            )
        )

        await self._ws_manager.broadcast("autonomous_started", {
            "interval_s": interval_s,
            "max_cycles": max_cycles,
        })

        logger.info("[api_server] autonomous loop started via API (interval=%ss)", interval_s)
        return _json_response({"status": "started", "interval_s": interval_s})

    async def _handle_autonomous_stop(self, request: "aiohttp.web.Request"):
        """POST /autonomous/stop — stop the optimization loop."""
        if self.autonomous_loop is None:
            return _error("Autonomous loop not configured", status=503)
        self.autonomous_loop.stop()
        await self._ws_manager.broadcast("autonomous_stopped", {})
        logger.info("[api_server] autonomous loop stopped via API")
        return _json_response({"status": "stopping"})

    async def _handle_autonomous_status(self, request: "aiohttp.web.Request"):
        """GET /autonomous/status — loop metrics and running state."""
        if self.autonomous_loop is None:
            return _json_response({"configured": False})
        running = (
            self._autonomous_task is not None
            and not self._autonomous_task.done()
        )
        data = {
            "configured": True,
            "running": running,
        }
        if hasattr(self.autonomous_loop, "metrics"):
            data["metrics"] = self.autonomous_loop.metrics.to_dict()
        return _json_response(data)

    async def _handle_novelty_status(self, request: "aiohttp.web.Request"):
        """GET /novelty/status — scanner status and queue depth."""
        if self.novelty_scanner is None:
            return _json_response({"configured": False})
        try:
            return _json_response({"configured": True, **self.novelty_scanner.status()})
        except Exception as exc:
            return _error(f"Scanner error: {exc}", status=500)

    async def _handle_novelty_scan(self, request: "aiohttp.web.Request"):
        """POST /novelty/scan — trigger an immediate scan."""
        if self.novelty_scanner is None:
            return _error("Novelty scanner not configured", status=503)
        try:
            # Run in background — respond immediately
            asyncio.create_task(self.novelty_scanner.run_once())
            await self._ws_manager.broadcast("novelty_scan_started", {})
            return _json_response({"status": "scan_triggered"})
        except Exception as exc:
            return _error(f"Scan trigger error: {exc}", status=500)

    async def _handle_websocket(self, request: "aiohttp.web.Request"):
        """
        WebSocket /ws — real-time event stream.

        Client messages:
          {"action": "ping"}
          {"action": "chat", "message": "..."}

        Server events:
          {"event": "chat_response", "data": {...}}
          {"event": "autonomous_started", "data": {...}}
          {"event": "autonomous_stopped", "data": {}}
          {"event": "novelty_scan_started", "data": {}}
          {"event": "pong", "data": {}}
        """
        from aiohttp import web, WSMsgType

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._ws_manager.register(ws)
        logger.debug("[api_server] WS client connected (total=%d)", self._ws_manager.count)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({"event": "error", "data": "invalid JSON"}))
                        continue

                    action = data.get("action", "")

                    if action == "ping":
                        await ws.send_str(json.dumps({"event": "pong", "data": {}}))

                    elif action == "chat":
                        message = data.get("message", "")
                        if message and self.agent:
                            asyncio.create_task(
                                self._ws_chat(ws, message, data.get("session_id", "ws"))
                            )
                        else:
                            await ws.send_str(json.dumps({
                                "event": "error",
                                "data": "no message or agent unavailable",
                            }))

                    elif action == "status":
                        await ws.send_str(json.dumps({
                            "event": "status",
                            "data": {"ws_connections": self._ws_manager.count},
                        }))

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._ws_manager.unregister(ws)
            logger.debug("[api_server] WS client disconnected (total=%d)", self._ws_manager.count)

        return ws

    async def _ws_chat(
        self,
        ws: "aiohttp.web.WebSocketResponse",
        message: str,
        session_id: str,
    ) -> None:
        """Run a chat task and stream the result back over a WebSocket."""
        t0 = time.time()
        try:
            # Notify start
            await ws.send_str(json.dumps({"event": "chat_start", "data": {"session_id": session_id}}))

            if hasattr(self.agent, "handle_task"):
                response = await self.agent.handle_task(message)
            else:
                response = await self.agent.chat([{"role": "user", "content": message}])

            duration_ms = (time.time() - t0) * 1000
            await ws.send_str(json.dumps({
                "event": "chat_response",
                "data": {
                    "session_id": session_id,
                    "response": response,
                    "duration_ms": round(duration_ms, 1),
                },
            }))

        except Exception as exc:
            await ws.send_str(json.dumps({
                "event": "chat_error",
                "data": {"error": str(exc), "session_id": session_id},
            }))


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_api_server(
    agent=None,
    router=None,
    autonomous_loop=None,
    novelty_scanner=None,
    self_healer=None,
    metrics_collector=None,
    agent_roster=None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> APIServer:
    """Create an APIServer from config defaults."""
    try:
        from config import get_config
        cfg = get_config()
        effective_host = cfg.api.host if not host else host
        effective_port = cfg.api.port if port == 8765 else port
    except Exception:
        effective_host = host
        effective_port = port

    return APIServer(
        agent=agent,
        router=router,
        autonomous_loop=autonomous_loop,
        novelty_scanner=novelty_scanner,
        self_healer=self_healer,
        metrics_collector=metrics_collector,
        agent_roster=agent_roster,
    )


async def run_api_server(
    agent=None,
    router=None,
    autonomous_loop=None,
    novelty_scanner=None,
    self_healer=None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Convenience function to create and start the API server."""
    server = create_api_server(
        agent=agent, router=router,
        autonomous_loop=autonomous_loop,
        novelty_scanner=novelty_scanner,
        self_healer=self_healer,
    )
    await server.start(host=host, port=port)
    logger.info("[api_server] running at http://%s:%d", host, port)
    try:
        await asyncio.Event().wait()  # run until cancelled
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
