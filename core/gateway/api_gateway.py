"""
MYCONEX API Gateway

aiohttp HTTP server that exposes TaskRouter.route() to local clients.
Manages per-session AgentContext so tools like Claude CLI, Copilot-X,
and Gemini can maintain conversation state across requests via context_id.

Default: http://localhost:8765

Endpoints
---------
POST /task
    Body:    {"task_type": "chat", "prompt": "...", "context_id": "abc123", "payload": {...}}
    Returns: {"success": true, "response": "...", "context_id": "abc123",
              "agent": "inference-primary", "model": "llama3.1:8b",
              "duration_ms": 1234.5, "error": null}

POST /chat
    Shorthand for task_type=chat.
    Body:    {"prompt": "...", "context_id": "abc123"}
    Returns: same shape as /task

GET /status
    Returns router.status() + active session count.

GET /health
    Returns {"status": "ok", "sessions": N}

GET /session/{context_id}
    Returns session metadata: turn count, last-4 turns preview.

DELETE /session/{context_id}
    Clears a session from the store.

POST /session/seed
    Pulls Discord channel history and seeds a new (or existing) session.
    Body:    {"channel_id": "1234567890", "limit": 50, "context_id": "abc123"}
    Returns: {"context_id": "abc123", "turns_loaded": 47}
    Requires DISCORD_BOT_TOKEN in environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Optional

from aiohttp import web

from orchestration.agents.base_agent import AgentContext
from orchestration.workflows.task_router import TaskRouter

logger = logging.getLogger(__name__)

# ─── Gateway ──────────────────────────────────────────────────────────────────


class APIGateway:
    """
    aiohttp HTTP server wrapping TaskRouter with session management.

    Sessions are stored in-memory as {context_id: AgentContext}.
    Context objects are passed directly into TaskRouter.route(), so
    the InferenceAgent appends turns in-place — no explicit write-back needed.

    Lifecycle::

        gw = APIGateway(config, router)
        await gw.start()   # blocks until cancelled
        await gw.stop()
    """

    def __init__(
        self,
        config: dict,
        router: TaskRouter,
    ) -> None:
        self._config = config
        self._router = router

        api_cfg = config.get("api", {})
        self._host: str = api_cfg.get("host", "127.0.0.1")
        self._port: int = int(api_cfg.get("port", 8765))

        # context_id → AgentContext (in-memory; survives for process lifetime)
        self._sessions: dict[str, AgentContext] = {}

        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._setup_routes()

    # ─── Routes ───────────────────────────────────────────────────────────────

    def _setup_routes(self) -> None:
        r = self._app.router
        r.add_post("/task", self._handle_task)
        r.add_post("/chat", self._handle_chat)
        r.add_get("/status", self._handle_status)
        r.add_get("/health", self._handle_health)
        r.add_post("/session/seed", self._handle_session_seed)
        r.add_get("/session/{context_id}", self._handle_session_get)
        r.add_delete("/session/{context_id}", self._handle_session_delete)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the aiohttp server and block until cancelled."""
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info("[api] Listening on http://%s:%d", self._host, self._port)
        try:
            # Block here until the surrounding task is cancelled
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("[api] Stopped.")

    # ─── Session Helpers ──────────────────────────────────────────────────────

    def _get_or_create_session(self, context_id: Optional[str]) -> tuple[str, AgentContext]:
        """Return (context_id, AgentContext), creating a new session if needed."""
        if context_id and context_id in self._sessions:
            return context_id, self._sessions[context_id]
        cid = context_id or str(uuid.uuid4())[:8]
        ctx = AgentContext()
        self._sessions[cid] = ctx
        return cid, ctx

    # ─── Handlers ─────────────────────────────────────────────────────────────

    async def _handle_task(self, request: web.Request) -> web.Response:
        """
        POST /task
        General-purpose task submission. Supports all task_types in the routing table.
        """
        try:
            body = await request.json()
        except Exception:
            return _json_error("invalid JSON body", 400)

        task_type = body.get("task_type") or body.get("type") or "chat"
        prompt = body.get("prompt") or body.get("message") or ""
        context_id = body.get("context_id")

        # Build the payload: inline payload dict wins; prompt key fills it
        payload: dict = body.get("payload") or {}
        if prompt and "prompt" not in payload:
            payload["prompt"] = prompt

        if not payload:
            return _json_error("'prompt' or 'payload' is required", 400)

        cid, ctx = self._get_or_create_session(context_id)

        try:
            result = await self._router.route(
                task_type=task_type,
                payload=payload,
                context=ctx,
            )
        except Exception as e:
            logger.error("[api] route error: %s", e, exc_info=True)
            return _json_error(f"internal error: {e}", 500)

        out = result.output or {}
        response_text = out.get("response") if isinstance(out, dict) else str(out)

        return _json_ok({
            "success": result.success,
            "response": response_text,
            "context_id": cid,
            "agent": result.agent_name,
            "model": result.model_used,
            "duration_ms": round(result.duration_ms, 1),
            "error": result.error,
        })

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """
        POST /chat
        Convenience wrapper — always routes as task_type=chat.
        """
        try:
            body = await request.json()
        except Exception:
            return _json_error("invalid JSON body", 400)

        prompt = body.get("prompt") or body.get("message") or ""
        if not prompt:
            return _json_error("'prompt' is required", 400)

        context_id = body.get("context_id")
        cid, ctx = self._get_or_create_session(context_id)

        try:
            result = await self._router.route(
                task_type="chat",
                payload={"prompt": prompt},
                context=ctx,
            )
        except Exception as e:
            logger.error("[api] chat route error: %s", e, exc_info=True)
            return _json_error(f"internal error: {e}", 500)

        out = result.output or {}
        response_text = out.get("response") if isinstance(out, dict) else str(out)

        return _json_ok({
            "success": result.success,
            "response": response_text,
            "context_id": cid,
            "agent": result.agent_name,
            "model": result.model_used,
            "duration_ms": round(result.duration_ms, 1),
            "error": result.error,
        })

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /status — router status + active session count."""
        data = self._router.status()
        data["active_sessions"] = len(self._sessions)
        return _json_ok(data)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — liveness probe."""
        return _json_ok({"status": "ok", "sessions": len(self._sessions)})

    async def _handle_session_get(self, request: web.Request) -> web.Response:
        """GET /session/{context_id} — session metadata and turn preview."""
        cid = request.match_info["context_id"]
        ctx = self._sessions.get(cid)
        if ctx is None:
            return _json_error("session not found", 404)

        turns = ctx.history
        return _json_ok({
            "context_id": cid,
            "session_id": ctx.session_id,
            "turn_count": len(turns),
            "metadata": ctx.metadata,
            "preview": [
                {"role": t.role, "content": t.content[:160]}
                for t in turns[-4:]
            ],
        })

    async def _handle_session_delete(self, request: web.Request) -> web.Response:
        """DELETE /session/{context_id} — clear a session."""
        cid = request.match_info["context_id"]
        removed = self._sessions.pop(cid, None)
        return _json_ok({"removed": removed is not None, "context_id": cid})

    async def _handle_session_seed(self, request: web.Request) -> web.Response:
        """
        POST /session/seed
        Fetches Discord channel history and seeds an AgentContext.
        Requires DISCORD_BOT_TOKEN in environment.

        Body: {"channel_id": "...", "limit": 50, "context_id": "..."}
        """
        try:
            body = await request.json()
        except Exception:
            return _json_error("invalid JSON body", 400)

        channel_id = body.get("channel_id")
        if not channel_id:
            return _json_error("'channel_id' is required", 400)

        limit = int(body.get("limit", 50))
        context_id = body.get("context_id")

        token = os.getenv("DISCORD_BOT_TOKEN", "")
        if not token or token.startswith("REPLACE_"):
            return _json_error("DISCORD_BOT_TOKEN is not set in environment", 500)

        try:
            from core.gateway.chat_history_retriever import ChatHistoryRetriever

            retriever = ChatHistoryRetriever(token)
            ctx = await retriever.seed_context(channel_id=int(channel_id), limit=limit)
            await retriever.close()
        except Exception as e:
            logger.error("[api] seed_context failed: %s", e, exc_info=True)
            return _json_error(f"seed failed: {e}", 500)

        cid = context_id or str(uuid.uuid4())[:8]
        self._sessions[cid] = ctx

        return _json_ok({
            "context_id": cid,
            "turns_loaded": len(ctx.history),
            "channel_id": str(channel_id),
        })


# ─── Response Helpers ─────────────────────────────────────────────────────────


def _json_ok(data: dict) -> web.Response:
    return web.Response(
        status=200,
        content_type="application/json",
        body=json.dumps(data),
    )


def _json_error(msg: str, status: int) -> web.Response:
    return web.Response(
        status=status,
        content_type="application/json",
        body=json.dumps({"error": msg}),
    )
