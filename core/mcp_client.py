"""
MYCONEX MCP Client
==================
Model Context Protocol (MCP) server connectivity for MYCONEX.

Bridges hermes-agent's mcp_tool into MYCONEX, enabling agents to:
  - Connect to any MCP-compatible server (local process or remote)
  - Discover tools exposed by MCP servers
  - Dispatch tool calls through the MCP protocol
  - Register discovered MCP tools into MYCONEX's agentic_tools system

MCP servers are configured in mesh_config.yaml under the `mcp.servers` key:

    mcp:
      servers:
        - name: filesystem
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        - name: github
          command: npx
          args: ["-y", "@modelcontextprotocol/server-github"]
          env:
            GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
        - name: custom-server
          command: python
          args: ["-m", "my_mcp_server"]

See https://modelcontextprotocol.io for the MCP spec.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MCPServerConfig":
        """Create from a config dict (e.g., loaded from YAML)."""
        env_raw = d.get("env", {})
        # Expand ${VAR} references
        env = {
            k: os.path.expandvars(str(v))
            for k, v in env_raw.items()
        }
        return cls(
            name=d["name"],
            command=d["command"],
            args=d.get("args", []),
            env=env,
            timeout_s=float(d.get("timeout_s", 30.0)),
            enabled=d.get("enabled", True),
        )


@dataclass
class MCPToolSpec:
    """A tool discovered from an MCP server."""
    server_name: str
    tool_name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"mcp_{self.server_name}_{self.tool_name}"

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling schema."""
        return {
            "name": self.full_name,
            "description": f"[MCP:{self.server_name}] {self.description}",
            "parameters": self.input_schema or {"type": "object", "properties": {}},
        }


# ─── MCP Client ───────────────────────────────────────────────────────────────

class MCPClient:
    """
    Manages connections to one or more MCP servers and dispatches tool calls.

    Tries two strategies in order:
    1. hermes mcp_tool (if hermes-agent is available) — most complete
    2. Direct JSON-RPC subprocess protocol — stdlib-only fallback

    Usage:
        client = MCPClient([MCPServerConfig("filesystem", "npx", ["-y", "@mcp/server-filesystem", "/tmp"])])
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("filesystem", "read_file", {"path": "/tmp/test.txt"})
        await client.disconnect()
    """

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self._servers = {s.name: s for s in servers if s.enabled}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._tools: dict[str, list[MCPToolSpec]] = {}   # server_name → tools
        self._hermes_available: Optional[bool] = None
        self._hermes_dispatch: Any = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> dict[str, bool]:
        """
        Connect to all configured MCP servers.

        Returns {server_name: connected} mapping.
        """
        results = {}
        for name, cfg in self._servers.items():
            try:
                ok = await self._connect_server(cfg)
                results[name] = ok
            except Exception as exc:
                logger.warning("[MCPClient] connect %s failed: %s", name, exc)
                results[name] = False
        return results

    async def disconnect(self) -> None:
        """Terminate all MCP server subprocesses."""
        for name, proc in list(self._processes.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._processes.clear()
        self._tools.clear()

    # ── Tool Discovery ────────────────────────────────────────────────────────

    async def list_tools(self, server_name: Optional[str] = None) -> list[MCPToolSpec]:
        """
        List all tools from one or all connected MCP servers.
        """
        if server_name:
            return self._tools.get(server_name, [])
        return [t for tools in self._tools.values() for t in tools]

    def get_openai_schemas(self) -> list[dict]:
        """Return all discovered MCP tools as OpenAI function schemas."""
        return [t.to_openai_schema() for t in await_or_sync(self.list_tools())]

    # ── Tool Dispatch ─────────────────────────────────────────────────────────

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
        timeout_s: Optional[float] = None,
    ) -> str:
        """
        Call a tool on a specific MCP server.

        Returns the tool result as a JSON string.
        """
        cfg = self._servers.get(server_name)
        if cfg is None:
            return json.dumps({"error": f"Unknown MCP server: {server_name}"})

        # Strategy 1: hermes mcp_tool dispatch
        if await self._try_hermes_dispatch(server_name, tool_name, arguments):
            return self._last_hermes_result  # type: ignore[attr-defined]

        # Strategy 2: Direct JSON-RPC over stdio
        proc = self._processes.get(server_name)
        if proc is None:
            return json.dumps({"error": f"MCP server {server_name!r} not connected"})

        return await self._jsonrpc_call(
            proc, tool_name, arguments,
            timeout_s=timeout_s or cfg.timeout_s,
        )

    async def call_tool_by_full_name(
        self,
        full_tool_name: str,
        arguments: dict,
    ) -> str:
        """
        Dispatch using the full_name format `mcp_{server}_{tool}`.
        """
        # Strip "mcp_" prefix, then split on first "_" to get server name
        if full_tool_name.startswith("mcp_"):
            remainder = full_tool_name[4:]
        else:
            remainder = full_tool_name

        # Find matching server (longest prefix match)
        for server_name in sorted(self._servers.keys(), key=len, reverse=True):
            prefix = server_name.replace("-", "_").replace(" ", "_")
            if remainder.startswith(prefix + "_"):
                tool_name = remainder[len(prefix) + 1:]
                return await self.call_tool(server_name, tool_name, arguments)

        return json.dumps({"error": f"Cannot parse MCP tool name: {full_tool_name}"})

    # ── MYCONEX Registration ──────────────────────────────────────────────────

    async def register_with_myconex(self) -> int:
        """
        Register all discovered MCP tools into MYCONEX's agentic_tools registry.

        Returns number of tools registered.
        """
        tools = await self.list_tools()
        if not tools:
            return 0

        registered = 0
        try:
            from integrations.hermes_bridge import HermesToolBridge  # type: ignore[import]
            registry = HermesToolBridge.get_registry()
            if registry is None:
                return 0

            for tool in tools:
                try:
                    server_name = tool.server_name
                    tool_name = tool.tool_name
                    client_ref = self  # capture for closure

                    def make_handler(sname: str, tname: str):
                        def handler(**kwargs) -> str:
                            loop = asyncio.new_event_loop()
                            try:
                                return loop.run_until_complete(
                                    client_ref.call_tool(sname, tname, kwargs)
                                )
                            finally:
                                loop.close()
                        return handler

                    registry.register(
                        name=tool.full_name,
                        toolset="mcp",
                        schema=tool.to_openai_schema(),
                        handler=make_handler(server_name, tool_name),
                        description=tool.description,
                    )
                    registered += 1
                except Exception as exc:
                    logger.debug("[MCPClient] register tool %s failed: %s", tool.full_name, exc)
        except Exception as exc:
            logger.warning("[MCPClient] MYCONEX registration failed: %s", exc)

        logger.info("[MCPClient] registered %d MCP tools", registered)
        return registered

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _connect_server(self, cfg: MCPServerConfig) -> bool:
        """Start MCP server subprocess and perform handshake."""
        env = {**os.environ, **cfg.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                cfg.command, *cfg.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._processes[cfg.name] = proc

            # MCP initialize handshake
            await self._jsonrpc_send(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "myconex", "version": "0.2.0"},
                },
            })
            init_resp = await asyncio.wait_for(
                self._jsonrpc_read(proc), timeout=cfg.timeout_s
            )
            if not init_resp or "error" in init_resp:
                logger.warning("[MCPClient] initialize failed for %s: %s", cfg.name, init_resp)
                return False

            # Send initialized notification
            await self._jsonrpc_send(proc, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })

            # Discover tools
            await self._discover_tools(cfg.name, proc, cfg.timeout_s)
            logger.info(
                "[MCPClient] connected %s: %d tools",
                cfg.name, len(self._tools.get(cfg.name, [])),
            )
            return True
        except Exception as exc:
            logger.warning("[MCPClient] connect %s failed: %s", cfg.name, exc)
            return False

    async def _discover_tools(
        self, server_name: str, proc: asyncio.subprocess.Process, timeout_s: float
    ) -> None:
        await self._jsonrpc_send(proc, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        resp = await asyncio.wait_for(self._jsonrpc_read(proc), timeout=timeout_s)
        if resp is None or "error" in resp:
            return
        raw_tools = resp.get("result", {}).get("tools", [])
        self._tools[server_name] = [
            MCPToolSpec(
                server_name=server_name,
                tool_name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in raw_tools
        ]

    async def _jsonrpc_call(
        self,
        proc: asyncio.subprocess.Process,
        tool_name: str,
        arguments: dict,
        timeout_s: float = 30.0,
    ) -> str:
        req_id = id(arguments)
        await self._jsonrpc_send(proc, {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        resp = await asyncio.wait_for(self._jsonrpc_read(proc), timeout=timeout_s)
        if resp is None:
            return json.dumps({"error": "no response from MCP server"})
        if "error" in resp:
            return json.dumps({"error": resp["error"]})
        result = resp.get("result", {})
        content = result.get("content", [])
        # Flatten text content blocks
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else json.dumps(result)
        return json.dumps(result)

    @staticmethod
    async def _jsonrpc_send(proc: asyncio.subprocess.Process, payload: dict) -> None:
        line = json.dumps(payload) + "\n"
        proc.stdin.write(line.encode())
        await proc.stdin.drain()

    @staticmethod
    async def _jsonrpc_read(proc: asyncio.subprocess.Process) -> Optional[dict]:
        try:
            line = await proc.stdout.readline()
            if not line:
                return None
            return json.loads(line.decode().strip())
        except (json.JSONDecodeError, Exception):
            return None

    async def _try_hermes_dispatch(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> bool:
        """Try dispatching via hermes mcp_tool. Returns True if successful."""
        if self._hermes_available is False:
            return False
        if self._hermes_available is None:
            try:
                from integrations.hermes_bridge import setup_hermes_path  # type: ignore[import]
                setup_hermes_path()
                from model_tools import handle_function_call  # type: ignore[import]
                self._hermes_available = True
                self._hermes_dispatch = handle_function_call
            except ImportError:
                self._hermes_available = False
                return False
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._hermes_dispatch(
                    "mcp_call",
                    {"server": server_name, "tool": tool_name, "arguments": arguments},
                ),
            )
            self._last_hermes_result = result  # type: ignore[attr-defined]
            return True
        except Exception:
            return False


# ─── MCPManager singleton ─────────────────────────────────────────────────────

_MCP_MANAGER: Optional[MCPClient] = None


def get_mcp_client() -> Optional[MCPClient]:
    """Return the shared MCPClient instance."""
    return _MCP_MANAGER


async def setup_mcp_from_config(config: dict) -> Optional[MCPClient]:
    """
    Initialize the global MCPClient from MYCONEX config dict.

    config should have structure:
        config["mcp"]["servers"] = [{"name": ..., "command": ..., "args": ...}, ...]
    """
    global _MCP_MANAGER
    mcp_cfg = config.get("mcp", {})
    server_dicts = mcp_cfg.get("servers", [])
    if not server_dicts:
        return None

    servers = []
    for d in server_dicts:
        try:
            servers.append(MCPServerConfig.from_dict(d))
        except Exception as exc:
            logger.warning("[MCPClient] invalid server config %s: %s", d, exc)

    if not servers:
        return None

    client = MCPClient(servers)
    results = await client.connect()
    connected = [name for name, ok in results.items() if ok]

    if connected:
        logger.info("[MCPClient] connected servers: %s", connected)
        _MCP_MANAGER = client
        return client

    logger.warning("[MCPClient] no MCP servers connected")
    return None


# ─── Utility ─────────────────────────────────────────────────────────────────

def await_or_sync(coro):
    """Run a coroutine synchronously if no event loop is running."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — can't use run_until_complete
        # Caller should await directly
        return []
    except RuntimeError:
        return asyncio.run(coro)
