"""
MYCONEX Discord Gateway

Connects a Discord bot to the MYCONEX task-routing mesh.

Incoming Discord messages are routed through the TaskRouter as "chat" tasks.
Conversation history is kept per-channel in AgentContext sessions (in-memory).

Required env vars (set in .env):
    DISCORD_BOT_TOKEN           — bot token from Discord Developer Portal
    DISCORD_APPLICATION_ID      — 1469408384070586432

Optional env vars:
    DISCORD_ALLOWED_USERS       — comma-separated user IDs; empty = allow all
    DISCORD_REQUIRE_MENTION     — "true" / "false"  (default: false)
    DISCORD_FREE_RESPONSE_CHANNELS — comma-separated channel IDs
    DISCORD_AUTO_THREAD         — "true" / "false"  (default: false)
    DISCORD_ALLOW_BOTS          — "none" / "all"    (default: none)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─── Discord import guard ─────────────────────────────────────────────────────

try:
    import discord
    from discord import Message as DiscordMessage, Intents
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None          # type: ignore[assignment]
    DiscordMessage = Any    # type: ignore[assignment,misc]
    Intents = Any           # type: ignore[assignment]
    commands = None         # type: ignore[assignment]

# ─── MYCONEX imports ──────────────────────────────────────────────────────────

from orchestration.agents.base_agent import AgentContext
from orchestration.workflows.task_router import TaskRouter

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_MESSAGE_LENGTH = 2000
MAX_SESSION_TURNS = 30          # trim history after this many turns
MAX_TRACKED_THREADS = 500
THREAD_STATE_FILE = Path.home() / ".myconex" / "discord_threads.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_discord_id(entry: str) -> str:
    """Strip mention syntax and 'user:' prefix from a Discord ID entry."""
    entry = entry.strip()
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    if entry.lower().startswith("user:"):
        entry = entry[5:]
    return entry.strip()


def _chunk_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into Discord-safe chunks, preferring newline boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ─── Gateway ──────────────────────────────────────────────────────────────────

class DiscordGateway:
    """
    MYCONEX Discord bot gateway.

    Lifecycle::

        gw = DiscordGateway(config, router)
        await gw.start()   # blocks until cancelled
        await gw.stop()    # clean disconnect
    """

    def __init__(self, config: dict, router: TaskRouter) -> None:
        self._config = config
        self._router = router
        self._discord_cfg: dict = config.get("discord", {})

        self._token: str = (
            os.getenv("DISCORD_BOT_TOKEN")
            or self._discord_cfg.get("token", "")
        )
        self._app_id: str = (
            os.getenv("DISCORD_APPLICATION_ID", "1469408384070586432")
        )

        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._running = False

        # Access control
        self._allowed_user_ids: set[str] = set()

        # Per-channel conversation contexts  {channel_id: AgentContext}
        self._sessions: Dict[str, AgentContext] = {}

        # Track threads the bot has participated in (skip @mention requirement)
        self._bot_participated_threads: set[str] = self._load_thread_state()

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Discord and run the bot. Blocks until cancelled."""
        if not DISCORD_AVAILABLE:
            logger.error("[discord] discord.py not installed. Run: pip install discord.py")
            return

        if not self._token or self._token.startswith("REPLACE_"):
            logger.error(
                "[discord] DISCORD_BOT_TOKEN not set. "
                "Edit .env and add your bot token from "
                "https://discord.com/developers/applications/%s/bot",
                self._app_id,
            )
            return

        connected = await self._connect()
        if not connected:
            logger.error("[discord] Failed to connect — gateway not started.")
            return

        logger.info("[discord] Gateway running.")

    async def stop(self) -> None:
        """Disconnect cleanly."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning("[discord] Error during disconnect: %s", e)
        self._client = None
        self._ready_event.clear()
        logger.info("[discord] Disconnected.")

    # ─── Connection ───────────────────────────────────────────────────────────

    async def _connect(self) -> bool:
        try:
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = True

            self._client = commands.Bot(
                command_prefix="!",
                intents=intents,
                application_id=int(self._app_id),
            )

            # Parse allowed users from env
            allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
            if allowed_env:
                self._allowed_user_ids = {
                    _clean_discord_id(u) for u in allowed_env.split(",") if u.strip()
                }

            self._register_events()
            self._register_slash_commands()

            # Start bot in background task
            asyncio.create_task(self._client.start(self._token))

            # Wait for on_ready (30s timeout)
            await asyncio.wait_for(self._ready_event.wait(), timeout=30)
            self._running = True
            return True

        except asyncio.TimeoutError:
            logger.error("[discord] Timed out waiting for Discord connection.")
            return False
        except Exception as e:
            logger.error("[discord] Connection failed: %s", e, exc_info=True)
            return False

    # ─── Event Registration ───────────────────────────────────────────────────

    def _register_events(self) -> None:
        gw = self  # capture for closures

        @self._client.event
        async def on_ready():
            logger.info(
                "[discord] Connected as %s (ID: %s)",
                gw._client.user,
                gw._client.user.id,
            )
            # Resolve any username entries in allowed list to numeric IDs
            await gw._resolve_allowed_usernames()
            # Sync slash commands globally
            try:
                synced = await gw._client.tree.sync()
                logger.info("[discord] Synced %d slash command(s).", len(synced))
            except Exception as e:
                logger.warning("[discord] Slash command sync failed: %s", e)
            gw._ready_event.set()

        @self._client.event
        async def on_message(message: DiscordMessage):
            # Ignore our own messages
            if message.author == gw._client.user:
                return
            # Bot filtering
            if getattr(message.author, "bot", False):
                allow_bots = os.getenv("DISCORD_ALLOW_BOTS", "none").lower()
                if allow_bots == "none":
                    return
                if allow_bots == "mentions":
                    if gw._client.user not in message.mentions:
                        return
                # "all" falls through
            await gw._handle_message(message)

    # ─── Message Handling ─────────────────────────────────────────────────────

    async def _handle_message(self, message: DiscordMessage) -> None:
        """Filter, build context, route to TaskRouter, send reply."""

        is_thread = discord and isinstance(message.channel, discord.Thread)
        thread_id = str(message.channel.id) if is_thread else None
        parent_channel_id: Optional[str] = None
        if is_thread:
            parent = getattr(message.channel, "parent", None)
            if parent is not None:
                parent_channel_id = str(parent.id)

        # ── Mention / free-channel gating (server messages only) ──────────────
        is_dm = discord and isinstance(message.channel, discord.DMChannel)
        if not is_dm:
            free_raw = os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "")
            free_channels = {c.strip() for c in free_raw.split(",") if c.strip()}
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)

            require_mention = (
                os.getenv("DISCORD_REQUIRE_MENTION", "false").lower()
                not in ("false", "0", "no")
            )
            is_free = bool(channel_ids & free_channels)
            in_bot_thread = is_thread and thread_id in self._bot_participated_threads

            if require_mention and not is_free and not in_bot_thread:
                if self._client.user not in message.mentions:
                    return

            # Strip the @mention from content so it doesn't pollute the prompt
            if self._client.user and self._client.user in message.mentions:
                message.content = (
                    message.content
                    .replace(f"<@{self._client.user.id}>", "")
                    .replace(f"<@!{self._client.user.id}>", "")
                    .strip()
                )

        # ── Auto-thread ───────────────────────────────────────────────────────
        auto_thread = (
            os.getenv("DISCORD_AUTO_THREAD", "false").lower() in ("true", "1", "yes")
        )
        reply_channel = message.channel
        if auto_thread and not is_thread and not is_dm:
            thread = await self._auto_create_thread(message)
            if thread:
                reply_channel = thread
                thread_id = str(thread.id)
                is_thread = True
                self._track_thread(thread_id)

        # ── Access control ────────────────────────────────────────────────────
        if not self._is_allowed_user(str(message.author.id)):
            logger.debug("[discord] Ignoring message from non-allowed user %s", message.author.id)
            return

        prompt = message.content.strip()
        if not prompt:
            return

        # ── Track thread participation ─────────────────────────────────────────
        if thread_id:
            self._track_thread(thread_id)

        # ── Route through TaskRouter ───────────────────────────────────────────
        channel_id = str(reply_channel.id)
        context = self._get_or_create_session(channel_id)

        async with reply_channel.typing():
            result = await self._router.route(
                task_type="chat",
                payload={"prompt": prompt},
                context=context,
            )

        if result.success:
            output = result.output or {}
            response_text = output.get("response") or "*(no response)*"
        else:
            response_text = f"*(error: {result.error})*"

        await self._send_reply(reply_channel, response_text)

    async def _send_reply(self, channel, content: str, reference=None) -> None:
        """Send a (potentially long) reply, chunking at 2000 chars."""
        chunks = _chunk_message(content)
        for i, chunk in enumerate(chunks):
            try:
                await channel.send(
                    content=chunk,
                    reference=reference if i == 0 else None,
                )
            except Exception as e:
                logger.error("[discord] Failed to send message chunk: %s", e)

    # ─── Session Management ───────────────────────────────────────────────────

    def _get_or_create_session(self, channel_id: str) -> AgentContext:
        if channel_id not in self._sessions:
            self._sessions[channel_id] = AgentContext()
        ctx = self._sessions[channel_id]
        ctx.trim(MAX_SESSION_TURNS)
        return ctx

    def _reset_session(self, channel_id: str) -> None:
        self._sessions.pop(channel_id, None)

    # ─── Access Control ───────────────────────────────────────────────────────

    def _is_allowed_user(self, user_id: str) -> bool:
        if not self._allowed_user_ids:
            return True
        return user_id in self._allowed_user_ids

    async def _resolve_allowed_usernames(self) -> None:
        """Resolve username strings in allowed list to numeric IDs via guild members."""
        if not self._allowed_user_ids or not self._client:
            return

        numeric: set[str] = set()
        to_resolve: set[str] = set()
        for entry in self._allowed_user_ids:
            (numeric if entry.isdigit() else to_resolve).add(entry.lower())

        if not to_resolve:
            self._allowed_user_ids = numeric
            return

        logger.info("[discord] Resolving %d username(s): %s", len(to_resolve), to_resolve)
        for guild in self._client.guilds:
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                logger.warning("[discord] Could not fetch members for guild %s: %s", guild.name, e)
                continue
            for member in members:
                names = {
                    member.name.lower(),
                    member.display_name.lower(),
                    (member.global_name or "").lower(),
                }
                matched = names & to_resolve
                if matched:
                    uid = str(member.id)
                    numeric.add(uid)
                    to_resolve -= matched
                    logger.info("[discord] Resolved '%s' → %s", matched, uid)
            if not to_resolve:
                break

        if to_resolve:
            logger.warning("[discord] Could not resolve usernames: %s", to_resolve)

        self._allowed_user_ids = numeric
        os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric))

    # ─── Thread Helpers ───────────────────────────────────────────────────────

    async def _auto_create_thread(self, message: DiscordMessage):
        content = (message.content or "").strip()
        name = (content[:77] + "...") if len(content) > 80 else content or "MYCONEX"
        try:
            return await message.create_thread(name=name, auto_archive_duration=1440)
        except Exception as e:
            logger.warning("[discord] Auto-thread creation failed: %s", e)
            return None

    def _track_thread(self, thread_id: str) -> None:
        if thread_id not in self._bot_participated_threads:
            self._bot_participated_threads.add(thread_id)
            if len(self._bot_participated_threads) > MAX_TRACKED_THREADS:
                # Drop oldest entries (convert to list, keep last N)
                kept = list(self._bot_participated_threads)[-MAX_TRACKED_THREADS:]
                self._bot_participated_threads = set(kept)
            self._save_thread_state()

    def _load_thread_state(self) -> set[str]:
        try:
            if THREAD_STATE_FILE.exists():
                data = json.loads(THREAD_STATE_FILE.read_text())
                if isinstance(data, list):
                    return set(data)
        except Exception as e:
            logger.debug("[discord] Could not load thread state: %s", e)
        return set()

    def _save_thread_state(self) -> None:
        try:
            THREAD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            THREAD_STATE_FILE.write_text(
                json.dumps(list(self._bot_participated_threads))
            )
        except Exception as e:
            logger.debug("[discord] Could not save thread state: %s", e)

    # ─── Slash Commands ───────────────────────────────────────────────────────

    def _register_slash_commands(self) -> None:
        if not self._client:
            return

        tree = self._client.tree
        gw = self

        # /ask — one-shot question (no history)
        @tree.command(name="ask", description="Ask MYCONEX a one-shot question")
        @discord.app_commands.describe(prompt="Your question")
        async def slash_ask(interaction: discord.Interaction, prompt: str):
            await interaction.response.defer()
            if not gw._is_allowed_user(str(interaction.user.id)):
                await interaction.followup.send("Not authorized.", ephemeral=True)
                return
            result = await gw._router.route(
                task_type="chat",
                payload={"prompt": prompt},
                context=AgentContext(),  # fresh context, no history
            )
            if result.success:
                output = result.output or {}
                text = output.get("response") or str(output)
            else:
                text = f"*(error: {result.error})*"
            for chunk in _chunk_message(text):
                await interaction.followup.send(chunk)

        # /reset — clear conversation history for this channel
        @tree.command(name="reset", description="Clear conversation history for this channel")
        async def slash_reset(interaction: discord.Interaction):
            if not gw._is_allowed_user(str(interaction.user.id)):
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            channel_id = str(interaction.channel_id)
            gw._reset_session(channel_id)
            await interaction.response.send_message(
                "Conversation history cleared.", ephemeral=True
            )

        # /status — show MYCONEX node info
        @tree.command(name="status", description="Show MYCONEX node status")
        async def slash_status(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            router_status = gw._router.status()
            tier = router_status.get("tier", "?")
            agents = router_status.get("agents", [])
            agent_lines = "\n".join(
                f"  • {a['name']} ({a['type']}) — {a['state']} | model: {a['model']}"
                for a in agents
            )
            active_sessions = len(gw._sessions)
            lines = [
                f"**MYCONEX Node**",
                f"Tier: `{tier}`",
                f"Agents ({len(agents)}):\n{agent_lines}",
                f"Active sessions: {active_sessions}",
                f"Bot: {gw._client.user}",
            ]
            await interaction.followup.send("\n".join(lines), ephemeral=True)

        # /tier — quick tier check
        @tree.command(name="tier", description="Show the current node tier")
        async def slash_tier(interaction: discord.Interaction):
            tier = gw._router.node_tier
            await interaction.response.send_message(
                f"Node tier: `{tier}`", ephemeral=True
            )
