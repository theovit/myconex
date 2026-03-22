"""
MYCONEX Chat History Retriever

Fetches Discord channel message history via discord.py and converts it
to an AgentContext for seeding MYCONEX sessions before task submission.

This makes it possible for tools like Claude CLI, Copilot-X, and Gemini
to pre-load conversation context from Discord before asking follow-up
questions or submitting tasks to the mesh.

Usage (standalone)::

    import asyncio, os
    from core.gateway.chat_history_retriever import ChatHistoryRetriever

    async def main():
        retriever = ChatHistoryRetriever(os.getenv("DISCORD_BOT_TOKEN"))
        ctx = await retriever.seed_context(channel_id=1234567890, limit=50)
        await retriever.close()
        print(ctx.to_messages())

    asyncio.run(main())

Usage (via API gateway)::

    curl -X POST http://localhost:8765/session/seed \\
         -H "Content-Type: application/json" \\
         -d '{"channel_id": "1234567890", "limit": 50}'

The returned context_id can then be passed to /task or /chat to include
this history as context in the next inference call.

Notes
-----
- Only `login()` is called (no WebSocket), so startup is fast (~1-2s).
- Messages with empty content (embeds, attachments only) are skipped.
- Consecutive messages from the same role are merged to reduce turn count.
- The bot's own messages are assigned the "assistant" role automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

from orchestration.agents.base_agent import AgentContext

logger = logging.getLogger(__name__)


class ChatHistoryRetriever:
    """
    Fetches Discord channel history and converts it to an AgentContext.

    Internally uses discord.py's HTTP client (REST only, no gateway WebSocket).
    Call ``close()`` when done to release the HTTP session.
    """

    def __init__(self, token: str) -> None:
        try:
            import discord as _discord  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "discord.py is not installed. Run: pip install discord.py"
            )

        if not token or token.startswith("REPLACE_"):
            raise ValueError(
                "DISCORD_BOT_TOKEN is not set. "
                "Add it to .env or export it before calling ChatHistoryRetriever."
            )

        self._token = token
        self._client: Optional[object] = None  # discord.Client, typed as object to avoid import-time dep

    # ─── Internal client management ───────────────────────────────────────────

    async def _get_client(self):
        """Return a logged-in discord.Client, creating one if needed."""
        if self._client is not None:
            return self._client

        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        # login() authenticates via REST only — no WebSocket connection.
        # This is sufficient for fetch_channel() + channel.history().
        await client.login(self._token)
        self._client = client
        logger.debug("[retriever] discord client logged in")
        return self._client

    # ─── Public API ───────────────────────────────────────────────────────────

    async def fetch_channel_history(
        self,
        channel_id: int,
        limit: int = 50,
        oldest_first: bool = True,
    ) -> list[dict]:
        """
        Fetch up to ``limit`` messages from a Discord channel.

        Returns a list of dicts::

            {
                "author":     "display_name",
                "author_id":  "numeric_id_string",
                "content":    "message text",
                "timestamp":  "2026-03-22T01:00:00+00:00",
                "is_bot":     False,
            }

        Messages with no text content (attachments/embeds only) are skipped.
        """
        client = await self._get_client()
        channel = await client.fetch_channel(channel_id)

        messages: list[dict] = []
        async for msg in channel.history(limit=limit, oldest_first=oldest_first):
            content = (msg.content or "").strip()
            if not content:
                continue  # skip embed/attachment-only messages
            messages.append({
                "author": msg.author.display_name,
                "author_id": str(msg.author.id),
                "content": content,
                "timestamp": msg.created_at.isoformat(),
                "is_bot": getattr(msg.author, "bot", False),
            })

        logger.info(
            "[retriever] fetched %d messages from channel %d (limit=%d)",
            len(messages),
            channel_id,
            limit,
        )
        return messages

    async def seed_context(
        self,
        channel_id: int,
        limit: int = 50,
        system_prompt: Optional[str] = None,
        bot_user_id: Optional[str] = None,
    ) -> AgentContext:
        """
        Build an AgentContext seeded with Discord channel history.

        Role assignment
        ~~~~~~~~~~~~~~~
        - Bot messages         → ``"assistant"`` role
        - All other messages   → ``"user"`` role

        Consecutive messages from the same role are merged (joined by ``\\n``)
        to reduce the turn count and avoid hitting context-window limits.

        Parameters
        ----------
        channel_id:
            Discord channel ID to fetch history from.
        limit:
            Maximum number of messages to fetch (oldest_first order).
        system_prompt:
            Optional system prompt prepended to the context.
        bot_user_id:
            Numeric Discord user ID of the bot (as string). If omitted,
            falls back to ``client.user.id`` (if available) or uses the
            ``is_bot`` flag to detect bot messages.

        Returns
        -------
        AgentContext
            Ready to pass directly to TaskRouter.route() or store in the
            API gateway session store via ``POST /session/seed``.
        """
        messages = await self.fetch_channel_history(channel_id, limit=limit)

        ctx = AgentContext()
        ctx.metadata["source"] = "discord"
        ctx.metadata["channel_id"] = str(channel_id)
        ctx.metadata["seeded_turns"] = len(messages)

        if system_prompt:
            ctx.add("system", system_prompt)

        # Resolve bot ID from logged-in client if not supplied
        if bot_user_id is None and self._client is not None:
            me = getattr(self._client, "user", None)
            if me is not None:
                bot_user_id = str(me.id)

        prev_role: Optional[str] = None
        for msg in messages:
            is_bot_msg = msg["is_bot"] or (
                bot_user_id is not None and msg["author_id"] == bot_user_id
            )
            role = "assistant" if is_bot_msg else "user"

            if role == prev_role and ctx.history:
                # Merge consecutive same-role messages to reduce turn count
                ctx.history[-1].content += "\n" + msg["content"]
            else:
                ctx.add(role, msg["content"])

            prev_role = role

        logger.info(
            "[retriever] seeded context: %d source messages → %d turns (channel=%d)",
            len(messages),
            len(ctx.history),
            channel_id,
        )
        return ctx

    async def close(self) -> None:
        """Close the discord HTTP session."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug("[retriever] close error (non-fatal): %s", e)
            self._client = None
            logger.debug("[retriever] closed")
