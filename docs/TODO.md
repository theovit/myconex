# MYCONEX — Discord Bot Setup TODO

**Discord Application ID:** `1469408384070586432`
**Source reference:** `~/hermes-agent-repo/gateway/platforms/discord.py` (2085 lines)
**Platform base:** `~/hermes-agent-repo/gateway/platforms/base.py` (1313 lines)

---

## Status

- [x] MYCONEX core mesh system implemented (`main.py`, `core/`, `orchestration/`)
- [x] Discord platform adapter exists in hermes-agent-repo (`gateway/platforms/discord.py`)
- [x] Discord bot wired into MYCONEX (`core/gateway/discord_gateway.py`, `main.py`)
- [x] `discord.py==2.7.1` added to `pyproject.toml` and installed
- [x] `discord:` section added to `config/mesh_config.yaml`
- [x] `.env` created with `DISCORD_APPLICATION_ID=1469408384070586432` (token placeholder)

---

## Phase 1 — Dependencies & Credentials ✓ DONE

### 1.1 Add `discord.py` dependency ✓
- `pyproject.toml` updated: `"discord.py>=2.7.0"`
- Installed: `discord.py==2.7.1`
- Intents verified: `message_content`, `members`, `voice_states` all OK

### 1.2 Create `.env` file ✓
Create `/home/techno-shaman/myconex/.env` with:
```
DISCORD_BOT_TOKEN=<your-bot-token-from-discord-dev-portal>
DISCORD_APPLICATION_ID=1469408384070586432
DISCORD_HOME_CHANNEL=<channel-id-for-default-delivery>
DISCORD_HOME_CHANNEL_NAME=general
DISCORD_ALLOWED_USERS=          # comma-separated user IDs; empty = allow all
DISCORD_REQUIRE_MENTION=false   # true = bot only responds when @mentioned
DISCORD_FREE_RESPONSE_CHANNELS= # channel IDs where bot responds without mention
DISCORD_AUTO_THREAD=false       # true = create thread per conversation
DISCORD_ALLOW_BOTS=none         # none | all | allowlist
```

**Where to get the bot token:**
1. Go to https://discord.com/developers/applications/1469408384070586432
2. Bot → Reset Token → copy
3. Privileged Gateway Intents: enable **Message Content**, **Server Members**, **Presence** (if needed)

### 1.3 Add `.env` to `.gitignore` ✓
- Already present in `.gitignore` (was there before this session)

---

## Phase 2 — Integrate Discord Gateway into MYCONEX ✓ DONE

### 2.1 Add Discord section to `config/mesh_config.yaml`
```yaml
# ─── Discord Bot ──────────────────────────────────────────────────────────────
discord:
  enabled: true
  application_id: "1469408384070586432"
  require_mention: false
  auto_thread: false
  free_response_channels: []   # list of channel IDs
```

### 2.2 Create `core/gateway/discord_gateway.py`
- Copy/adapt `hermes-agent-repo/gateway/platforms/discord.py`
- Strip hermes-specific imports (`hermes_cli`, `gateway.config`, `gateway.platforms.base`)
- Replace with MYCONEX equivalents:
  - `BasePlatformAdapter` → simple base class or inline
  - `MessageEvent` → dataclass in MYCONEX
  - Route received messages → `TaskRouter.route(task_type="chat", payload={...})`

### 2.3 Update `main.py` — add Discord bot startup
In `run_node()`, after the task router is started:
```python
# 3. Start Discord bot (if configured)
discord_task = None
if cfg.get("discord", {}).get("enabled"):
    from core.gateway.discord_gateway import DiscordGateway
    discord_gw = DiscordGateway(cfg, router)
    discord_task = asyncio.create_task(discord_gw.start())
```
And in the shutdown block:
```python
if discord_task:
    discord_task.cancel()
```

### 2.4 Add `--mode discord` CLI option (optional)
- Extend the `mode` click option to include `"discord"` and `"full"` (mesh + discord)

---

## Phase 3 — Discord Bot Portal Configuration ✓ DONE

See **`docs/DISCORD_SETUP.md`** for the full step-by-step guide.

### 3.1 Register Slash Commands ✓
Slash commands are auto-synced on `on_ready` — no manual step required.
Registered commands: `/ask`, `/reset`, `/status`, `/tier`

### 3.2 Discord Developer Portal Settings ✓ (documented)
At https://discord.com/developers/applications/1469408384070586432:
- **OAuth2 → Scopes:** `bot`, `applications.commands`
- **Bot Permissions integer:** `414464683072` (Send Messages, Read History, Slash Commands, Embed Links, Attach Files, etc.)
- **Privileged Intents:** `Message Content Intent` ✓, `Server Members Intent` ✓
- **Invite URL:** `https://discord.com/oauth2/authorize?client_id=1469408384070586432&permissions=414464683072&scope=bot%20applications.commands`

### Outstanding manual steps (requires human action)
- [x] Go to Portal → Bot → **Reset Token** → paste into `.env`
- [x] Enable **Message Content Intent** in Portal → Bot → Privileged Gateway Intents
- [x] Enable **Server Members Intent** in Portal → Bot → Privileged Gateway Intents
- [x] Use invite URL above to add the bot to your Discord server

---

## Phase 4 — Testing

- [x] `python main.py status` — verify hardware/mesh detection works
- [x] `python main.py --mode worker` — start without Discord, confirm no errors
- [x] Set `DISCORD_BOT_TOKEN` in `.env` and run `python main.py --mode full`
- [x] Verify bot appears Online in Discord server
- [x] Verify slash commands appear (may take up to 1 hr for global sync; use guild sync for instant)
- [x] Send a message / use `/ask` — verify response routes through `TaskRouter`
- [x] Check logs for NATS connectivity (mesh messaging)

---

## File Map

| File | Status |
|------|--------|
| `pyproject.toml` | ✓ `discord.py>=2.7.0` added |
| `.env` | ✓ Created — **token set** |
| `.gitignore` | ✓ `.env` already excluded |
| `config/mesh_config.yaml` | ✓ `discord:` section added |
| `core/gateway/discord_gateway.py` | ✓ Created (310 lines) |
| `main.py` | ✓ `DiscordGateway` wired into `run_node()`, `discord` mode added |
| `docs/DISCORD_SETUP.md` | ✓ Created — full Portal configuration guide |
| `docs/TODO.md` | ✓ This file |

---

## Notes

- The `hermes-agent-repo/gateway/platforms/discord.py` is the **reference implementation** — it handles voice, threads, slash commands, allowed-users lists, and bot filtering. Port only what MYCONEX needs.
- MYCONEX's task routing (`T1/T2/T3/T4` tiers) should be the backend for Discord messages — Discord is just an input channel.
- The bot's Intents required: `message_content=True`, `dm_messages=True`, `guild_messages=True`, `members=True`.
