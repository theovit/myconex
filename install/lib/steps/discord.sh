#!/usr/bin/env bash
# discord.sh — discord.py deps + token collection
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_discord"

step_discord() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Configuring Discord gateway"
    # discord.py is already in requirements.txt — just collect the token
    if [[ -z "${DISCORD_BOT_TOKEN:-}" ]]; then
        if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
            echo "ERROR: DISCORD_BOT_TOKEN env var required for unattended Discord setup" >&2
            exit 1
        fi
        source "$(dirname "${BASH_SOURCE[0]}")/../ui.sh"
        ui_input "Discord bot token:" DISCORD_BOT_TOKEN secret
    fi
    # Token written to .env by config.sh — just export for this session
    export DISCORD_BOT_TOKEN
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "discord: done"
}
