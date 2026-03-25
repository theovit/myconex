#!/usr/bin/env bash
# MYCONEX local launcher
# Usage:
#   ./start.sh            — mesh + Discord bot + API (recommended)
#   ./start.sh discord    — Discord bot only (no API)
#   ./start.sh worker     — task worker only (no bot, no API)
#
# Install as a systemd service (persists across reboots):
#   sudo cp spore/systemd/myconex.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now myconex
#   journalctl -u myconex -f

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-full}"

export PYTHONPATH="$SCRIPT_DIR"

# Load .env if not already loaded
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
fi

# Ensure NATS is running (required for mesh coordination)
if ! docker ps --filter "name=myconex-nats" --filter "status=running" --format "{{.Names}}" | grep -q myconex-nats; then
    echo "Starting NATS (JetStream)..."
    docker run -d --name myconex-nats --restart unless-stopped -p 4222:4222 nats:latest -js 2>/dev/null \
        || docker start myconex-nats 2>/dev/null \
        || echo "WARNING: Could not start NATS — mesh coordination may be unavailable"
    sleep 1
fi

echo "Starting MYCONEX in '$MODE' mode..."
exec python3 "$SCRIPT_DIR/main.py" --mode "$MODE"
