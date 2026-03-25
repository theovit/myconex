#!/usr/bin/env bash
# Launch Brave with CDP remote debugging so the DLAM Playwright client can attach.
# Your existing profile (and DLAM login session) is preserved.
#
# Usage:
#   ./scripts/launch_brave_dlam.sh
#
# After running this, Brave opens as normal — just with port 9222 available
# so Buzlock can connect to the DLAM tab.

BRAVE="/opt/brave.com/brave/brave"
DEBUG_PORT=9222

# Check if already running with debug port
if curl -s "http://localhost:${DEBUG_PORT}/json/version" &>/dev/null; then
    echo "Brave is already running with remote debugging on port ${DEBUG_PORT}."
    echo "You're all set — Buzlock can connect to DLAM."
    exit 0
fi

echo "Launching Brave with --remote-debugging-port=${DEBUG_PORT} ..."
echo "(Your profile and DLAM login are preserved)"
echo ""
echo "After Brave opens, navigate to: https://dlam.rabbit.tech/"
echo "Then you can ask Buzlock to use DLAM for any web task."

exec "$BRAVE" \
    --remote-debugging-port="${DEBUG_PORT}" \
    --restore-last-session \
    "$@"
