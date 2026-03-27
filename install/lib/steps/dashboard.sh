#!/usr/bin/env bash
# dashboard.sh — web dashboard deps
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_dashboard"

step_dashboard() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing dashboard"
    local py; py=$(command -v python3.11 || command -v python3)
    # Dashboard lives at dashboard/app.py; its deps are in requirements.txt
    "$py" -m pip install --quiet gradio 2>/dev/null || true
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "dashboard: done"
}
