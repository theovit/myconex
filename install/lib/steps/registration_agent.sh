#!/usr/bin/env bash
# registration_agent.sh — lightweight node: Ollama + mesh announce agent
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_registration_agent"

step_registration_agent() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing mesh registration agent"
    # The registration agent is a thin Python script bundled with myconex
    local py; py=$(command -v python3.11 || command -v python3)
    "$py" -m pip install --quiet -r "${MYCONEX_REPO_ROOT}/requirements.txt"
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "registration_agent: done"
}
