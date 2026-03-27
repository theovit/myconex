#!/usr/bin/env bash
# hermes.sh — hermes-agent + flash-moe submodules
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_hermes"

step_hermes() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Initialising hermes-agent and flash-moe submodules"
    git -C "${MYCONEX_REPO_ROOT:?}" submodule update --init --recursive \
        integrations/hermes-agent integrations/flash-moe
    local py; py=$(command -v python3.11 || command -v python3)
    "$py" -m pip install --quiet -e "${MYCONEX_REPO_ROOT}/integrations/hermes-agent"
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "hermes: done"
}
