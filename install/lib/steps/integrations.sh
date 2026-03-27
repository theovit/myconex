#!/usr/bin/env bash
# integrations.sh — optional ingester API keys (Gmail, YouTube, RSS, podcast)
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_integrations"

step_integrations() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Configuring integrations"
    # Deps already in requirements.txt; collect optional API keys
    for var in GMAIL_CLIENT_ID GMAIL_CLIENT_SECRET; do
        if [[ -z "${!var:-}" ]]; then
            if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
                echo "ERROR: ${var} env var required for unattended integrations setup" >&2
                exit 1
            fi
            source "$(dirname "${BASH_SOURCE[0]}")/../ui.sh"
            ui_input "${var}:" "$var" secret
        fi
        export "${var?}"
    done
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "integrations: done"
}
