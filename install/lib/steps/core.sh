#!/usr/bin/env bash
# core.sh — Python 3.11+, pip deps, git submodules
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

STEP_NAME="core"
SENTINEL="${HOME}/.myconex/.installed_core"

step_core() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[core] already installed, skipping"; return 0; }

    log_step "Installing core: Python, pip deps, submodules"

    # 1. Python 3.11+
    local py; py=$(detect_python || true)
    if [[ -z "$py" ]]; then
        local pkg; pkg=$(detect_pkg_manager)
        log_step "Installing Python 3.11"
        case "$pkg" in
            apt)    sudo apt-get install -y python3.11 python3.11-venv python3-pip ;;
            dnf)    sudo dnf install -y python3.11 ;;
            pacman) sudo pacman -Sy --noconfirm python ;;
            apk)    sudo apk add python3 py3-pip ;;
        esac
        py=$(detect_python)
    fi
    log_step "Python: $py"

    # 2. pip deps from project root requirements.txt
    local req="${MYCONEX_REPO_ROOT:?}/requirements.txt"
    if [[ "${REINSTALL:-}" ]]; then
        "$py" -m pip install --upgrade -r "$req"
    else
        "$py" -m pip install -r "$req"
    fi

    # 3. Git submodules (hermes-agent, flash-moe)
    git -C "$MYCONEX_REPO_ROOT" submodule update --init --recursive

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "core: done"
}
