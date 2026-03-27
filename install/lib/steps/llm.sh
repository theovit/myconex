#!/usr/bin/env bash
# llm.sh — Ollama install + model pull by tier
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../profiles.sh"

SENTINEL="${HOME}/.myconex/.installed_llm"

step_llm() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[llm] already installed"; return 0; }

    local tier="${MYCONEX_DETECTED_TIER:?tier not set}"
    local model="${MYCONEX_OLLAMA_MODEL:-$(model_for_tier "$tier")}"

    log_step "Installing LLM backend (Ollama, model: $model)"

    # Install Ollama if missing
    if ! command -v ollama &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Pull model (idempotent — ollama pull is a no-op if already present)
    log_step "Pulling $model (this may take a while)"
    ollama pull "$model"

    # Always pull embedding model
    ollama pull nomic-embed-text

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "llm: done"
}
