#!/usr/bin/env bash
# MYCONEX Installer — canonical entry point
# Usage: ./install.sh [--role hub|full-node|lightweight] [--unattended answers.yaml]
#        [--save-answers] [--answers-out path] [--no-tui] [--reinstall]
#        [--skip-verify] [--log path]
set -euo pipefail

MYCONEX_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MYCONEX_REPO_ROOT

MYCONEX_LOG="${HOME}/.myconex/install.log"
MYCONEX_ROLE=""
MYCONEX_ANSWERS_FILE=""
MYCONEX_SAVE_ANSWERS=""
MYCONEX_ANSWERS_OUT="./myconex-answers.yaml"
REINSTALL=""
SKIP_VERIFY=""
PARSE_ONLY=""  # test hook: source + parse flags without executing
export REINSTALL

# ── Parse flags ──────────────────────────────────────────────────────────────
_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --role)         MYCONEX_ROLE="$2";         shift 2 ;;
            --unattended)   MYCONEX_ANSWERS_FILE="$2"; shift 2 ;;
            --save-answers) MYCONEX_SAVE_ANSWERS=1;    shift ;;
            --answers-out)  MYCONEX_ANSWERS_OUT="$2";  shift 2 ;;
            --no-tui)       UI_MODE="plain";            shift ;;
            --reinstall)    REINSTALL=1;                shift ;;
            --skip-verify)  SKIP_VERIFY=1;             shift ;;
            --log)          MYCONEX_LOG="$2";          shift 2 ;;
            --parse-only)   PARSE_ONLY=1;              shift ;;
            --help|-h)
                echo "Usage: install.sh [--role hub|full-node|lightweight]"
                echo "       [--unattended answers.yaml] [--save-answers] [--answers-out path]"
                echo "       [--no-tui] [--reinstall] [--skip-verify] [--log path]"
                exit 0 ;;
            *) echo "Unknown flag: $1" >&2; exit 1 ;;
        esac
    done
    export MYCONEX_ROLE UI_MODE MYCONEX_ANSWERS_FILE MYCONEX_ANSWERS_OUT
}

# ── Logging ───────────────────────────────────────────────────────────────────
log_step() { echo "[install] $*" | tee -a "${MYCONEX_LOG}"; }
export -f log_step

# ── Load libraries ────────────────────────────────────────────────────────────
_load_libs() {
    local lib="${MYCONEX_REPO_ROOT}/install/lib"
    source "${lib}/detect.sh"
    source "${lib}/ui.sh"
    source "${lib}/profiles.sh"
    for step in "${lib}/steps/"*.sh; do source "$step"; done
}

# ── Answer file ───────────────────────────────────────────────────────────────
_load_answers() {
    local file="$1"
    [[ -f "$file" ]] || { echo "ERROR: answer file not found: $file" >&2; exit 1; }
    # Requires yq; parse key→env var
    command -v yq &>/dev/null || { echo "ERROR: yq required for --unattended" >&2; exit 1; }
    MYCONEX_ROLE=$(yq e '.role' "$file")
    MYCONEX_NODE_NAME=$(yq e '.node_name // ""' "$file")
    MYCONEX_HUB_ADDRESS=$(yq e '.hub_address // ""' "$file")
    FEAT_DISCORD=$(yq e '.features.discord // false' "$file")
    FEAT_INTEGRATIONS=$(yq e '.features.integrations // false' "$file")
    FEAT_DASHBOARD=$(yq e '.features.dashboard // false' "$file")
    MYCONEX_OLLAMA_MODEL=$(yq e '.ollama_model // ""' "$file")
    export MYCONEX_ROLE MYCONEX_NODE_NAME MYCONEX_HUB_ADDRESS
    export FEAT_DISCORD FEAT_INTEGRATIONS FEAT_DASHBOARD MYCONEX_OLLAMA_MODEL
}

_save_answers() {
    local out="$1"
    cat > "$out" <<YAML
role: ${MYCONEX_ROLE}
node_name: "${MYCONEX_NODE_NAME:-}"
hub_address: "${MYCONEX_HUB_ADDRESS:-}"

features:
  discord: ${FEAT_DISCORD:-false}
  integrations: ${FEAT_INTEGRATIONS:-false}
  dashboard: ${FEAT_DASHBOARD:-false}

api_keys:
  discord_bot_token: ""
  openrouter_api_key: ""
  nous_api_key: ""

ollama_model: "${MYCONEX_OLLAMA_MODEL:-}"
YAML
    echo "Answers saved to: $out"
}

# ── Verification ──────────────────────────────────────────────────────────────
_verify() {
    local role="$1"
    log_step "Running verification checks"
    local ok=1
    _check_http() {
        local name="$1" url="$2"
        if curl -sf "$url" &>/dev/null; then
            printf "  \033[32m✓\033[0m  %-12s %s\n" "$name" "$url"
        else
            printf "  \033[31m✗\033[0m  %-12s %s  (FAILED)\n" "$name" "$url"
            ok=0
        fi
    }
    _check_redis() {
        if redis-cli -h localhost ping 2>/dev/null | grep -q PONG; then
            printf "  \033[32m✓\033[0m  %-12s redis://localhost:6379\n" "Redis"
        else
            printf "  \033[31m✗\033[0m  %-12s redis://localhost:6379  (FAILED)\n" "Redis"
            ok=0
        fi
    }
    if [[ "$role" == "hub" ]]; then
        _check_http "NATS"    "http://localhost:8222/healthz"
        _check_redis
        _check_http "Qdrant"  "http://localhost:6333/healthz"
        _check_http "Ollama"  "http://localhost:11434/api/tags"
        _check_http "LiteLLM" "http://localhost:4000/health/liveliness"
        _check_http "API"     "http://localhost:8765/health"
    fi
    [[ "$ok" == 1 ]] && log_step "All checks passed" || log_step "Some checks failed — see log"
}

# ── Main flow ─────────────────────────────────────────────────────────────────
main() {
    _parse_flags "$@"
    [[ -n "$PARSE_ONLY" ]] && return 0

    mkdir -p "$(dirname "$MYCONEX_LOG")"
    _load_libs

    # Detect environment
    MYCONEX_DETECTED_TIER=$(detect_tier)
    local display_mode; display_mode=$(detect_display_mode)
    UI_MODE="${UI_MODE:-$display_mode}"
    export MYCONEX_DETECTED_TIER UI_MODE

    # Load answers or run interactive flow
    if [[ -n "$MYCONEX_ANSWERS_FILE" ]]; then
        UI_MODE="unattended"
        _load_answers "$MYCONEX_ANSWERS_FILE"
    else
        # Role selection
        if [[ -z "$MYCONEX_ROLE" ]]; then
            ui_menu "What is this machine?" "hub" "full-node" "lightweight"
            MYCONEX_ROLE="$REPLY"
            export MYCONEX_ROLE
        fi

        # Feature selection
        FEAT_DISCORD=false; FEAT_INTEGRATIONS=false; FEAT_DASHBOARD=false
        if [[ "$MYCONEX_ROLE" != "lightweight" ]] && [[ "$MYCONEX_ROLE" != "full-node" || "$MYCONEX_DETECTED_TIER" != "T4" ]]; then
            ui_checklist "Optional features:" \
                "discord:off" "integrations:off" "dashboard:off"
            for sel in "${SELECTED[@]}"; do
                case "$sel" in
                    discord)      FEAT_DISCORD=true ;;
                    integrations) FEAT_INTEGRATIONS=true ;;
                    dashboard)    FEAT_DASHBOARD=true ;;
                esac
            done
        fi
        export FEAT_DISCORD FEAT_INTEGRATIONS FEAT_DASHBOARD

        # Configure
        ui_input "Node name [$(hostname)]:" MYCONEX_NODE_NAME
        MYCONEX_NODE_NAME="${MYCONEX_NODE_NAME:-$(hostname)}"
        export MYCONEX_NODE_NAME
        if [[ "$MYCONEX_ROLE" != "hub" ]]; then
            ui_input "Hub address (blank = mDNS auto-discover):" MYCONEX_HUB_ADDRESS
            export MYCONEX_HUB_ADDRESS
        fi
    fi

    # Save answers if requested (no install)
    if [[ -n "$MYCONEX_SAVE_ANSWERS" ]]; then
        _save_answers "$MYCONEX_ANSWERS_OUT"
        return 0
    fi

    # Plan preview
    log_step "Role: $MYCONEX_ROLE | Tier: $MYCONEX_DETECTED_TIER | UI: $UI_MODE"
    ui_confirm "Proceed with installation?" || { echo "Aborted."; exit 0; }

    # Execute steps
    step_core
    if [[ "$MYCONEX_ROLE" == "hub" ]]; then
        step_hub_services
        step_gpu
        step_llm
        step_hermes
    elif [[ "$MYCONEX_ROLE" == "full-node" ]]; then
        step_llm
        profile_requires_for_tier "full-node" "$MYCONEX_DETECTED_TIER" "gpu"        && step_gpu
        profile_requires_for_tier "full-node" "$MYCONEX_DETECTED_TIER" "hermes_moe" && step_hermes
    else  # lightweight
        step_registration_agent
        step_llm
    fi
    [[ "$FEAT_DISCORD"      == "true" ]] && step_discord
    [[ "$FEAT_INTEGRATIONS" == "true" ]] && step_integrations
    [[ "$FEAT_DASHBOARD"    == "true" ]] && step_dashboard
    step_config
    step_systemd

    # Verify
    [[ -z "$SKIP_VERIFY" ]] && _verify "$MYCONEX_ROLE"

    log_step "Installation complete. Node: ${MYCONEX_NODE_NAME} | Tier: ${MYCONEX_DETECTED_TIER}"
}

main "$@"
