#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_myconex.sh — Launch MYCONEX in autonomous mode with nohup
#
# Usage:
#   ./run_myconex.sh              # autonomous mode (default)
#   ./run_myconex.sh --mode cli   # interactive REPL
#   ./run_myconex.sh --mode api   # REST API server
#   ./run_myconex.sh --stop       # stop a running background instance
#   ./run_myconex.sh --status     # show running instance status
#   ./run_myconex.sh --logs       # tail the log file
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYCONEX_DIR="${HOME}/.myconex"
LOG_FILE="${MYCONEX_DIR}/myconex.log"
PID_FILE="${MYCONEX_DIR}/myconex.pid"
CONFIG_FILE="${SCRIPT_DIR}/config/mesh_config.yaml"
MODE="autonomous"
INTERVAL="5.0"
EXTRA_ARGS=()

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop)
            if [[ -f "${PID_FILE}" ]]; then
                PID=$(cat "${PID_FILE}")
                if kill -0 "${PID}" 2>/dev/null; then
                    echo "Stopping MYCONEX (PID ${PID})…"
                    kill -TERM "${PID}"
                    sleep 2
                    if kill -0 "${PID}" 2>/dev/null; then
                        kill -KILL "${PID}"
                        echo "Force-killed PID ${PID}."
                    else
                        echo "MYCONEX stopped."
                    fi
                    rm -f "${PID_FILE}"
                else
                    echo "PID ${PID} is not running. Removing stale PID file."
                    rm -f "${PID_FILE}"
                fi
            else
                echo "No PID file found at ${PID_FILE}. MYCONEX may not be running."
            fi
            exit 0
            ;;
        --status)
            if [[ -f "${PID_FILE}" ]]; then
                PID=$(cat "${PID_FILE}")
                if kill -0 "${PID}" 2>/dev/null; then
                    echo "MYCONEX is running (PID ${PID})."
                    echo "Log: ${LOG_FILE}"
                    echo "Tail: tail -f ${LOG_FILE}"
                else
                    echo "MYCONEX is NOT running (stale PID ${PID})."
                    rm -f "${PID_FILE}"
                fi
            else
                echo "MYCONEX is NOT running (no PID file)."
            fi
            exit 0
            ;;
        --logs)
            if [[ -f "${LOG_FILE}" ]]; then
                tail -f "${LOG_FILE}"
            else
                echo "No log file at ${LOG_FILE}"
            fi
            exit 0
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --foreground|-f)
            FOREGROUND=1
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Environment setup ─────────────────────────────────────────────────────────
mkdir -p "${MYCONEX_DIR}"

# Load .env if present
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Auto-detect Python executable
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

# Auto-activate virtualenv if present
VENV_DIRS=("${SCRIPT_DIR}/.venv" "${SCRIPT_DIR}/venv" "${HOME}/.venvs/myconex")
for VENV in "${VENV_DIRS[@]}"; do
    if [[ -f "${VENV}/bin/activate" ]]; then
        echo "Activating virtualenv: ${VENV}"
        # shellcheck disable=SC1091
        source "${VENV}/bin/activate"
        PYTHON="${VENV}/bin/python"
        break
    fi
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    EXISTING_PID=$(cat "${PID_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "MYCONEX is already running (PID ${EXISTING_PID})."
        echo "Use --stop to stop it, or --status for details."
        exit 1
    else
        echo "Removing stale PID file."
        rm -f "${PID_FILE}"
    fi
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    "${PYTHON}" "-m" "myconex"
    "--mode" "${MODE}"
    "--interval" "${INTERVAL}"
)

if [[ -f "${CONFIG_FILE}" ]]; then
    CMD+=("--config" "${CONFIG_FILE}")
fi

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ── Launch ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
    echo "Starting MYCONEX in foreground (mode=${MODE})…"
    exec "${CMD[@]}"
else
    echo "Starting MYCONEX in background (mode=${MODE})…"
    echo "Log: ${LOG_FILE}"
    echo "PID file: ${PID_FILE}"
    echo ""

    nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
    BGPID=$!

    echo "${BGPID}" > "${PID_FILE}"
    echo "MYCONEX started with PID ${BGPID}."
    echo ""
    echo "Commands:"
    echo "  ${BASH_SOURCE[0]} --status   # check status"
    echo "  ${BASH_SOURCE[0]} --logs     # tail logs"
    echo "  ${BASH_SOURCE[0]} --stop     # stop"
fi
