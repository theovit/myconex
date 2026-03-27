#!/usr/bin/env bash
# detect.sh — OS, hardware, and environment detection

# --- Package manager ---
# Calls each candidate with a no-op flag so exported bash function mocks
# (which return non-zero to signal "absent") override real binaries in PATH.
detect_pkg_manager() {
    if apt-get --version &>/dev/null; then echo "apt"
    elif dnf --version &>/dev/null;   then echo "dnf"
    elif pacman --version &>/dev/null; then echo "pacman"
    elif apk --version &>/dev/null;   then echo "apk"
    else echo "unknown"; return 1
    fi
}

# --- WSL detection ---
# Testable via MYCONEX_TEST_KERNEL override
detect_wsl() {
    local kernel="${MYCONEX_TEST_KERNEL:-$(uname -r)}"
    [[ "$kernel" == *microsoft* ]]
}

# --- Display / UI mode ---
detect_display_mode() {
    # --no-tui and --unattended are handled before this is called
    if [[ -n "${SSH_CONNECTION:-}" ]] || [[ -z "${DISPLAY:-}" ]]; then
        echo "plain"
    elif command -v whiptail &>/dev/null; then
        echo "tui"
    else
        echo "plain"
    fi
}

# --- Python version check ---
detect_python() {
    local py
    for py in python3.11 python3.12 python3; do
        if command -v "$py" &>/dev/null; then
            local ver; ver=$("$py" -c "import sys; print(sys.version_info[:2])")
            if [[ "$ver" > "(3, 10)" ]]; then echo "$py"; return 0; fi
        fi
    done
    return 1
}

# --- Docker check ---
detect_docker() {
    command -v docker &>/dev/null && docker info &>/dev/null
}

# --- GPU VRAM (MB) via nvidia-smi ---
_detect_gpu_vram_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]'
}

# --- CPU core count (testable override via MYCONEX_TEST_CPU_CORES) ---
_detect_cpu_cores() {
    echo "${MYCONEX_TEST_CPU_CORES:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"
}

# --- RAM in GB (testable override via MYCONEX_TEST_RAM_GB) ---
_detect_ram_gb() {
    if [[ -n "${MYCONEX_TEST_RAM_GB:-}" ]]; then echo "$MYCONEX_TEST_RAM_GB"; return; fi
    local kb; kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    echo $(( ${kb:-0} / 1024 / 1024 ))
}

# --- Tier classification ---
detect_tier() {
    local vram_mb; vram_mb=$(_detect_gpu_vram_mb)
    if [[ -n "$vram_mb" && "$vram_mb" -gt 0 ]]; then
        if   [[ "$vram_mb" -ge 24000 ]]; then echo "T1"
        elif [[ "$vram_mb" -ge 7000  ]]; then echo "T2"
        else                                  echo "T3"
        fi
        return
    fi
    # CPU-only path
    local cores ram
    cores=$(_detect_cpu_cores)
    ram=$(_detect_ram_gb)
    if [[ "$cores" -ge 8 && "$ram" -ge 16 ]]; then echo "T3"
    else                                           echo "T4"
    fi
}
