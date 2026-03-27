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
