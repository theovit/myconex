#!/usr/bin/env bash
# gpu.sh — NVIDIA Container Toolkit (skipped if no NVIDIA GPU)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

SENTINEL="${HOME}/.myconex/.installed_gpu"

step_gpu() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[gpu] already installed"; return 0; }

    # Auto-skip if no NVIDIA GPU
    if ! command -v nvidia-smi &>/dev/null; then
        log_step "No NVIDIA GPU detected — skipping GPU setup"
        touch "$SENTINEL"
        return 0
    fi

    log_step "Installing NVIDIA Container Toolkit"
    local pkg; pkg=$(detect_pkg_manager)
    case "$pkg" in
        apt)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update -q
            sudo apt-get install -y nvidia-container-toolkit
            ;;
        dnf)
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            sudo dnf install -y nvidia-container-toolkit
            ;;
        *)
            log_step "WARNING: cannot auto-install NVIDIA toolkit for package manager: $pkg"
            log_step "Install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            return 0
            ;;
    esac

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "gpu: done"
}
