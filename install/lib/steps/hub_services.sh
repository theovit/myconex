#!/usr/bin/env bash
# hub_services.sh — Docker install + docker compose --profile full
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

SENTINEL="${HOME}/.myconex/.installed_hub_services"

step_hub_services() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[hub_services] already installed"; return 0; }

    log_step "Installing hub services (Docker + compose stack)"

    # Install Docker if missing
    if ! detect_docker; then
        log_step "Installing Docker"
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        log_step "Docker installed. You may need to log out and back in for group changes."
    fi

    # On reinstall: pull latest images
    if [[ -f "$SENTINEL" ]]; then
        log_step "Pulling latest images"
        docker compose -f "${MYCONEX_REPO_ROOT}/services/docker-compose.yml" \
            --profile full pull
    fi

    log_step "Starting hub services"
    docker compose -f "${MYCONEX_REPO_ROOT}/services/docker-compose.yml" \
        --profile full up -d

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "hub_services: done"
}
