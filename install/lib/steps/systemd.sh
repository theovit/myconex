#!/usr/bin/env bash
# systemd.sh — install and enable systemd units (or start.sh fallback)
set -euo pipefail

SENTINEL="${HOME}/.myconex/.installed_systemd"
UNIT_DIR="/etc/systemd/system"

step_systemd() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing service units"

    local role="${MYCONEX_ROLE:?}"

    if systemctl --version &>/dev/null && [[ -d "$UNIT_DIR" ]]; then
        _install_systemd_unit "$role"
    else
        log_step "systemd not available — installing start.sh fallback"
        _install_start_sh "$role"
    fi

    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "systemd: done"
}

_install_systemd_unit() {
    local role="$1"
    case "$role" in
        hub)
            sudo tee "${UNIT_DIR}/myconex-hub.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Hub Services
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}/services
ExecStart=/usr/bin/docker compose --profile full up
ExecStop=/usr/bin/docker compose --profile full down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-hub.service
            ;;
        full-node)
            local py; py=$(command -v python3.11 || command -v python3)
            sudo tee "${UNIT_DIR}/myconex-node.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Mesh Node
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}
ExecStart=${py} -m myconex --mode worker
Restart=on-failure
RestartSec=10
Environment="MYCONEX_CONFIG=/etc/myconex/mesh_config.yaml"

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-node.service
            ;;
        lightweight)
            local py; py=$(command -v python3.11 || command -v python3)
            sudo tee "${UNIT_DIR}/myconex-registration.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Mesh Registration Agent
After=network-online.target ollama.service

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}
ExecStart=${py} -m myconex --mode registration
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-registration.service
            ;;
    esac
}

_install_start_sh() {
    local role="$1"
    local start="${HOME}/.myconex/start.sh"
    mkdir -p "$(dirname "$start")"
    case "$role" in
        hub)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}/services"
exec docker compose --profile full up
SH
            ;;
        full-node)
            local py; py=$(command -v python3.11 || command -v python3)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}"
exec ${py} -m myconex --mode worker
SH
            ;;
        lightweight)
            local py; py=$(command -v python3.11 || command -v python3)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}"
exec ${py} -m myconex --mode registration
SH
            ;;
    esac
    chmod +x "$start"
    log_step "Start with: $start"
}
