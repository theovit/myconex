#!/usr/bin/env bash
# MYCONEX SPORE Installer
# Auto-onboards Linux machines into the AI mesh network
# Usage: curl -sSL https://mesh.local/spore | bash
# or: bash install.sh [--node-name NAME] [--mesh-host HOST] [--headless]

set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[SPORE]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()  { echo -e "${RED}[ERR]${RESET}   $*" >&2; }
banner() {
  echo -e "${CYAN}${BOLD}"
  cat <<'EOF'
  ███╗   ███╗██╗   ██╗ ██████╗ ██████╗ ███╗   ██╗███████╗██╗  ██╗
  ████╗ ████║╚██╗ ██╔╝██╔════╝██╔═══██╗████╗  ██║██╔════╝╚██╗██╔╝
  ██╔████╔██║ ╚████╔╝ ██║     ██║   ██║██╔██╗ ██║█████╗   ╚███╔╝
  ██║╚██╔╝██║  ╚██╔╝  ██║     ██║   ██║██║╚██╗██║██╔══╝   ██╔██╗
  ██║ ╚═╝ ██║   ██║   ╚██████╗╚██████╔╝██║ ╚████║███████╗██╔╝ ██╗
  ╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
  SPORE v0.1.0 — Distributed AI Mesh Node Installer
EOF
  echo -e "${RESET}"
}

# ─── Defaults ─────────────────────────────────────────────────────────────────
NODE_NAME="${HOSTNAME:-myconex-node}"
MESH_HOST=""
HEADLESS=false
INSTALL_DIR="/opt/myconex"
DATA_DIR="/var/lib/myconex"
CONFIG_DIR="/etc/myconex"
LOG_DIR="/var/log/myconex"
PYTHON_MIN="3.10"
DOCKER_MIN="24.0"

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-name)  NODE_NAME="$2"; shift 2 ;;
    --mesh-host)  MESH_HOST="$2"; shift 2 ;;
    --headless)   HEADLESS=true; shift ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--node-name NAME] [--mesh-host HOST] [--headless]"
      exit 0 ;;
    *) err "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Root check ───────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  err "Run as root or with sudo."
  exit 1
fi

banner

# ─── OS Detection ─────────────────────────────────────────────────────────────
detect_os() {
  if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS_ID="${ID}"
    OS_VERSION="${VERSION_ID}"
    OS_LIKE="${ID_LIKE:-}"
  else
    err "Cannot detect OS. /etc/os-release not found."
    exit 1
  fi

  log "Detected OS: ${PRETTY_NAME}"

  case "$OS_ID" in
    ubuntu|debian|raspbian) PKG_MGR="apt-get" ;;
    fedora|rhel|centos|rocky|alma) PKG_MGR="dnf" ;;
    arch|manjaro) PKG_MGR="pacman" ;;
    *)
      if echo "$OS_LIKE" | grep -q "debian"; then PKG_MGR="apt-get"
      elif echo "$OS_LIKE" | grep -q "rhel\|fedora"; then PKG_MGR="dnf"
      else warn "Unsupported OS, attempting apt-get fallback"; PKG_MGR="apt-get"
      fi ;;
  esac
}

# ─── Hardware Detection ───────────────────────────────────────────────────────
detect_hardware() {
  log "Detecting hardware..."

  # CPU
  CPU_CORES=$(nproc)
  CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs || echo "Unknown")
  CPU_ARCH=$(uname -m)

  # RAM (in GB)
  TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  TOTAL_RAM_GB=$(echo "scale=1; $TOTAL_RAM_KB / 1048576" | bc)

  # GPU Detection
  GPU_NAME="None"
  VRAM_GB=0
  HAS_NVIDIA=false
  HAS_AMD=false
  HAS_APPLE=false

  if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    VRAM_GB=$(echo "scale=1; ${VRAM_MB:-0} / 1024" | bc)
  elif lspci 2>/dev/null | grep -qi "amd\|radeon"; then
    HAS_AMD=true
    GPU_NAME=$(lspci | grep -i "vga\|3d\|display" | grep -i "amd\|radeon" | head -1 | sed 's/.*: //' || echo "AMD GPU")
    # rocm-smi for VRAM if available
    if command -v rocm-smi &>/dev/null; then
      VRAM_MB=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | grep -v "Device" | awk -F, '{print $2}' | head -1 | xargs || echo "0")
      VRAM_GB=$(echo "scale=1; ${VRAM_MB:-0} / 1048576" | bc)
    fi
  fi

  # Detect Raspberry Pi
  IS_RPI=false
  if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_RPI=true
    RPI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    log "Raspberry Pi detected: $RPI_MODEL"
  fi

  # Storage
  DISK_AVAIL_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')

  log "CPU: ${CPU_MODEL} (${CPU_CORES} cores, ${CPU_ARCH})"
  log "RAM: ${TOTAL_RAM_GB} GB"
  log "GPU: ${GPU_NAME} (VRAM: ${VRAM_GB} GB)"
  log "Disk: ${DISK_AVAIL_GB} GB available"
}

# ─── Node Tier Classification ─────────────────────────────────────────────────
classify_tier() {
  log "Classifying node tier..."

  # Convert to integer for comparisons
  VRAM_INT=$(echo "$VRAM_GB" | cut -d. -f1)
  RAM_INT=$(echo "$TOTAL_RAM_GB" | cut -d. -f1)

  if [[ "$IS_RPI" == "true" ]] || [[ $RAM_INT -lt 8 ]]; then
    NODE_TIER="T4"
    TIER_LABEL="Edge/Embedded"
    TIER_ROLE="sensor,relay,lightweight-inference"
  elif [[ $VRAM_INT -ge 24 ]]; then
    NODE_TIER="T1"
    TIER_LABEL="Apex (Heavy GPU)"
    TIER_ROLE="large-model,training,heavy-inference,coordinator"
  elif [[ $VRAM_INT -ge 8 ]]; then
    NODE_TIER="T2"
    TIER_LABEL="Mid-GPU"
    TIER_ROLE="medium-model,inference,embedding"
  elif [[ $CPU_CORES -ge 16 ]] && [[ $RAM_INT -ge 16 ]]; then
    NODE_TIER="T3"
    TIER_LABEL="CPU-Heavy"
    TIER_ROLE="orchestration,embedding,lightweight-inference"
  elif [[ $RAM_INT -ge 16 ]]; then
    NODE_TIER="T3"
    TIER_LABEL="CPU-Heavy"
    TIER_ROLE="orchestration,embedding,lightweight-inference"
  else
    NODE_TIER="T4"
    TIER_LABEL="Edge"
    TIER_ROLE="relay,sensor"
  fi

  log "Node tier: ${NODE_TIER} — ${TIER_LABEL}"
  log "Assigned roles: ${TIER_ROLE}"
}

# ─── Dependency Installation ──────────────────────────────────────────────────
install_base_deps() {
  log "Installing base dependencies..."

  case "$PKG_MGR" in
    apt-get)
      apt-get update -qq
      apt-get install -y -qq \
        curl wget git jq bc avahi-daemon avahi-utils \
        python3 python3-pip python3-venv python3-dev \
        build-essential libssl-dev libffi-dev \
        net-tools iproute2 lsof
      ;;
    dnf)
      dnf install -y -q \
        curl wget git jq bc avahi avahi-tools \
        python3 python3-pip python3-devel \
        gcc openssl-devel libffi-devel \
        net-tools iproute lsof
      ;;
    pacman)
      pacman -Sy --noconfirm \
        curl wget git jq bc avahi python python-pip \
        base-devel openssl net-tools iproute2 lsof
      ;;
  esac
}

install_docker() {
  if command -v docker &>/dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    log "Docker already installed: ${DOCKER_VERSION}"
    return 0
  fi

  log "Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
  usermod -aG docker "${SUDO_USER:-root}" 2>/dev/null || true
  log "Docker installed successfully."
}

install_ollama() {
  if command -v ollama &>/dev/null; then
    log "Ollama already installed."
    return 0
  fi

  log "Installing Ollama..."
  curl -fsSL https://ollama.ai/install.sh | sh
  systemctl enable --now ollama 2>/dev/null || true
  log "Ollama installed."

  # Pull default model based on tier
  case "$NODE_TIER" in
    T1) OLLAMA_MODEL="llama3.1:70b" ;;
    T2) OLLAMA_MODEL="llama3.1:8b" ;;
    T3) OLLAMA_MODEL="llama3.2:3b" ;;
    T4) OLLAMA_MODEL="phi3:mini" ;;
  esac

  log "Pulling default model for ${NODE_TIER}: ${OLLAMA_MODEL}"
  ollama pull "$OLLAMA_MODEL" &
  log "Model pull started in background (PID $!)"
}

install_python_deps() {
  log "Setting up Python virtual environment..."
  python3 -m venv "${INSTALL_DIR}/venv"
  source "${INSTALL_DIR}/venv/bin/activate"

  pip install --quiet --upgrade pip
  pip install --quiet \
    nats-py \
    zeroconf \
    redis \
    pyyaml \
    fastapi \
    uvicorn \
    httpx \
    psutil \
    gputil \
    loguru \
    click \
    rich \
    asyncio-mqtt

  log "Python dependencies installed."
}

# ─── Avahi / mDNS Setup ───────────────────────────────────────────────────────
configure_mdns() {
  log "Configuring mDNS (Avahi)..."

  systemctl enable --now avahi-daemon 2>/dev/null || true

  # Create service advertisement
  mkdir -p /etc/avahi/services
  cat > /etc/avahi/services/myconex.service <<EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">MYCONEX-${NODE_NAME}</name>
  <service>
    <type>_ai-mesh._tcp</type>
    <port>8765</port>
    <txt-record>tier=${NODE_TIER}</txt-record>
    <txt-record>roles=${TIER_ROLE}</txt-record>
    <txt-record>version=0.1.0</txt-record>
    <txt-record>node=${NODE_NAME}</txt-record>
  </service>
</service-group>
EOF

  systemctl restart avahi-daemon 2>/dev/null || true
  log "mDNS service registered: MYCONEX-${NODE_NAME}._ai-mesh._tcp"
}

# ─── MYCONEX Installation ─────────────────────────────────────────────────────
install_myconex() {
  log "Installing MYCONEX to ${INSTALL_DIR}..."

  mkdir -p "${INSTALL_DIR}" "${DATA_DIR}" "${CONFIG_DIR}" "${LOG_DIR}"

  # Write node config
  cat > "${CONFIG_DIR}/node.yaml" <<EOF
node:
  name: "${NODE_NAME}"
  tier: "${NODE_TIER}"
  tier_label: "${TIER_LABEL}"
  roles: [$(echo "$TIER_ROLE" | tr ',' '\n' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')]

hardware:
  cpu_cores: ${CPU_CORES}
  cpu_arch: "${CPU_ARCH}"
  ram_gb: ${TOTAL_RAM_GB}
  gpu_name: "${GPU_NAME}"
  vram_gb: ${VRAM_GB}
  has_nvidia: ${HAS_NVIDIA}
  has_amd: ${HAS_AMD}
  is_rpi: ${IS_RPI}

mesh:
  mdns_service: "_ai-mesh._tcp"
  api_port: 8765
  nats_port: 4222
  mesh_host: "${MESH_HOST}"

ollama:
  host: "localhost"
  port: 11434
  default_model: "${OLLAMA_MODEL:-phi3:mini}"

created_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

  log "Node config written to ${CONFIG_DIR}/node.yaml"

  # Create systemd service
  cat > /etc/systemd/system/myconex.service <<EOF
[Unit]
Description=MYCONEX AI Mesh Node
After=network.target avahi-daemon.service docker.service
Wants=avahi-daemon.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/main.py --config ${CONFIG_DIR}/node.yaml
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_DIR}/myconex.log
StandardError=append:${LOG_DIR}/myconex-error.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  log "Systemd service registered: myconex.service"
}

# ─── Peer Discovery ───────────────────────────────────────────────────────────
discover_peers() {
  log "Scanning for existing mesh peers..."

  if command -v avahi-browse &>/dev/null; then
    PEERS=$(avahi-browse -t -r _ai-mesh._tcp 2>/dev/null | grep "hostname\|address\|port" | head -30 || echo "")
    if [[ -n "$PEERS" ]]; then
      log "Found mesh peers:"
      echo "$PEERS" | while read -r line; do
        echo "  $line"
      done
    else
      log "No existing peers found. This node will initialize as mesh seed."
    fi
  fi
}

# ─── Docker Compose Launch ────────────────────────────────────────────────────
launch_services() {
  if [[ "$NODE_TIER" == "T4" ]]; then
    log "T4 node: skipping Docker service stack (resource constrained)."
    return 0
  fi

  if [[ -f "${INSTALL_DIR}/services/docker-compose.yml" ]]; then
    log "Starting MYCONEX service stack..."
    cd "${INSTALL_DIR}/services"
    docker compose up -d
    log "Services started."
  else
    warn "docker-compose.yml not found. Skipping service stack."
  fi
}

# ─── Summary ──────────────────────────────────────────────────────────────────
print_summary() {
  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║        SPORE Installation Complete           ║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${RESET}"
  echo ""
  echo -e "  Node Name  : ${BOLD}${NODE_NAME}${RESET}"
  echo -e "  Tier       : ${BOLD}${NODE_TIER} — ${TIER_LABEL}${RESET}"
  echo -e "  Roles      : ${TIER_ROLE}"
  echo -e "  Config     : ${CONFIG_DIR}/node.yaml"
  echo -e "  Logs       : ${LOG_DIR}/"
  echo -e "  mDNS       : MYCONEX-${NODE_NAME}._ai-mesh._tcp"
  echo ""
  echo -e "  Start      : ${BOLD}systemctl start myconex${RESET}"
  echo -e "  Status     : ${BOLD}systemctl status myconex${RESET}"
  echo -e "  Logs       : ${BOLD}journalctl -u myconex -f${RESET}"
  echo ""
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  detect_os
  detect_hardware
  classify_tier
  install_base_deps
  install_docker
  install_ollama
  mkdir -p "${INSTALL_DIR}"
  install_python_deps
  configure_mdns
  install_myconex
  discover_peers
  launch_services
  print_summary
}

main "$@"
