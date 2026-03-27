#!/bin/sh
# MYCONEX Ollama startup — detects hardware tier and pulls the best model.
# OLLAMA_MODEL env var overrides auto-detection.
set -e

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
echo "[ollama-init] Waiting for Ollama server..."
until ollama list >/dev/null 2>&1; do
    sleep 2
done
echo "[ollama-init] Ollama server ready."

# Determine model
if [ -n "$OLLAMA_MODEL" ]; then
    MODEL="$OLLAMA_MODEL"
    echo "[ollama-init] Using configured model: $MODEL"
else
    # Detect VRAM via nvidia-smi (available via device passthrough)
    VRAM_MB=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' \r')
    fi
    VRAM_MB=${VRAM_MB:-0}

    # Detect RAM (Linux /proc/meminfo)
    RAM_GB=0
    if [ -f /proc/meminfo ]; then
        RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    fi

    # CPU core count
    CPU_CORES=0
    if command -v nproc >/dev/null 2>&1; then
        CPU_CORES=$(nproc)
    fi

    # Tier classification matching hardware.py logic:
    #   T1: VRAM >= 24576 MB (24 GB)
    #   T2: VRAM >= 7168  MB (7 GB, maps to 8 GB nominal)
    #   T3: CPU >= 16 cores OR RAM >= 16 GB
    #   T4: everything else
    if [ "$VRAM_MB" -ge 24576 ] 2>/dev/null; then
        MODEL="llama3.1:70b"
        TIER="T1"
    elif [ "$VRAM_MB" -ge 7168 ] 2>/dev/null; then
        MODEL="llama3.1:8b"
        TIER="T2"
    elif [ "$CPU_CORES" -ge 16 ] || [ "$RAM_GB" -ge 16 ]; then
        MODEL="llama3.2:3b"
        TIER="T3"
    else
        MODEL="phi3:mini"
        TIER="T4"
    fi

    echo "[ollama-init] Hardware: VRAM=${VRAM_MB}MB RAM=${RAM_GB}GB CPU=${CPU_CORES} → ${TIER} → ${MODEL}"
fi

# Pull model only if not already present
if ollama list 2>/dev/null | grep -q "^${MODEL}[: ]"; then
    echo "[ollama-init] Model ${MODEL} already present, skipping pull."
else
    echo "[ollama-init] Pulling ${MODEL} ..."
    ollama pull "$MODEL"
    echo "[ollama-init] Pull complete."
fi

# Hand off to the server process
wait "$OLLAMA_PID"
