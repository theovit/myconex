#!/usr/bin/env bash
# MYCONEX Model Downloader
# Downloads lightweight models from config/lightweight_models.yaml
# Source: https://dev.to/jaipalsingh/15-best-lightweight-language-models-worth-running-in-2026-297g
#
# Usage:
#   ./scripts/download_models.sh                  # interactive menu
#   ./scripts/download_models.sh ollama all       # pull every model via Ollama
#   ./scripts/download_models.sh ollama t3        # pull T3-recommended models
#   ./scripts/download_models.sh hf <model-id>    # download one model via HuggingFace CLI
#   ./scripts/download_models.sh gguf <model-id>  # download GGUF via HuggingFace CLI
#
# Requirements:
#   Ollama pulls  — ollama must be running (systemctl start ollama)
#   HuggingFace   — pip install huggingface_hub  &&  huggingface-cli login

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GGUF_DIR="${GGUF_DIR:-$HOME/.myconex/models/gguf}"

# ─── Colour helpers ───────────────────────────────────────────────────────────
bold=$'\e[1m'; reset=$'\e[0m'; green=$'\e[32m'; yellow=$'\e[33m'; red=$'\e[31m'; cyan=$'\e[36m'
info()  { echo "${cyan}[INFO]${reset}  $*"; }
ok()    { echo "${green}[OK]${reset}    $*"; }
warn()  { echo "${yellow}[WARN]${reset}  $*"; }
die()   { echo "${red}[ERROR]${reset} $*" >&2; exit 1; }

# ─── Model catalogue (id|ollama_tag|hf_repo|gguf_repo|size_gb|min_ram_gb|description) ──
# Mirrors config/lightweight_models.yaml — update both if adding models.
declare -A OLLAMA_TAG HF_REPO GGUF_REPO SIZE_GB MIN_RAM DESC

_reg() {
    local id="$1"
    OLLAMA_TAG[$id]="$2"
    HF_REPO[$id]="$3"
    GGUF_REPO[$id]="$4"
    SIZE_GB[$id]="$5"
    MIN_RAM[$id]="$6"
    DESC[$id]="$7"
}

# id                    ollama_tag                  hf_repo                                          gguf_repo                                              size  ram  desc
_reg qwen3-0.6b         "qwen3:0.6b"               "Qwen/Qwen3-0.6B-Instruct"                       "Qwen/Qwen3-0.6B-Instruct-GGUF"                       0.4   2   "Qwen3 0.6B — 119 langs, 32K ctx, ultra-lightweight"
_reg tinyllama-1.1b     "tinyllama:1.1b"            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"             "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"               0.7   2   "TinyLlama 1.1B — 2K ctx, 3T tokens, fast"
_reg deepseek-r1-1.5b   "deepseek-r1:1.5b"         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"      "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"         1.1   3   "DeepSeek-R1 1.5B — chain-of-thought reasoning"
_reg qwen3-1.7b         "qwen3:1.7b"               "Qwen/Qwen3-1.7B-Instruct"                       "Qwen/Qwen3-1.7B-Instruct-GGUF"                       1.2   3   "Qwen3 1.7B — 119 langs, 32K ctx"
_reg phi4-mini          "phi4-mini:3.8b"            "microsoft/Phi-4-mini-instruct"                  "bartowski/Phi-4-mini-instruct-GGUF"                   2.5   4   "Phi-4 Mini 3.8B — 128K ctx, STEM, MIT"
_reg phi4-mini-reasoning "phi4-mini-reasoning:3.8b" "microsoft/Phi-4-mini-reasoning"                 "bartowski/Phi-4-mini-reasoning-GGUF"                  2.5   4   "Phi-4 Mini Reasoning 3.8B — RL reasoning, MIT"
_reg smollm3-3b         "smollm3:3b"               "HuggingFaceTB/SmolLM3-3B"                       "HuggingFaceTB/SmolLM3-3B-GGUF"                        2.0   4   "SmolLM3 3B — 11T tokens, Apache 2.0"
_reg stablelm-zephyr-3b "stablelm-zephyr:3b"        "stabilityai/stablelm-zephyr-3b"                 "TheBloke/stablelm-zephyr-3b-GGUF"                     2.0   4   "StableLM Zephyr 3B — DPO aligned, noncommercial"
_reg qwen3-4b           "qwen3:4b"                 "Qwen/Qwen3-4B-Instruct"                         "Qwen/Qwen3-4B-Instruct-GGUF"                          2.6   4   "Qwen3 4B — 119 langs, 32K ctx, recommended T3"
_reg gemma3-4b          "gemma3:4b"                "google/gemma-3-4b-it"                           "bartowski/gemma-3-4b-it-GGUF"                         2.6   6   "Gemma 3 4B — vision, 128K ctx, 35 langs"
_reg gemma3n-e2b        "gemma3n:e2b"              "google/gemma-3n-E2B-it"                         "bartowski/gemma-3n-E2B-it-GGUF"                       3.0   6   "Gemma 3n E2B — vision+audio multimodal"
_reg mistral-7b         "mistral:7b"               "mistralai/Mistral-7B-Instruct-v0.3"             "TheBloke/Mistral-7B-Instruct-v0.3-GGUF"               4.1   8   "Mistral 7B — GQA, 32K ctx, Apache 2.0"
_reg llama31-8b         "llama3.1:8b"              "meta-llama/Meta-Llama-3.1-8B-Instruct"          "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"            4.7   8   "Llama 3.1 8B — 128K ctx, tool-calling, T2 default"
_reg deepseek-r1-8b     "deepseek-r1:8b"           "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"       "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"          5.0   8   "DeepSeek-R1 8B — strong reasoning, MIT"
_reg qwen3-8b           "qwen3:8b"                 "Qwen/Qwen3-8B-Instruct"                         "Qwen/Qwen3-8B-Instruct-GGUF"                          5.2   8   "Qwen3 8B — 119 langs, 128K ctx, recommended T2"
_reg glm4-9b            "glm4:9b"                  "THUDM/glm-4-9b-chat-0414"                       "bartowski/glm-4-9b-chat-0414-GGUF"                    6.2  10   "GLM-4 9B — code, function-calling, SVG/HTML"

ALL_IDS=(
    qwen3-0.6b tinyllama-1.1b deepseek-r1-1.5b qwen3-1.7b
    phi4-mini phi4-mini-reasoning smollm3-3b stablelm-zephyr-3b
    qwen3-4b gemma3-4b gemma3n-e2b
    mistral-7b llama31-8b deepseek-r1-8b qwen3-8b glm4-9b
)

# Tier recommendations
T4_IDS=(qwen3-0.6b tinyllama-1.1b)
T3_IDS=(qwen3-4b phi4-mini smollm3-3b)
T2_IDS=(qwen3-8b llama31-8b mistral-7b)
REASONING_IDS=(phi4-mini-reasoning deepseek-r1-1.5b deepseek-r1-8b)
VISION_IDS=(gemma3-4b gemma3n-e2b)

# ─── Helpers ──────────────────────────────────────────────────────────────────

check_ollama() {
    command -v ollama &>/dev/null || die "ollama not found. Install from https://ollama.com"
    ollama list &>/dev/null || die "Ollama daemon not running. Start it with: ollama serve"
}

check_hf() {
    command -v huggingface-cli &>/dev/null || {
        warn "huggingface-cli not found."
        info "Install: pip install huggingface_hub"
        info "Login:   huggingface-cli login"
        exit 1
    }
}

ollama_pull() {
    local id="$1"
    local tag="${OLLAMA_TAG[$id]}"
    info "Pulling ${bold}${tag}${reset}  (${DESC[$id]})  ~${SIZE_GB[$id]}GB"
    ollama pull "$tag" && ok "$tag" || warn "Failed: $tag"
}

hf_download() {
    local repo="$1"
    local dest="${2:-}"
    info "Downloading ${bold}${repo}${reset}"
    if [[ -n "$dest" ]]; then
        huggingface-cli download "$repo" --local-dir "$dest"
    else
        huggingface-cli download "$repo"
    fi
    ok "$repo"
}

gguf_download() {
    local id="$1"
    local repo="${GGUF_REPO[$id]}"
    local dest="$GGUF_DIR/$id"
    mkdir -p "$dest"
    info "Downloading GGUF ${bold}${repo}${reset} → $dest"
    # Download only Q4_K_M quantization if available, else all GGUF files
    huggingface-cli download "$repo" --include "*.gguf" --local-dir "$dest" && ok "$repo" || warn "Failed: $repo"
}

print_catalogue() {
    echo
    echo "${bold}MYCONEX Lightweight Model Catalogue${reset}"
    echo "─────────────────────────────────────────────────────────────────"
    printf "%-24s %-8s %-6s  %s\n" "ID" "SIZE" "RAM" "DESCRIPTION"
    echo "─────────────────────────────────────────────────────────────────"
    for id in "${ALL_IDS[@]}"; do
        printf "%-24s %-8s %-6s  %s\n" \
            "$id" "${SIZE_GB[$id]}GB" "${MIN_RAM[$id]}GB" "${DESC[$id]}"
    done
    echo
}

interactive_menu() {
    print_catalogue
    echo "${bold}Quick groups:${reset}"
    echo "  t4        — T4 edge nodes (≤2GB RAM): ${T4_IDS[*]}"
    echo "  t3        — T3 CPU nodes  (4GB RAM):  ${T3_IDS[*]}"
    echo "  t2        — T2 GPU nodes  (8GB RAM):  ${T2_IDS[*]}"
    echo "  reasoning — Best reasoning models:    ${REASONING_IDS[*]}"
    echo "  vision    — Vision/multimodal models: ${VISION_IDS[*]}"
    echo "  all       — Every model in the list"
    echo
    read -rp "Pull via Ollama? Enter group or model ID (or 'q' to quit): " choice
    [[ "$choice" == "q" ]] && exit 0
    check_ollama
    pull_group "$choice"
}

pull_group() {
    local group="$1"
    local ids=()
    case "$group" in
        t4)        ids=("${T4_IDS[@]}") ;;
        t3)        ids=("${T3_IDS[@]}") ;;
        t2)        ids=("${T2_IDS[@]}") ;;
        reasoning) ids=("${REASONING_IDS[@]}") ;;
        vision)    ids=("${VISION_IDS[@]}") ;;
        all)       ids=("${ALL_IDS[@]}") ;;
        *)
            # Treat as single model ID
            if [[ -v "OLLAMA_TAG[$group]" ]]; then
                ids=("$group")
            else
                die "Unknown group or model ID: '$group'"
            fi
            ;;
    esac

    echo
    info "Pulling ${#ids[@]} model(s): ${ids[*]}"
    echo
    local failed=()
    for id in "${ids[@]}"; do
        ollama_pull "$id" || failed+=("$id")
    done

    echo
    if [[ ${#failed[@]} -eq 0 ]]; then
        ok "All downloads complete."
    else
        warn "Failed: ${failed[*]}"
    fi
}

# ─── Entry point ──────────────────────────────────────────────────────────────

MODE="${1:-menu}"
ARG2="${2:-}"

case "$MODE" in
    menu)
        interactive_menu
        ;;

    ollama)
        check_ollama
        [[ -z "$ARG2" ]] && die "Usage: $0 ollama <group|model-id|all>"
        pull_group "$ARG2"
        ;;

    hf)
        check_hf
        [[ -z "$ARG2" ]] && die "Usage: $0 hf <model-id>"
        [[ -v "HF_REPO[$ARG2]" ]] || die "Unknown model ID: $ARG2"
        hf_download "${HF_REPO[$ARG2]}"
        ;;

    gguf)
        check_hf
        mkdir -p "$GGUF_DIR"
        if [[ "$ARG2" == "all" ]]; then
            for id in "${ALL_IDS[@]}"; do
                gguf_download "$id"
            done
        elif [[ -n "$ARG2" ]]; then
            [[ -v "GGUF_REPO[$ARG2]" ]] || die "Unknown model ID: $ARG2"
            gguf_download "$ARG2"
        else
            die "Usage: $0 gguf <model-id|all>"
        fi
        ;;

    list)
        print_catalogue
        ;;

    *)
        echo "Usage: $0 [menu|list|ollama <group>|hf <id>|gguf <id>]"
        echo "       $0 ollama t3           # pull T3 models"
        echo "       $0 ollama qwen3-8b     # pull one model"
        echo "       $0 gguf phi4-mini      # download GGUF weights"
        echo "       $0 list                # show catalogue"
        exit 1
        ;;
esac
