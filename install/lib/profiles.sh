#!/usr/bin/env bash
# profiles.sh — component matrix and tier→model mapping
# Must stay in sync with core/classifier/hardware.py:TIER_DEFINITIONS

# model_for_tier <tier>  → prints the Ollama model name for that tier
model_for_tier() {
    case "${1:?tier required}" in
        T1) echo "llama3.1:70b" ;;
        T2) echo "qwen3:8b"     ;;
        T3) echo "qwen3:4b"     ;;
        T4) echo "qwen3:0.6b"   ;;
        *)  return 1            ;;
    esac
}

# profile_requires <role> <component>   → 0 if required, 1 if not
profile_requires() {
    local role="$1" comp="$2"
    case "$role" in
        hub)
            case "$comp" in
                core|hub_services|llm_backends|gpu|hermes_moe|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        lightweight)
            case "$comp" in
                core|registration_agent|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        full-node)
            # Default to T2 behaviour; use profile_requires_for_tier for tier-specific checks
            case "$comp" in
                core|llm_backends|gpu|hermes_moe|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

# profile_requires_for_tier <role> <tier> <component>
profile_requires_for_tier() {
    local role="$1" tier="$2" comp="$3"
    if [[ "$role" != "full-node" ]]; then
        profile_requires "$role" "$comp"
        return
    fi
    case "$tier" in
        T1|T2)
            case "$comp" in
                core|llm_backends|gpu|hermes_moe|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        T3)
            case "$comp" in
                core|llm_backends|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        T4)
            case "$comp" in
                core|systemd|ollama) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

# profile_optional <role> <component>   → 0 if optional, 1 if not
profile_optional() {
    local role="$1" comp="$2"
    case "$role" in
        hub|full-node)
            case "$comp" in
                discord|integrations|dashboard) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}
