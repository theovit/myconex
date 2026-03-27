#!/usr/bin/env bash
# ui.sh — TUI abstraction layer
# UI_MODE: tui | plain | unattended  (set by install.sh before sourcing)

# Single-choice menu. Sets REPLY to chosen value.
# Usage: ui_menu "Title" "item1" "item2" ...
ui_menu() {
    local title="$1"; shift
    local items=("$@")
    case "${UI_MODE:-plain}" in
        tui)
            local menu_items=()
            for i in "${!items[@]}"; do menu_items+=("$i" "${items[$i]}"); done
            REPLY=$(whiptail --title "$title" --menu "" 15 60 "${#items[@]}" \
                "${menu_items[@]}" 3>&1 1>&2 2>&3)
            REPLY="${items[$REPLY]}"
            ;;
        unattended)
            REPLY="${items[0]}"  # first option; override via answer file
            ;;
        *)
            echo "$title"
            for i in "${!items[@]}"; do echo "  $((i+1))) ${items[$i]}"; done
            read -r -p "Choice [1]: " choice
            REPLY="${items[$(( ${choice:-1} - 1 ))]}"
            ;;
    esac
}

# Multi-select checklist. Sets SELECTED as space-separated values.
# Usage: ui_checklist "Title" "item1:on" "item2:off" ...
ui_checklist() {
    local title="$1"; shift
    SELECTED=()
    case "${UI_MODE:-plain}" in
        tui)
            local check_items=()
            for item in "$@"; do
                local name="${item%%:*}" state="${item##*:}"
                check_items+=("$name" "" "$state")
            done
            local result
            result=$(whiptail --title "$title" --checklist "" 20 60 "${#@}" \
                "${check_items[@]}" 3>&1 1>&2 2>&3)
            # shellcheck disable=SC2206
            IFS=' ' read -r -a SELECTED <<< "${result//\"/}"
            ;;
        unattended)
            for item in "$@"; do
                [[ "${item##*:}" == "on" ]] && SELECTED+=("${item%%:*}")
            done
            ;;
        *)
            echo "$title (space to toggle, enter to confirm)"
            for item in "$@"; do
                local name="${item%%:*}" state="${item##*:}"
                read -r -p "  Include ${name}? [${state}]: " ans
                ans="${ans:-$state}"
                [[ "$ans" =~ ^(on|yes|y|1)$ ]] && SELECTED+=("$name")
            done
            ;;
    esac
}

# Text / secret input. Stores result in named variable.
# Usage: ui_input "Prompt" VAR_NAME [secret]
ui_input() {
    local prompt="$1" varname="$2" secret="${3:-}"
    # Unattended: read from MYCONEX_VAL_<VARNAME>
    if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
        local envkey="MYCONEX_VAL_${varname}"
        printf -v "$varname" '%s' "${!envkey:-}"
        return
    fi
    local val
    if [[ -n "$secret" ]]; then
        read -r -s -p "$prompt " val; echo
    else
        read -r -p "$prompt " val
    fi
    printf -v "$varname" '%s' "$val"
}

# Yes/no confirmation. Returns 0=yes, 1=no.
# Usage: ui_confirm "Proceed?"
ui_confirm() {
    local prompt="$1"
    case "${UI_MODE:-plain}" in
        tui)
            whiptail --yesno "$prompt" 8 40
            ;;
        unattended)
            return 0
            ;;
        *)
            read -r -p "$prompt [Y/n]: " ans
            [[ "${ans:-y}" =~ ^[Yy] ]]
            ;;
    esac
}
