#!/usr/bin/env bats
load 'test_helper'

setup() {
    source_lib "steps/config.sh"
    MYCONEX_REPO_ROOT="$(mktemp -d)"
    mkdir -p "${MYCONEX_REPO_ROOT}/config"
    MYCONEX_NODE_NAME="test-hub"
    MYCONEX_DETECTED_TIER="T2"
    MYCONEX_ROLE="hub"
    UI_MODE="unattended"
    # Export all needed vars
    export MYCONEX_REPO_ROOT MYCONEX_NODE_NAME MYCONEX_DETECTED_TIER MYCONEX_ROLE UI_MODE
    # Stub log_step (defined by install.sh at runtime)
    log_step() { echo "[step] $*"; }
    export -f log_step
}

teardown() { rm -rf "$MYCONEX_REPO_ROOT"; }

@test "config.sh writes node.yaml with correct role" {
    run step_config
    assert_success
    run grep 'name: "test-hub"' "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
}

@test "config.sh writes node.yaml with detected tier" {
    run step_config
    assert_success
    run grep 'tier: "T2"' "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
}

@test "config.sh does not overwrite existing node.yaml on upgrade" {
    echo "existing: true" > "${MYCONEX_REPO_ROOT}/config/node.yaml"
    REINSTALL=""
    export REINSTALL
    # Create sentinel to simulate "already installed"
    mkdir -p "${HOME}/.myconex"
    touch "${HOME}/.myconex/.installed_config"
    run step_config
    run grep "existing: true" "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
    # Clean up sentinel
    rm -f "${HOME}/.myconex/.installed_config"
}
