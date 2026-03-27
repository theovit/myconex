#!/usr/bin/env bats
load 'test_helper'

setup() { source_lib "profiles.sh"; }

@test "hub requires core" {
    run profile_requires "hub" "core"
    assert_success
}

@test "hub requires hub_services" {
    run profile_requires "hub" "hub_services"
    assert_success
}

@test "lightweight skips hub_services" {
    run profile_requires "lightweight" "hub_services"
    assert_failure
}

@test "T4 full-node skips hermes_moe" {
    run profile_requires_for_tier "full-node" "T4" "hermes_moe"
    assert_failure
}

@test "T2 full-node requires hermes_moe" {
    run profile_requires_for_tier "full-node" "T2" "hermes_moe"
    assert_success
}

@test "model_for_tier T2 returns qwen3:8b" {
    run model_for_tier "T2"
    assert_output "qwen3:8b"
}

@test "model_for_tier T4 returns qwen3:0.6b" {
    run model_for_tier "T4"
    assert_output "qwen3:0.6b"
}
