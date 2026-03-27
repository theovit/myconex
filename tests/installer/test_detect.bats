#!/usr/bin/env bats
load 'test_helper'

setup() { source_lib "detect.sh"; }

@test "detect_pkg_manager returns apt on Ubuntu" {
    mock_cmd apt-get 0
    run detect_pkg_manager
    assert_success
    assert_output "apt"
}

@test "detect_pkg_manager returns dnf on Fedora" {
    apt-get() { return 1; }; export -f apt-get
    mock_cmd dnf 0
    run detect_pkg_manager
    assert_success
    assert_output "dnf"
}

@test "detect_pkg_manager returns pacman on Arch" {
    apt-get() { return 1; }; export -f apt-get
    dnf() { return 1; }; export -f dnf
    mock_cmd pacman 0
    run detect_pkg_manager
    assert_success
    assert_output "pacman"
}

@test "detect_wsl returns true inside WSL" {
    MYCONEX_TEST_KERNEL="Linux-5.15-microsoft-standard-WSL2"
    run detect_wsl
    assert_success
}

@test "detect_wsl returns false outside WSL" {
    MYCONEX_TEST_KERNEL="Linux-6.1.0-generic"
    run detect_wsl
    assert_failure
}

@test "detect_display returns tui when DISPLAY is set" {
    DISPLAY=":0"
    mock_cmd whiptail 0
    run detect_display_mode
    assert_output "tui"
}

@test "detect_display returns plain over SSH" {
    unset DISPLAY
    SSH_CONNECTION="10.0.0.1 12345 10.0.0.2 22"
    run detect_display_mode
    assert_output "plain"
}

@test "detect_tier returns T1 for >24GB VRAM" {
    # nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits outputs a bare integer
    mock_cmd nvidia-smi 0 "24564"
    run detect_tier
    assert_output "T1"
}

@test "detect_tier returns T2 for 8-24GB VRAM" {
    mock_cmd nvidia-smi 0 "12288"
    run detect_tier
    assert_output "T2"
}

@test "detect_tier returns T3 for no GPU, >8 cores" {
    nvidia-smi() { return 1; }; export -f nvidia-smi
    MYCONEX_TEST_CPU_CORES=20
    MYCONEX_TEST_RAM_GB=32
    run detect_tier
    assert_output "T3"
}

@test "detect_tier returns T4 for no GPU, <=4 cores" {
    nvidia-smi() { return 1; }; export -f nvidia-smi
    MYCONEX_TEST_CPU_CORES=4
    MYCONEX_TEST_RAM_GB=4
    run detect_tier
    assert_output "T4"
}
