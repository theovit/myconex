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
