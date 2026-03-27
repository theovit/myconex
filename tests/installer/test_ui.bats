#!/usr/bin/env bats
load 'test_helper'

setup() {
    source_lib "ui.sh"
    export UI_MODE="plain"   # default to plain in tests
}

@test "ui_confirm returns 0 on 'y' input" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=plain
        echo y | ui_confirm 'Proceed?'
    "
    assert_success
}

@test "ui_confirm returns 1 on 'n' input" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=plain
        echo n | ui_confirm 'Proceed?'
    "
    assert_failure
}

@test "ui_confirm in unattended mode returns 0 (auto-yes)" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=unattended
        ui_confirm 'Proceed?'
    "
    assert_success
}

@test "ui_input in unattended mode reads from MYCONEX_UNATTENDED_VALUES" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=unattended
        MYCONEX_VAL_NODE_NAME='test-hub'
        ui_input 'Node name:' NODE_NAME
        echo \"\$NODE_NAME\"
    "
    assert_output "test-hub"
}
