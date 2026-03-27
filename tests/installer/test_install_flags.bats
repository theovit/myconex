#!/usr/bin/env bats
load 'test_helper'

@test "install.sh --help exits 0" {
    run bash "${REPO_ROOT}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "install.sh --role hub sets MYCONEX_ROLE=hub" {
    run bash -c "
        source '${REPO_ROOT}/install.sh' --parse-only --role hub
        echo \"\$MYCONEX_ROLE\"
    "
    assert_output "hub"
}

@test "install.sh --unattended missing file exits 1" {
    run bash "${REPO_ROOT}/install.sh" --unattended /nonexistent.yaml
    assert_failure
    assert_output --partial "not found"
}
