# Load bats helpers
load '../lib/bats-support/load'
load '../lib/bats-assert/load'

# Repo root (two levels up from tests/installer/)
REPO_ROOT="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"

# Source a lib file from the installer
source_lib() {
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/install/lib/${1}"
}

# Mock a command: mock_cmd <name> <exit_code> [stdout]
# Creates a temporary function override visible in the test scope
mock_cmd() {
    local name="$1" exit_code="$2" output="${3:-}"
    eval "${name}() { echo \"${output}\"; return ${exit_code}; }"
    export -f "${name}"
}
