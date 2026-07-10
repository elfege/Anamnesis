#!/usr/bin/env bash
# e2e_gate.sh — the E2E gate CI runs on every PR to main.
#
# Behavior:
#   1. Ensure the anamnesis-app dev stack is up + healthy.
#   2. Install test deps if missing (httpx + pytest).
#   3. Run tests/e2e/ against the live stack via ANAMNESIS_BASE_URL.
#   4. Non-zero exit if any test fails — CI wiring sees the failure.
#
# The test suite is intentionally NON-MUTATING against production data:
#   - It exercises safety-gate refusal codes (402 without confirm_cost)
#   - It creates rolling episodes under a distinct handle ("e2e-test-runner")
#     so they don't interfere with real dellserver-anamnesis:* sessions
#   - It does NOT hit real RunPod, does NOT spend money
#
# Safe to run on the live dev stack. For a truly hermetic run, compose-up
# a throwaway stack first — see docker-compose.test.yml (future work).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL="${ANAMNESIS_BASE_URL:-http://192.168.10.20:3010}"
echo "e2e-gate: target = $BASE_URL"

# --- 1. Health probe with retries (60s max) --------------------
echo "e2e-gate: waiting for /health ..."
for i in {1..30}; do
    if curl -sf -o /dev/null --max-time 3 "$BASE_URL/health"; then
        echo "  healthy"
        break
    fi
    if (( i == 30 )); then
        echo "e2e-gate: /health never came up at $BASE_URL after 60s" >&2
        exit 1
    fi
    sleep 2
done

# --- 2. Ensure test deps ---------------------------------------
if ! python3 -c "import pytest, httpx" 2>/dev/null; then
    echo "e2e-gate: installing pytest + httpx ..."
    # PEP 668 on modern Debian/Ubuntu requires either --user or a venv or
    # --break-system-packages. Try in order of least-invasive.
    if pip install --quiet --user pytest httpx 2>/dev/null; then :; \
    elif pip install --quiet --break-system-packages pytest httpx 2>/dev/null; then :; \
    else
        echo "e2e-gate: failed to install pytest/httpx — try 'pipx install pytest httpx' or a venv" >&2
        exit 1
    fi
fi

# --- 3. Run tests ----------------------------------------------
cd "$REPO_DIR"
echo "e2e-gate: running pytest tests/e2e/ ..."
ANAMNESIS_BASE_URL="$BASE_URL" python3 -m pytest tests/e2e/ -v --tb=short "$@"
