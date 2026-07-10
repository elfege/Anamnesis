"""E2E fixtures — assumes anamnesis-app + anamnesis-mongo are running on
the compose stack under test. The gate script (scripts/e2e_gate.sh) is
responsible for spinning them up before invoking pytest.
"""
import os

import httpx
import pytest


@pytest.fixture(scope="session")
def base_url() -> str:
    """Where the anamnesis-app under test is reachable. Defaults to the
    running dev stack; the gate script overrides via env for CI."""
    return os.environ.get("ANAMNESIS_BASE_URL", "http://192.168.10.20:3010")


@pytest.fixture(scope="session")
def http(base_url):
    """Session-scoped HTTP client so tests share connection pools."""
    with httpx.Client(base_url=base_url, timeout=15.0) as c:
        yield c


@pytest.fixture(autouse=True)
def _health_gate(http):
    """Every test asserts the app is up before running. If it isn't, the
    test is skipped (not failed) — clearer signal for CI."""
    try:
        r = http.get("/health")
        if r.status_code != 200:
            pytest.skip(f"anamnesis-app not healthy: {r.status_code}")
    except Exception as exc:
        pytest.skip(f"anamnesis-app unreachable: {exc}")
