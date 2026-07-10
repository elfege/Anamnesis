"""Smoke tests — must pass on every PR before merge to main.

Covers the three main API surfaces this session added: rolling-episode
upsert (MSG-526), RunPod lifecycle safety gates (pin #7), consolidation
status (M0-M3). Plus core /health and dashboard stats.

These tests DO NOT mutate the corpus or spend money. Safe to run on prod.
"""
import uuid

import pytest


def test_health(http):
    r = http.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard_stats_reachable(http):
    r = http.get("/api/dashboard/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_episodes" in body
    assert isinstance(body["total_episodes"], int)


def test_rolling_upsert_create_then_append(http):
    """Contract: same (handle, session_id) → same episode_id, delta appends."""
    session = str(uuid.uuid4())
    payload = {
        "handle": "e2e-test-runner",
        "session_id": session,
        "delta": "smoke turn 1",
        "tags": ["e2e-smoke"],
    }
    r1 = http.post("/api/episodes/rolling/upsert", json=payload)
    assert r1.status_code == 200, r1.text
    ep1 = r1.json()
    assert ep1["created"] is True
    assert ep1["compaction_triggered"] is False

    payload["delta"] = "smoke turn 2"
    r2 = http.post("/api/episodes/rolling/upsert", json=payload)
    assert r2.status_code == 200, r2.text
    ep2 = r2.json()
    assert ep2["episode_id"] == ep1["episode_id"]
    assert ep2["created"] is False
    assert ep2["total_chars"] > ep1["total_chars"]


def test_rolling_upsert_missing_delta_400(http):
    r = http.post("/api/episodes/rolling/upsert", json={"handle": "e2e", "session_id": "s"})
    assert r.status_code == 400


def test_runpod_lifecycle_tiers(http):
    r = http.get("/api/avatar/runpod/lifecycle/tiers")
    assert r.status_code == 200
    tiers = r.json()["tiers"]
    aliases = {t["alias"] for t in tiers}
    assert {"rtx3090", "rtx4090", "a100", "h100"}.issubset(aliases)
    # A100 + H100 must require hard confirm
    for t in tiers:
        if t["alias"] in {"a100", "h100"}:
            assert t["hard_confirm"] is True


def test_runpod_lifecycle_start_refuses_without_confirm(http):
    """Pin #7 contract: no start without confirm_cost=true."""
    r = http.post("/api/avatar/runpod/lifecycle/start", json={"gpu_tier": "rtx3090"})
    assert r.status_code == 402
    assert "confirm_cost" in r.text.lower()


def test_runpod_lifecycle_start_h100_refuses_without_ack_string(http):
    """Above the $1/hr threshold, need cost_ack_string too."""
    r = http.post(
        "/api/avatar/runpod/lifecycle/start",
        json={"gpu_tier": "h100", "confirm_cost": True},
    )
    assert r.status_code == 402
    assert "cost_ack_string" in r.text.lower()


def test_consolidation_status_reachable(http):
    r = http.get("/api/consolidation/status")
    assert r.status_code == 200
    body = r.json()
    assert "currently_superseded" in body
    assert "currently_consolidated" in body
    assert "recent_runs" in body
