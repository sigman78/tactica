"""Dashboard API tests: meta, presets CRUD, job lifecycle, SSE, replay frames."""
from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tactica.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(presets_dir=tmp_path / "experiments", root_dir=tmp_path)
    with TestClient(app) as c:
        yield c


def wait_done(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in ("done", "failed", "cancelled"):
            return info
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def start(client: TestClient, kind: str, config: dict) -> str:
    res = client.post("/api/jobs", json={"kind": kind, "config": config})
    assert res.status_code == 200, res.text
    return res.json()["id"]


# --------------------------------------------------------------------- #
# meta + presets


def test_meta(client):
    meta = client.get("/api/meta").json()
    assert "open_field" in meta["scenarios"]
    assert {u["glyph"] for u in meta["units"]} == {"P", "A", "G", "S", "C"}
    assert "tournament" in meta["kinds"]


def test_preset_crud(client):
    assert client.get("/api/presets").json() == []
    body = {"name": "my-exp", "kind": "tournament",
            "description": "d", "config": {"agents": "random,heuristic"}}
    assert client.post("/api/presets", json=body).status_code == 200
    presets = client.get("/api/presets").json()
    assert [p["name"] for p in presets] == ["my-exp"]
    assert presets[0]["config"]["agents"] == "random,heuristic"
    assert client.delete("/api/presets/my-exp").status_code == 200
    assert client.get("/api/presets").json() == []
    assert client.delete("/api/presets/my-exp").status_code == 404
    assert client.post("/api/presets", json={"name": "../evil", "kind": "play"}
                       ).status_code == 400
    assert client.post("/api/presets", json={"name": "x", "kind": "nope"}
                       ).status_code == 400


# --------------------------------------------------------------------- #
# jobs


def test_bad_job_kind(client):
    res = client.post("/api/jobs", json={"kind": "nope", "config": {}})
    assert res.status_code == 400


def test_tournament_job(client):
    job_id = start(client, "tournament", {
        "agents": "random,heuristic", "scenarios": "open_field",
        "pairs": 2, "seed": 1, "workers": 1})
    info = wait_done(client, job_id)
    assert info["status"] == "done", info["error"]
    result = info["result"]
    assert result["n_games"] == 4
    [entry] = [m for m in result["matrix"]
               if m["a"] == "heuristic" and m["b"] == "random"]
    assert entry["mean"] > 0.9  # heuristic crushes random
    games = client.get(f"/api/jobs/{job_id}/games").json()
    assert len(games) == 4

    # frames for the first logged game re-simulate and hash-verify
    frames = client.get(f"/api/jobs/{job_id}/frames/0").json()
    assert frames["board"] == {"w": 11, "h": 9}
    assert len(frames["frames"]) >= 2
    deployment = frames["frames"][0]
    assert deployment["action"] is None
    assert len(deployment["stacks"]) == 10  # open_field: 5 stacks per side
    assert client.get(f"/api/jobs/{job_id}/frames/99").status_code == 404


def test_failed_job_surfaces_error(client):
    job_id = start(client, "tournament", {"agents": "random"})  # < 2 agents
    info = wait_done(client, job_id)
    assert info["status"] == "failed"
    assert "two agent specs" in info["error"]


def test_play_job_and_sse_history(client):
    job_id = start(client, "play", {
        "p0": "heuristic", "p1": "random",
        "scenario": "skirmish", "seed": 3})
    info = wait_done(client, job_id)
    assert info["status"] == "done"
    assert info["result"]["winner"] in (0, 1, None)

    # SSE replays full history and terminates after the terminal status
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as res:
        for line in res.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    types = [e["type"] for e in events]
    assert types[0] == "status" and events[0]["data"]["status"] == "running"
    assert types[-1] == "status" and events[-1]["data"]["status"] == "done"
    assert any(t == "log" for t in types)


def test_sprt_job(client):
    # identical weights -> H0 or max-pairs; tiny cap keeps it fast
    job_id = start(client, "sprt", {
        "candidate": "random", "baseline": "random",
        "scenarios": "open_field", "max_pairs": 3, "seed": 1})
    info = wait_done(client, job_id)
    assert info["status"] == "done", info["error"]
    result = info["result"]
    assert result["pairs"] <= 3
    assert len(result["trajectory"]) == result["pairs"]
    assert len(result["bounds"]) == 2


def test_skill_curve_job(client):
    job_id = start(client, "skill-curve", {
        "agent": "random", "epsilons": [0, 0.5], "pairs": 1,
        "scenarios": "open_field", "workers": 1, "seed": 1})
    info = wait_done(client, job_id)
    assert info["status"] == "done", info["error"]
    assert [p["eps"] for p in info["result"]["points"]] == [0, 0.5]


def test_noise_floor_job(client):
    job_id = start(client, "noise-floor", {
        "agent": "heuristic", "pairs": 2,
        "scenarios": "open_field", "workers": 1, "seed": 1})
    info = wait_done(client, job_id)
    assert info["status"] == "done", info["error"]
    rows = info["result"]["rows"]
    assert rows[0]["scenario"] == "open_field"
    # deterministic agent on mirrored pairs: pair-level noise is exactly zero
    assert rows[0]["pair_mean"] == 0.5 and rows[0]["pair_ci"] == 0.0


# --------------------------------------------------------------------- #
# replay files on disk


def test_replay_files(client, tmp_path):
    from tactica.eval.runner import run_match, write_jsonl
    from tactica.scenario import load_scenario

    log_dir = tmp_path / "replays"
    log_dir.mkdir(exist_ok=True)
    record = run_match("random", "random", load_scenario("skirmish"), 5)
    write_jsonl(log_dir / "x.jsonl", [record.to_dict()])

    files = client.get("/api/replays").json()
    assert {"file": "replays/x.jsonl", "games": 1} in files
    rows = client.get("/api/replays/games", params={"file": "replays/x.jsonl"}).json()
    assert rows[0]["scenario"] == "skirmish"
    frames = client.get("/api/replays/frames",
                        params={"file": "replays/x.jsonl", "index": 0}).json()
    assert frames["scenario"] == "skirmish"
    assert frames["frames"][0]["stacks"]
    assert len(frames["frames"]) == len(record.actions) + 1

    assert client.get("/api/replays/games",
                      params={"file": "../outside.jsonl"}).status_code in (400, 404)


# --------------------------------------------------------------------- #
# interactive human-vs-agent games


def test_game_lifecycle(client):
    res = client.post("/api/games", json={
        "agent": "random", "scenario": "skirmish", "seed": 2, "human_side": 0})
    assert res.status_code == 200, res.text
    state = res.json()
    assert state["status"] == "your-turn"        # agent auto-played to us
    assert state["active"] is not None
    active = next(s for s in state["stacks"] if s["uid"] == state["active"])
    assert active["side"] == 0
    assert state["legal"], "human turn must offer legal actions"
    assert state["queue"][0]["uid"] == state["active"]
    # attack entries carry a damage preview
    for a in state["legal"]:
        if a["type"].startswith("MELEE_") or a["type"] == "RANGED_ATTACK":
            assert a["est"] >= 1 and "target_uid" in a

    # illegal action id -> 400 (pick an id outside the legal set)
    legal_ids = {a["id"] for a in state["legal"]}
    illegal = next(i for i in range(12 * 99) if i not in legal_ids)
    bad = client.post(f"/api/games/{state['id']}/act", json={"action": illegal})
    assert bad.status_code == 400

    # play the whole battle picking the first legal action every turn
    for _ in range(600):
        if state["status"] == "over":
            break
        state = client.post(f"/api/games/{state['id']}/act",
                            json={"action": state["legal"][0]["id"]}).json()
    assert state["status"] == "over"
    assert state["you_won"] in (True, False, None)
    assert state["log"], "battle log must narrate the game"

    # structured events mirror the log, with combat effects for animation
    assert len(state["events"]) == len(state["log"])
    assert {e["t"] for e in state["events"]} <= {"move", "attack", "wait", "defend"}
    attack_events = [e for e in state["events"] if e["t"] == "attack"]
    assert attack_events, "a finished battle must contain attacks"
    for e in attack_events:
        assert "target" in e and "to" in e
        for ef in e["effects"]:
            assert ef["dmg"] >= 1 and ef["count"] >= 0
            assert isinstance(ef["dead"], bool)

    # cannot act after the end
    res = client.post(f"/api/games/{state['id']}/act", json={"action": 0})
    assert res.status_code == 400

    # save -> hash-verified replay round-trips through the file API
    saved = client.post(f"/api/games/{state['id']}/save")
    assert saved.status_code == 200
    file = saved.json()["file"]
    frames = client.get("/api/replays/frames",
                        params={"file": f"replays/human-{state['id']}.jsonl",
                                "index": 0})
    assert frames.status_code == 200, (file, frames.text)
    assert frames.json()["specs"][0] == "human"

    assert any(g["id"] == state["id"] for g in client.get("/api/games").json())


def test_game_validation(client):
    assert client.post("/api/games", json={"scenario": "nope"}).status_code == 400
    assert client.post("/api/games", json={"human_side": 7}).status_code == 400
    assert client.post("/api/games", json={"agent": "bogus"}).status_code == 400
    assert client.get("/api/games/deadbeef").status_code == 404
    # saving an unfinished game is refused
    state = client.post("/api/games", json={"agent": "random"}).json()
    assert client.post(f"/api/games/{state['id']}/save").status_code == 400


def test_cancel_job(client):
    job_id = start(client, "sprt", {
        "candidate": "random", "baseline": "random",
        "scenarios": "all", "max_pairs": 5000, "seed": 1})
    client.post(f"/api/jobs/{job_id}/cancel")
    info = wait_done(client, job_id)
    assert info["status"] == "cancelled"


def test_legal_payload_exposes_directional_melee(client):
    # Swordsman next to a pikeman: several approach sides, each its own entry
    # with a `from` square and an `est`, exactly one marked is_default.
    res = client.post("/api/games", json={
        "agent": "random", "scenario": "skirmish", "seed": 4, "human_side": 0})
    state = res.json()
    melee = [a for a in state["legal"] if a["type"].startswith("MELEE_")]
    if melee:  # skirmish opening may be out of melee range; only assert shape
        for a in melee:
            assert a["est"] >= 1
            assert "from" in a and "target_uid" in a and "dir" in a
        per_target = {}
        for a in melee:
            per_target.setdefault(a["target_uid"], []).append(a)
        for entries in per_target.values():
            assert sum(1 for a in entries if a.get("is_default")) == 1
