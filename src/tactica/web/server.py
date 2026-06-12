"""FastAPI app behind ``tactica web``: REST + SSE over the job manager.

Endpoints (all JSON unless noted):

- ``GET  /``                        dashboard page (static)
- ``GET  /api/meta``                scenarios, unit stats, weights files, spec help
- ``GET  /api/presets``             list experiment presets
- ``POST /api/presets``             create/update a preset (body: {name, kind, config, ...})
- ``DELETE /api/presets/{name}``    remove a preset
- ``POST /api/jobs``                start a job (body: {kind, config})
- ``GET  /api/jobs``                job summaries, newest first
- ``GET  /api/jobs/{id}``           summary + result
- ``POST /api/jobs/{id}/cancel``    cooperative cancel
- ``GET  /api/jobs/{id}/events``    SSE stream of job events
- ``GET  /api/jobs/{id}/games``     per-game rows of a finished/running job
- ``GET  /api/jobs/{id}/frames/{index}``  replay frames for one job game
- ``GET  /api/replays``             *.jsonl logs found on disk
- ``GET  /api/replays/games``       game rows of one log file
- ``GET  /api/replays/frames``      replay frames for one log row
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tactica.eval.runner import read_jsonl
from tactica.scenario import BUILTIN_SCENARIOS
from tactica.units import GLYPHS, STATS
from tactica.web.frames import game_frames
from tactica.web.games import GameManager
from tactica.web.jobs import RUNNERS, JobManager

STATIC_DIR = Path(__file__).parent / "static"
PRESET_NAME_RE = re.compile(r"^[\w][\w .-]{0,63}$")

AGENT_SPEC_HELP = [
    "random", "heuristic", "weighted", "weighted:weights/conservative.json",
    "mcts:32", "mcts:64:1.4", "epsilon:0.1:heuristic",
]


def _game_summary(index: int, g: dict) -> dict:
    return {"index": index, "scenario": g["scenario_name"],
            "specs": list(g["specs"]), "winner": g["winner"],
            "rounds": g["rounds"], "seed": g["seed"]}


def create_app(presets_dir: str | Path = "experiments",
               root_dir: str | Path = ".") -> FastAPI:
    app = FastAPI(title="tactica dashboard", docs_url=None, redoc_url=None)
    manager = JobManager()
    games = GameManager(replays_dir=Path(root_dir) / "replays")
    presets_path = Path(presets_dir)
    root = Path(root_dir).resolve()

    # ------------------------------------------------------------------ #
    # meta + presets

    @app.get("/api/meta")
    def meta() -> dict:
        weights = sorted(str(p.as_posix()) for p in root.glob("weights/*.json"))
        return {
            "scenarios": {name: sc.to_dict()
                          for name, sc in BUILTIN_SCENARIOS.items()},
            "units": [{"name": s.name, "glyph": GLYPHS[t], "speed": s.speed,
                       "attack": s.attack, "defense": s.defense,
                       "dmg": [s.dmg_min, s.dmg_max], "hp": s.hp,
                       "ranged": s.is_ranged, "flyer": s.is_flyer,
                       "initiative": s.initiative}
                      for t, s in STATS.items()],
            "weights": [Path(w).relative_to(root).as_posix()
                        if Path(w).is_absolute() else w for w in weights],
            "agent_examples": AGENT_SPEC_HELP,
            "kinds": list(RUNNERS),
        }

    def _preset_file(name: str) -> Path:
        if not PRESET_NAME_RE.match(name):
            raise HTTPException(400, f"invalid preset name {name!r}")
        return presets_path / f"{name}.json"

    @app.get("/api/presets")
    def list_presets() -> list[dict]:
        out = []
        if presets_path.is_dir():
            for f in sorted(presets_path.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    data["name"] = f.stem
                    out.append(data)
                except (json.JSONDecodeError, OSError):
                    out.append({"name": f.stem, "error": "unreadable preset"})
        return out

    @app.post("/api/presets")
    def save_preset(body: dict) -> dict:
        name = body.get("name", "")
        kind = body.get("kind")
        if kind not in RUNNERS:
            raise HTTPException(400, f"invalid kind {kind!r}")
        f = _preset_file(name)
        presets_path.mkdir(parents=True, exist_ok=True)
        payload = {"kind": kind, "description": body.get("description", ""),
                   "config": body.get("config", {})}
        f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {**payload, "name": name}

    @app.delete("/api/presets/{name}")
    def delete_preset(name: str) -> dict:
        f = _preset_file(name)
        if not f.is_file():
            raise HTTPException(404, f"no preset {name!r}")
        f.unlink()
        return {"deleted": name}

    # ------------------------------------------------------------------ #
    # jobs

    @app.post("/api/jobs")
    def start_job(body: dict) -> dict:
        kind = body.get("kind")
        config = body.get("config", {})
        if not isinstance(config, dict):
            raise HTTPException(400, "config must be an object")
        try:
            job = manager.submit(kind, config)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return job.summary()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return manager.list()

    def _job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/jobs/{job_id}")
    def job_info(job_id: str) -> dict:
        return _job(job_id).info()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        job = _job(job_id)
        job.cancel_flag.set()
        return job.summary()

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str) -> StreamingResponse:
        _job(job_id)  # 404 before the stream starts

        def gen():
            for event in manager.subscribe(job_id):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/games")
    def job_games(job_id: str) -> list[dict]:
        job = _job(job_id)
        return [_game_summary(i, g) for i, g in enumerate(job.games)]

    @app.get("/api/jobs/{job_id}/frames/{index}")
    def job_frames(job_id: str, index: int) -> dict:
        job = _job(job_id)
        if not 0 <= index < len(job.games):
            raise HTTPException(404, f"game index {index} out of range "
                                     f"(job has {len(job.games)} games)")
        return game_frames(job.games[index])

    # ------------------------------------------------------------------ #
    # interactive human-vs-agent games

    def _game(game_id: str):
        try:
            return games.get(game_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/api/games")
    def create_game(body: dict) -> dict:
        try:
            session = games.create(
                agent_spec=body.get("agent", "heuristic"),
                scenario_name=body.get("scenario", "open_field"),
                seed=int(body.get("seed", 1)),
                human_side=int(body.get("human_side", 0)),
                deterministic=bool(body.get("deterministic", False)))
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from None
        return session.state()

    @app.get("/api/games")
    def list_games() -> list[dict]:
        return games.list()

    @app.get("/api/games/{game_id}")
    def game_state(game_id: str) -> dict:
        return _game(game_id).state()

    @app.post("/api/games/{game_id}/act")
    def game_act(game_id: str, body: dict) -> dict:
        session = _game(game_id)
        try:
            action_id = int(body["action"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "body needs an integer 'action'") from None
        with session.lock:
            try:
                session.act(action_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
        return session.state()

    @app.post("/api/games/{game_id}/save")
    def game_save(game_id: str) -> dict:
        try:
            return {"file": games.save_replay(game_id)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    # ------------------------------------------------------------------ #
    # replay files on disk

    def _safe_log_path(file: str) -> Path:
        p = (root / file).resolve()
        if root not in p.parents and p != root:
            raise HTTPException(400, "path escapes project root")
        if not (p.is_file() and p.suffix == ".jsonl"):
            raise HTTPException(404, f"no JSONL log at {file!r}")
        return p

    @app.get("/api/replays")
    def list_replays() -> list[dict]:
        out = []
        for p in sorted(root.glob("*.jsonl")) + sorted(root.glob("replays/*.jsonl")):
            try:
                n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
            except OSError:
                continue
            out.append({"file": p.relative_to(root).as_posix(), "games": n})
        return out

    @app.get("/api/replays/games")
    def replay_games(file: str) -> list[dict]:
        rows = read_jsonl(_safe_log_path(file))
        return [_game_summary(i, g) for i, g in enumerate(rows)]

    @app.get("/api/replays/frames")
    def replay_frames(file: str, index: int = 0) -> dict:
        rows = read_jsonl(_safe_log_path(file))
        if not 0 <= index < len(rows):
            raise HTTPException(404, f"index {index} out of range "
                                     f"(file has {len(rows)} games)")
        try:
            return game_frames(rows[index])
        except AssertionError as exc:
            raise HTTPException(409, str(exc)) from None

    # ------------------------------------------------------------------ #
    # static page

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
