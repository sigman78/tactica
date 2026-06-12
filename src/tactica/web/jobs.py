"""Background evaluation jobs with live event streams.

Each job runs one evaluation kind (tournament / sprt / skill-curve /
noise-floor / play) in a daemon thread, reusing the generator-based pair
runner from :mod:`tactica.eval.runner`. Progress is published as a list of
events that late subscribers replay from the start, so an SSE client that
connects (or reconnects) mid-run still sees the full history.

Cancellation is cooperative: the runner checks a flag between mirrored
pairs and abandons the worker pool without waiting for queued tasks.
"""
from __future__ import annotations

import itertools
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from tactica.eval.runner import _pair_task, derive_seed, write_jsonl
from tactica.eval.stats import mean_ci95, score_to_elo, sprt_bounds, sprt_llr
from tactica.eval.tournament import TournamentResult, openskill_ratings
from tactica.scenario import resolve_scenarios

TERMINAL = ("done", "failed", "cancelled")

PairTask = tuple[str, str, dict, int]
Emit = Callable[[str, dict], None]


def _finite(x: float) -> float | None:
    """JSON-safe number: json.dumps emits bare ``Infinity`` (invalid JSON)
    for the inf CI that mean_ci95 returns on single samples."""
    return x if math.isfinite(x) else None


def _game_score0(game: dict) -> float:
    winner = game["winner"]
    return 0.5 if winner is None else (1.0 if winner == 0 else 0.0)


def stream_pairs(tasks: list[PairTask], workers: int,
                 cancelled: threading.Event) -> Iterator[tuple[dict, dict, float]]:
    """Like :func:`tactica.eval.runner.run_pairs` but cancellable: stops
    yielding once ``cancelled`` is set and abandons the pool without
    waiting for queued tasks. Results stream back in task order."""
    if workers <= 1:
        for t in tasks:
            if cancelled.is_set():
                return
            yield _pair_task(t)
        return
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(_pair_task, t) for t in tasks]
        for fut in futures:
            if cancelled.is_set():
                return
            yield fut.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ----------------------------------------------------------------------- #
# Job kinds


def _matrix_rows(result: TournamentResult) -> list[dict]:
    return [{"a": a, "b": b, "mean": mean, "ci": _finite(ci), "n": n}
            for (a, b), (mean, ci, n) in result.matrix().items()]


def _run_tournament(config: dict, emit: Emit, cancelled: threading.Event,
                    games_out: list[dict]) -> dict:
    specs = config["agents"]
    if isinstance(specs, str):
        specs = [s.strip() for s in specs.split(",") if s.strip()]
    if len(specs) < 2:
        raise ValueError("tournament needs at least two agent specs")
    scenarios = resolve_scenarios(config.get("scenarios", "all"))
    pairs = int(config.get("pairs", 20))
    base_seed = int(config.get("seed", 1))
    workers = int(config.get("workers", 0)) or None

    result = TournamentResult(agents=list(specs))
    tasks: list[PairTask] = []
    meta: list[tuple[str, str, str, int]] = []
    for a, b in itertools.combinations(specs, 2):
        for sc in scenarios:
            for i in range(pairs):
                tasks.append((a, b, sc.to_dict(), derive_seed(base_seed, sc.name, i)))
                meta.append((a, b, sc.name, i))
    emit("log", {"line": f"{len(tasks)} mirrored pairs ({2 * len(tasks)} games), "
                         f"{len(specs)} agents x {len(scenarios)} scenarios"})

    t0 = time.perf_counter()
    last_snapshot = 0.0
    from os import cpu_count
    n_workers = workers if workers else (cpu_count() or 1)
    for k, (g1, g2, score) in enumerate(
            stream_pairs(tasks, n_workers, cancelled)):
        a, b, sc_name, i = meta[k]
        for g, mirrored in ((g1, False), (g2, True)):
            g["pair_index"] = i
            g["mirrored"] = mirrored
            result.games.append(g)
        result.scores.setdefault((a, b), []).append(score)
        result.by_scenario.setdefault((a, b, sc_name), []).append(score)
        now = time.perf_counter()
        if now - last_snapshot > 0.4 or k + 1 == len(tasks):
            last_snapshot = now
            emit("progress", {"completed": k + 1, "total": len(tasks),
                              "elapsed": now - t0,
                              "matrix": _matrix_rows(result)})
        if (k + 1) % 25 == 0:
            emit("log", {"line": f"{k + 1}/{len(tasks)} pairs done "
                                 f"({(k + 1) / (now - t0):.1f} pairs/s)"})
    if cancelled.is_set():
        raise JobCancelled
    games_out.extend(result.games)

    ratings = openskill_ratings(result)
    by_scenario = [{"a": a, "b": b, "scenario": sc,
                    "mean": mean_ci95(vals)[0],
                    "ci": _finite(mean_ci95(vals)[1]),
                    "n": len(vals)}
                   for (a, b, sc), vals in sorted(result.by_scenario.items())]
    out_path = config.get("out")
    if out_path:
        write_jsonl(out_path, result.games)
        emit("log", {"line": f"game log written to {out_path}"})
    return {
        "agents": specs,
        "matrix": _matrix_rows(result),
        "by_scenario": by_scenario,
        "ratings": None if ratings is None else [
            {"agent": ag, "mu": mu, "sigma": sigma, "ordinal": ordinal}
            for ag, mu, sigma, ordinal in ratings],
        "n_games": len(result.games),
        "duration": time.perf_counter() - t0,
    }


def _run_sprt(config: dict, emit: Emit, cancelled: threading.Event,
              games_out: list[dict]) -> dict:
    candidate = config["candidate"]
    baseline = config["baseline"]
    scenarios = resolve_scenarios(config.get("scenarios", "all"))
    elo0 = float(config.get("elo0", 0.0))
    elo1 = float(config.get("elo1", 10.0))
    alpha = float(config.get("alpha", 0.05))
    beta = float(config.get("beta", 0.05))
    max_pairs = int(config.get("max_pairs", 5000))
    base_seed = int(config.get("seed", 1))
    lower, upper = sprt_bounds(alpha, beta)
    emit("log", {"line": f"SPRT: H1 elo>={elo1:g} vs H0 elo<={elo0:g}, "
                         f"alpha={alpha:g} beta={beta:g}, "
                         f"LLR bounds [{lower:.3f}, {upper:.3f}]"})

    wins = draws = losses = 0
    trajectory: list[float] = []
    verdict = "inconclusive (max pairs reached)"
    pair_index = 0
    t0 = time.perf_counter()
    while pair_index < max_pairs:
        if cancelled.is_set():
            raise JobCancelled
        sc = scenarios[pair_index % len(scenarios)]
        seed = derive_seed(base_seed, sc.name, pair_index)
        g1, g2, _score = _pair_task((candidate, baseline, sc.to_dict(), seed))
        games_out.extend((g1, g2))
        for g, cand_side in ((g1, 0), (g2, 1)):
            if g["winner"] is None:
                draws += 1
            elif g["winner"] == cand_side:
                wins += 1
            else:
                losses += 1
        llr = sprt_llr(wins, draws, losses, elo0, elo1)
        trajectory.append(llr)
        pair_index += 1
        emit("progress", {"completed": pair_index, "total": max_pairs,
                          "llr": llr, "wins": wins, "draws": draws,
                          "losses": losses, "elapsed": time.perf_counter() - t0})
        if pair_index % 25 == 0:
            emit("log", {"line": f"n={wins + draws + losses}  "
                                 f"WDL={wins}/{draws}/{losses}  LLR={llr:+.3f}"})
        if llr >= upper:
            verdict = "accept H1 (candidate is stronger)"
            break
        if llr <= lower:
            verdict = "accept H0 (no improvement)"
            break

    n = wins + draws + losses
    score = (wins + draws / 2) / n if n else 0.5
    return {
        "verdict": verdict,
        "pairs": pair_index,
        "wins": wins, "draws": draws, "losses": losses,
        "score": score,
        "elo": score_to_elo(score),
        "llr": trajectory[-1] if trajectory else 0.0,
        "bounds": [lower, upper],
        "trajectory": trajectory,
        "duration": time.perf_counter() - t0,
    }


def _run_skill_curve(config: dict, emit: Emit, cancelled: threading.Event,
                     games_out: list[dict]) -> dict:
    agent = config.get("agent", "heuristic")
    epsilons = config.get("epsilons", [0, 0.05, 0.1, 0.2, 0.5])
    if isinstance(epsilons, str):
        epsilons = [float(e) for e in epsilons.split(",")]
    scenarios = resolve_scenarios(config.get("scenarios", "all"))
    pairs = int(config.get("pairs", 100))
    base_seed = int(config.get("seed", 1))
    from os import cpu_count
    workers = int(config.get("workers", 0)) or (cpu_count() or 1)

    points: list[dict] = []
    total = len(epsilons) * len(scenarios) * pairs
    completed = 0
    t0 = time.perf_counter()
    for eps in epsilons:
        spec = agent if eps == 0 else f"epsilon:{eps:g}:{agent}"
        tasks = [(spec, agent, sc.to_dict(), derive_seed(base_seed, sc.name, i))
                 for sc in scenarios for i in range(pairs)]
        scores: list[float] = []
        for g1, g2, s in stream_pairs(tasks, workers, cancelled):
            games_out.extend((g1, g2))
            scores.append(s)
            completed += 1
            if completed % 10 == 0 or completed == total:
                emit("progress", {"completed": completed, "total": total,
                                  "elapsed": time.perf_counter() - t0})
        if cancelled.is_set():
            raise JobCancelled
        mean, ci = mean_ci95(scores)
        points.append({"eps": eps, "score": mean, "ci": _finite(ci),
                       "n": len(scores)})
        emit("point", points[-1])
        emit("log", {"line": f"eps={eps:<5g} score={mean:.4f} +/-{ci:.4f}"})
    return {"agent": agent, "points": points,
            "duration": time.perf_counter() - t0}


def _run_noise_floor(config: dict, emit: Emit, cancelled: threading.Event,
                     games_out: list[dict]) -> dict:
    agent = config.get("agent", "heuristic")
    scenarios = resolve_scenarios(config.get("scenarios", "all"))
    pairs = int(config.get("pairs", 100))
    base_seed = int(config.get("seed", 1))
    from os import cpu_count
    workers = int(config.get("workers", 0)) or (cpu_count() or 1)

    rows: list[dict] = []
    all_pair: list[float] = []
    all_game: list[float] = []
    total = len(scenarios) * pairs
    completed = 0
    t0 = time.perf_counter()
    for sc in scenarios:
        tasks = [(agent, agent, sc.to_dict(), derive_seed(base_seed, sc.name, i))
                 for i in range(pairs)]
        pair_scores: list[float] = []
        game_scores: list[float] = []
        for g1, g2, score in stream_pairs(tasks, workers, cancelled):
            games_out.extend((g1, g2))
            pair_scores.append(score)
            game_scores.extend((_game_score0(g1), _game_score0(g2)))
            completed += 1
            if completed % 10 == 0 or completed == total:
                emit("progress", {"completed": completed, "total": total,
                                  "elapsed": time.perf_counter() - t0})
        if cancelled.is_set():
            raise JobCancelled
        all_pair.extend(pair_scores)
        all_game.extend(game_scores)
        gm, gci = mean_ci95(game_scores)
        pm, pci = mean_ci95(pair_scores)
        rows.append({"scenario": sc.name, "game_mean": gm,
                     "game_ci": _finite(gci), "pair_mean": pm,
                     "pair_ci": _finite(pci), "n": len(pair_scores)})
        emit("row", rows[-1])
        emit("log", {"line": f"{sc.name}: game-level={gm:.4f}+/-{gci:.4f} "
                             f"(dev {gm - 0.5:+.4f})  pair-level={pm:.4f}+/-{pci:.4f}"})
    gm, gci = mean_ci95(all_game)
    pm, pci = mean_ci95(all_pair)
    return {"agent": agent, "rows": rows,
            "overall": {"game_mean": gm, "game_ci": _finite(gci),
                        "pair_mean": pm, "pair_ci": _finite(pci)},
            "duration": time.perf_counter() - t0}


def _run_play(config: dict, emit: Emit, cancelled: threading.Event,
              games_out: list[dict]) -> dict:
    from tactica.eval.runner import run_match
    from tactica.scenario import load_scenario
    scenario = load_scenario(config.get("scenario", "open_field"),
                             deterministic=config.get("deterministic") or None)
    seed = int(config.get("seed", 1))
    p0, p1 = config["p0"], config["p1"]
    record = run_match(p0, p1, scenario, seed)
    games_out.append(record.to_dict())
    outcome = ("draw" if record.winner is None
               else f"side {record.winner} ({record.specs[record.winner]}) wins")
    emit("log", {"line": f"{scenario.name} seed={seed}: {outcome} after "
                         f"{record.rounds} rounds, hash={record.state_hash}"})
    return {"scenario": scenario.name, "seed": seed, "specs": [p0, p1],
            "winner": record.winner, "rounds": record.rounds,
            "state_hash": record.state_hash, "n_actions": len(record.actions)}


RUNNERS: dict[str, Callable[[dict, Emit, threading.Event, list[dict]], dict]] = {
    "tournament": _run_tournament,
    "sprt": _run_sprt,
    "skill-curve": _run_skill_curve,
    "noise-floor": _run_noise_floor,
    "play": _run_play,
}


class JobCancelled(Exception):
    pass


# ----------------------------------------------------------------------- #
# Job manager


@dataclass
class Job:
    id: str
    kind: str
    config: dict
    status: str = "queued"
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    games: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    subscribers: list[queue.SimpleQueue] = field(default_factory=list)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "config": self.config,
                "status": self.status, "error": self.error,
                "created": self.created, "finished": self.finished,
                "n_games": len(self.games)}

    def info(self) -> dict:
        return {**self.summary(), "result": self.result}


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, config: dict) -> Job:
        if kind not in RUNNERS:
            raise ValueError(f"unknown job kind {kind!r}; "
                             f"valid: {', '.join(RUNNERS)}")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, config=config)
        with self._lock:
            self.jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise KeyError(f"no such job {job_id!r}") from None

    def list(self) -> list[dict]:
        return [j.summary() for j in
                sorted(self.jobs.values(), key=lambda j: -j.created)]

    def cancel(self, job_id: str) -> None:
        self.get(job_id).cancel_flag.set()

    # ------------------------------------------------------------------ #

    def _emit(self, job: Job, etype: str, data: dict) -> None:
        event = {"type": etype, "data": data}
        with job.lock:
            job.events.append(event)
            subs = list(job.subscribers)
        for q in subs:
            q.put(event)

    def _set_status(self, job: Job, status: str) -> None:
        job.status = status
        if status in TERMINAL:
            job.finished = time.time()
        self._emit(job, "status", {"status": status, "error": job.error,
                                   "result": job.result})

    def _run(self, job: Job) -> None:
        self._set_status(job, "running")
        emit = lambda etype, data: self._emit(job, etype, data)  # noqa: E731
        try:
            job.result = RUNNERS[job.kind](job.config, emit,
                                           job.cancel_flag, job.games)
            self._set_status(job, "done")
        except JobCancelled:
            self._set_status(job, "cancelled")
        except Exception as exc:  # surface anything to the dashboard
            job.error = f"{type(exc).__name__}: {exc}"
            self._set_status(job, "failed")

    def subscribe(self, job_id: str) -> Iterable[dict]:
        """Yield all past events, then live ones until the job reaches a
        terminal status. Includes periodic ``ping`` events as keepalives."""
        job = self.get(job_id)
        q: queue.SimpleQueue = queue.SimpleQueue()
        with job.lock:
            history = list(job.events)
            job.subscribers.append(q)
        try:
            terminal_seen = False
            for event in history:
                yield event
                if event["type"] == "status" and event["data"]["status"] in TERMINAL:
                    terminal_seen = True
            while not terminal_seen:
                try:
                    event = q.get(timeout=15.0)
                except queue.Empty:
                    yield {"type": "ping", "data": {}}
                    continue
                yield event
                if event["type"] == "status" and event["data"]["status"] in TERMINAL:
                    terminal_seen = True
        finally:
            with job.lock:
                if q in job.subscribers:
                    job.subscribers.remove(q)
