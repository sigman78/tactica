"""Game runner: seeded matches, mirrored pairs, JSONL logging, replay.

Seed discipline: ``derive_seed`` is a *stable* hash (sha256), so seeds are
reproducible across processes and Python versions (the builtin ``hash`` is
salted per process and must not be used). Common random numbers: pair ``i``
of every matchup on scenario ``s`` uses ``derive_seed(base_seed, s, i)``, so
all agent pairs face identical battle RNG streams and comparisons are paired.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from tactica.actions import Action
from tactica.agents import Agent, make_agent
from tactica.battle import Battle
from tactica.scenario import Scenario

MAX_SEED = 2**63


def derive_seed(*parts: object) -> int:
    """Stable, process-independent seed derivation from arbitrary parts."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % MAX_SEED


@dataclass
class GameRecord:
    scenario_name: str
    scenario: dict
    seed: int
    specs: tuple[str, str]
    agents: tuple[dict, dict]
    actions: list[int]
    winner: int | None
    rounds: int
    state_hash: str

    def score(self, side: int) -> float:
        """1 / 0.5 / 0 for the given side."""
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == side else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GameRecord":
        return cls(
            scenario_name=d["scenario_name"],
            scenario=d["scenario"],
            seed=int(d["seed"]),
            specs=tuple(d["specs"]),
            agents=tuple(d["agents"]),
            actions=list(d["actions"]),
            winner=d["winner"],
            rounds=int(d["rounds"]),
            state_hash=d["state_hash"],
        )


def play_game(
    scenario: Scenario,
    seed: int,
    agents: tuple[Agent, Agent],
    specs: tuple[str, str] = ("?", "?"),
    on_step: Callable[[Battle, Action], None] | None = None,
) -> GameRecord:
    battle = Battle.from_scenario(scenario, seed)
    actions: list[int] = []
    while not battle.is_terminal():
        action = agents[battle.current_player()].act(battle)
        battle.step(action)
        actions.append(action.id)
        if on_step is not None:
            on_step(battle, action)
    return GameRecord(
        scenario_name=scenario.name,
        scenario=scenario.to_dict(),
        seed=seed,
        specs=specs,
        agents=(agents[0].config(), agents[1].config()),
        actions=actions,
        winner=battle.winner(),
        rounds=battle.round,
        state_hash=battle.state_hash(),
    )


def run_match(spec0: str, spec1: str, scenario: Scenario, seed: int,
              on_step: Callable[[Battle, Action], None] | None = None,
              agent_salt: int = 0) -> GameRecord:
    """Play one game with freshly built, deterministically seeded agents.
    ``agent_salt`` decorrelates agent RNG streams between the two games of a
    mirrored pair; without it, stochastic self-play pairs would be clones."""
    agents = (make_agent(spec0, seed=derive_seed(seed, agent_salt, 0, spec0)),
              make_agent(spec1, seed=derive_seed(seed, agent_salt, 1, spec1)))
    return play_game(scenario, seed, agents, (spec0, spec1), on_step)


def run_mirrored_pair(spec_a: str, spec_b: str, scenario: Scenario,
                      seed: int) -> tuple[GameRecord, GameRecord, float]:
    """Two games on the same scenario+seed with sides swapped.
    Returns (game_ab, game_ba, pair_score_for_a)."""
    g1 = run_match(spec_a, spec_b, scenario, seed, agent_salt=0)
    g2 = run_match(spec_b, spec_a, scenario, seed, agent_salt=1)
    score_a = (g1.score(0) + g2.score(1)) / 2.0
    return g1, g2, score_a


def _pair_task(args: tuple[str, str, dict, int]) -> tuple[dict, dict, float]:
    """Top-level worker (picklable on Windows spawn) for one mirrored pair."""
    spec_a, spec_b, scenario_dict, seed = args
    scenario = Scenario.from_dict(scenario_dict)
    g1, g2, score = run_mirrored_pair(spec_a, spec_b, scenario, seed)
    return g1.to_dict(), g2.to_dict(), score


def run_pairs(tasks: list[tuple[str, str, dict, int]],
              workers: int = 1) -> Iterable[tuple[dict, dict, float]]:
    """Run mirrored-pair tasks, optionally across processes. Results stream
    back in task order regardless of worker count."""
    if workers <= 1:
        for t in tasks:
            yield _pair_task(t)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_pair_task, tasks, chunksize=1)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_game(record: GameRecord,
                on_step: Callable[[Battle, Action], None] | None = None) -> Battle:
    """Re-simulate a logged game; raises if the final state hash differs."""
    scenario = Scenario.from_dict(record.scenario)
    battle = Battle.from_scenario(scenario, record.seed)
    for action_id in record.actions:
        action = Action.from_id(action_id)
        battle.step(action)
        if on_step is not None:
            on_step(battle, action)
    final = battle.state_hash()
    if final != record.state_hash:
        raise AssertionError(
            f"replay diverged: hash {final} != logged {record.state_hash}")
    return battle
