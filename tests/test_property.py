"""Property tests: seeded random playthroughs checking mask/legal agreement,
state validity, and termination."""
from __future__ import annotations

import numpy as np
import pytest

from tactica.actions import BOARD_H, BOARD_W, N_ACTIONS, Action
from tactica.battle import ROUND_LIMIT, Battle
from tactica.scenario import BUILTIN_SCENARIOS, Scenario


def assert_valid_state(b: Battle, scenario: Scenario) -> None:
    living = [s for s in b.stacks.values() if s.alive]
    cells = [s.cell for s in living]
    assert len(cells) == len(set(cells)), "two stacks share a cell"
    for s in living:
        assert 1 <= s.count
        assert 1 <= s.top_hp <= s.stats.hp
        assert 0 <= s.cell < BOARD_W * BOARD_H
        assert s.cell not in scenario.obstacles
        assert 0 <= s.retaliations_left <= 1
    assert 1 <= b.round <= ROUND_LIMIT


@pytest.mark.parametrize("scenario_name", list(BUILTIN_SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1])
def test_random_playthrough_invariants(scenario_name: str, seed: int) -> None:
    scenario = BUILTIN_SCENARIOS[scenario_name]
    b = Battle.from_scenario(scenario, seed)
    rng = np.random.Generator(np.random.PCG64(seed + 1000))
    steps = 0
    max_steps = ROUND_LIMIT * len(b.stacks) * 2 + 100
    while not b.is_terminal():
        legal = b.legal_actions()
        mask = b.legal_action_mask()
        # Mask and action list agree exactly.
        assert mask.shape == (N_ACTIONS,)
        ids = sorted(a.id for a in legal)
        assert len(set(ids)) == len(ids), "duplicate legal action ids"
        assert sorted(np.flatnonzero(mask).tolist()) == ids
        # Every sampled legal action steps without error.
        action = legal[int(rng.integers(len(legal)))]
        b.step(action)
        assert_valid_state(b, scenario)
        steps += 1
        assert steps <= max_steps, "game failed to terminate within round cap"
    # Terminal accounting is consistent.
    r0, r1 = b.returns()
    assert r0 == -r1 or (r0, r1) == (0.0, 0.0)


def test_exhaustive_action_id_sweep() -> None:
    """Every one of the 495 ids either decodes+steps (masked) or raises
    (unmasked or non-canonical), at several distinct game states."""
    b = Battle.from_scenario(BUILTIN_SCENARIOS["archers_vs_cavalry"], 13)
    rng = np.random.Generator(np.random.PCG64(2))
    for _ in range(6):
        mask = b.legal_action_mask()
        for action_id in range(N_ACTIONS):
            if mask[action_id]:
                clone = b.clone()
                clone.step(Action.from_id(action_id))
            else:
                with pytest.raises(ValueError):
                    b.step(Action.from_id(action_id))
        legal = b.legal_actions()
        b.step(legal[int(rng.integers(len(legal)))])


def test_run_pairs_parallel_matches_inline() -> None:
    from tactica.eval.runner import derive_seed, run_pairs

    tasks = [("heuristic", "random",
              BUILTIN_SCENARIOS["open_field"].to_dict(),
              derive_seed(0, "open_field", i)) for i in range(3)]
    inline = list(run_pairs(tasks, workers=1))
    parallel = list(run_pairs(tasks, workers=2))
    assert inline == parallel


def test_illegal_actions_raise() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 7)
    mask = b.legal_action_mask()
    illegal_ids = np.flatnonzero(~mask)
    rng = np.random.Generator(np.random.PCG64(0))
    for action_id in rng.choice(illegal_ids, size=25, replace=False):
        with pytest.raises(ValueError):
            b.step(Action.from_id(int(action_id)))


def test_playout_terminates_and_stays_legal() -> None:
    for seed in range(5):
        b = Battle.from_scenario(BUILTIN_SCENARIOS["chokepoint"], seed)
        steps = b.playout(max_steps=10_000)
        assert b.is_terminal()
        assert steps <= 10_000
        assert_valid_state(b, BUILTIN_SCENARIOS["chokepoint"])


def test_observe_shape_and_ranges() -> None:
    for name in ("open_field", "archers_vs_cavalry"):
        b = Battle.from_scenario(BUILTIN_SCENARIOS[name], 3)
        rng = np.random.Generator(np.random.PCG64(9))
        for _ in range(30):
            if b.is_terminal():
                break
            obs = b.observe()
            assert obs.shape == (18, BOARD_H, BOARD_W)
            assert obs.dtype == np.float32
            assert obs.min() >= 0.0
            assert obs.max() <= 1.0
            assert obs[14].sum() == 1.0  # exactly one active unit
            legal = b.legal_actions()
            b.step(legal[int(rng.integers(len(legal)))])
