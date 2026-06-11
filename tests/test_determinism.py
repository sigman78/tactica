"""Determinism: identical seeds and action lists reproduce identical states;
clones are independent of their originals."""
from __future__ import annotations

import numpy as np

from tactica.actions import Action
from tactica.battle import Battle
from tactica.scenario import BUILTIN_SCENARIOS


def random_action_list(seed: int, scenario_name: str = "open_field") -> list[int]:
    b = Battle.from_scenario(BUILTIN_SCENARIOS[scenario_name], seed)
    rng = np.random.Generator(np.random.PCG64(seed + 5000))
    actions = []
    while not b.is_terminal():
        legal = b.legal_actions()
        a = legal[int(rng.integers(len(legal)))]
        b.step(a)
        actions.append(a.id)
    return actions


def test_same_seed_same_actions_same_hash() -> None:
    actions = random_action_list(11)
    hashes = []
    for _ in range(2):
        b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 11)
        mid_hashes = []
        for action_id in actions:
            b.step(Action.from_id(action_id))
            mid_hashes.append(b.state_hash())
        hashes.append(tuple(mid_hashes))
    assert hashes[0] == hashes[1]


def test_different_seeds_diverge() -> None:
    # Stochastic damage: the same action prefix yields different states.
    b1 = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 1)
    b2 = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 2)
    assert b1.state_hash() != b2.state_hash()  # rng state differs at least


def test_deterministic_mode_has_no_chance() -> None:
    sc = BUILTIN_SCENARIOS["open_field"].with_deterministic(True)
    # Different seeds, same actions -> same battle outcome (only the
    # initiative tie-shuffle and rng bookkeeping differ; compare stacks).
    actions = None
    finals = []
    for seed in (1, 99):
        b = Battle.from_scenario(sc, 42)  # same seed: identical tiebreaks
        b.reseed(seed)  # diverge the rng stream only
        if actions is None:
            rng = np.random.Generator(np.random.PCG64(0))
            actions = []
            while not b.is_terminal() and len(actions) < 60:
                legal = b.legal_actions()
                a = legal[int(rng.integers(len(legal)))]
                b.step(a)
                actions.append(a.id)
        else:
            for action_id in actions:
                b.step(Action.from_id(action_id))
        finals.append([(s.count, s.top_hp, s.cell)
                       for s in b.stacks.values()])
    assert finals[0] == finals[1]


def test_clone_divergence_leaves_original_intact() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["skirmish"], 5)
    for _ in range(4):
        b.step(b.legal_actions()[0])
    before = b.state_hash()
    legal_before = [a.id for a in b.legal_actions()]

    c = b.clone()
    assert c.state_hash() == before
    rng = np.random.Generator(np.random.PCG64(123))
    while not c.is_terminal():
        legal = c.legal_actions()
        c.step(legal[int(rng.integers(len(legal)))])

    assert b.state_hash() == before
    assert [a.id for a in b.legal_actions()] == legal_before
    assert not b.is_terminal()


def test_clone_plays_identically_without_divergence() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 8)
    c = b.clone()
    rng1 = np.random.Generator(np.random.PCG64(77))
    rng2 = np.random.Generator(np.random.PCG64(77))
    for battle, rng in ((b, rng1), (c, rng2)):
        for _ in range(40):
            if battle.is_terminal():
                break
            legal = battle.legal_actions()
            battle.step(legal[int(rng.integers(len(legal)))])
    assert b.state_hash() == c.state_hash()
