"""Agent tests: factory specs, epsilon behavior, MCTS sanity, and the
heuristic-beats-random strength smoke test."""
from __future__ import annotations

import numpy as np
import pytest

from tactica.actions import Action, ActionType, is_melee, xy_cell
from tactica.agents import (
    EpsilonAgent,
    HeuristicAgent,
    MCTSAgent,
    RandomAgent,
    WeightedAgent,
    make_agent,
)
from tactica.battle import Battle
from tactica.eval.runner import derive_seed, run_mirrored_pair
from tactica.scenario import BUILTIN_SCENARIOS, ArmySlot, Scenario
from tactica.units import UnitType


def test_factory_specs() -> None:
    assert isinstance(make_agent("random"), RandomAgent)
    assert isinstance(make_agent("heuristic"), HeuristicAgent)
    assert isinstance(make_agent("weighted"), WeightedAgent)
    mcts = make_agent("mcts:10:2.0")
    assert isinstance(mcts, MCTSAgent)
    assert (mcts.simulations, mcts.c_uct) == (10, 2.0)
    eps = make_agent("epsilon:0.25:heuristic")
    assert isinstance(eps, EpsilonAgent)
    assert eps.epsilon == 0.25
    assert isinstance(eps.inner, HeuristicAgent)
    with pytest.raises(ValueError):
        make_agent("alphazero")
    with pytest.raises(ValueError):
        make_agent("epsilon:0.5")


def test_agents_only_pick_legal_actions() -> None:
    for spec in ("random", "heuristic", "weighted", "epsilon:0.3:heuristic",
                 "mcts:8"):
        b = Battle.from_scenario(BUILTIN_SCENARIOS["archers_vs_cavalry"], 4)
        agent = make_agent(spec, seed=1)
        for _ in range(12):
            if b.is_terminal():
                break
            action = agent.act(b)
            assert b.legal_action_mask()[action.id]
            b.step(action)


def test_epsilon_zero_matches_inner() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 2)
    inner = HeuristicAgent()
    eps = EpsilonAgent(HeuristicAgent(), 0.0, seed=3)
    for _ in range(20):
        if b.is_terminal():
            break
        assert eps.act(b) == inner.act(b)
        b.step(inner.act(b))


def test_epsilon_one_is_random() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 2)
    eps = EpsilonAgent(HeuristicAgent(), 1.0, seed=3)
    picks = {eps.act(b).id for _ in range(30)}
    assert len(picks) > 3  # inner heuristic would always pick one action


def test_mcts_takes_a_winning_kill() -> None:
    sc = Scenario(
        "probe",
        army0=(ArmySlot(UnitType.CAVALRY, 5, xy_cell(5, 4)),),
        army1=(ArmySlot(UnitType.PIKEMAN, 3, xy_cell(6, 4)),),
    )
    b = Battle.from_scenario(sc, 1)
    while b.active_stack().unit_type != UnitType.CAVALRY:
        b.step(Action(ActionType.DEFEND))
    action = MCTSAgent(simulations=40, seed=1).act(b)
    assert action == Action(ActionType.MELEE_ATTACK, xy_cell(6, 4))


def test_mcts_handles_fewer_sims_than_actions() -> None:
    b = Battle.from_scenario(BUILTIN_SCENARIOS["open_field"], 6)
    agent = MCTSAgent(simulations=3, seed=2)
    action = agent.act(b)
    assert b.legal_action_mask()[action.id]


def test_weighted_rejects_unknown_feature(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_a_feature": 1.0}')
    with pytest.raises(ValueError, match="unknown weight"):
        WeightedAgent(bad)


def test_heuristic_beats_random_90pct_on_symmetric_map() -> None:
    """Strength smoke test from the spec: >=90% over 50 mirrored pairs."""
    scenario = BUILTIN_SCENARIOS["open_field"]
    scores = []
    for i in range(50):
        seed = derive_seed(0xBEEF, scenario.name, i)
        _, _, score = run_mirrored_pair("heuristic", "random", scenario, seed)
        scores.append(score)
    assert float(np.mean(scores)) >= 0.9


def test_heuristic_emits_legal_directional_melee() -> None:
    sc = Scenario(
        "probe",
        army0=(ArmySlot(UnitType.SWORDSMAN, 5, xy_cell(4, 4)),),
        army1=(ArmySlot(UnitType.PIKEMAN, 5, xy_cell(5, 4)),),
    )
    b = Battle.from_scenario(sc, 1)
    while b.active_stack().unit_type != UnitType.SWORDSMAN:
        b.step(Action(ActionType.DEFEND))
    action = HeuristicAgent().act(b)
    assert is_melee(action.type)
    assert action.target_cell == xy_cell(5, 4)
    assert b.legal_action_mask()[action.id]


def test_weighted_picks_legal_action_with_directional_melee() -> None:
    sc = Scenario(
        "probe",
        army0=(ArmySlot(UnitType.SWORDSMAN, 5, xy_cell(4, 4)),),
        army1=(ArmySlot(UnitType.PIKEMAN, 5, xy_cell(5, 4)),),
    )
    b = Battle.from_scenario(sc, 1)
    while b.active_stack().unit_type != UnitType.SWORDSMAN:
        b.step(Action(ActionType.DEFEND))
    action = WeightedAgent().act(b)
    assert b.legal_action_mask()[action.id]
