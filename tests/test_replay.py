"""Replay round-trip and JSONL logging tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from tactica.eval.runner import (
    GameRecord,
    derive_seed,
    read_jsonl,
    replay_game,
    run_match,
    run_mirrored_pair,
    write_jsonl,
)
from tactica.scenario import BUILTIN_SCENARIOS, Scenario


def test_derive_seed_is_stable_and_spread() -> None:
    assert derive_seed(1, "open_field", 0) == derive_seed(1, "open_field", 0)
    seeds = {derive_seed(1, "open_field", i) for i in range(100)}
    assert len(seeds) == 100


def test_replay_round_trip(tmp_path: Path) -> None:
    record = run_match("heuristic", "random", BUILTIN_SCENARIOS["open_field"], 42)
    path = tmp_path / "game.jsonl"
    write_jsonl(path, [record.to_dict()])
    loaded = GameRecord.from_dict(read_jsonl(path)[0])
    assert loaded == record
    battle = replay_game(loaded)
    assert battle.is_terminal()
    assert battle.winner() == record.winner
    assert battle.state_hash() == record.state_hash


def test_replay_detects_corruption() -> None:
    record = run_match("random", "random", BUILTIN_SCENARIOS["skirmish"], 7)
    record.state_hash = "0" * 16
    with pytest.raises(AssertionError, match="replay diverged"):
        replay_game(record)


def test_replay_works_from_scenario_dict_alone() -> None:
    # Replays must not depend on built-in scenario definitions staying put.
    record = run_match("heuristic", "heuristic",
                       BUILTIN_SCENARIOS["chokepoint"], 3)
    rebuilt = Scenario.from_dict(record.scenario)
    assert rebuilt == BUILTIN_SCENARIOS["chokepoint"]


def test_mirrored_pair_swaps_sides() -> None:
    g1, g2, score = run_mirrored_pair("heuristic", "random",
                                      BUILTIN_SCENARIOS["open_field"], 5)
    assert g1.specs == ("heuristic", "random")
    assert g2.specs == ("random", "heuristic")
    assert g1.seed == g2.seed
    assert 0.0 <= score <= 1.0


def test_scenario_json_round_trip(tmp_path: Path) -> None:
    sc = BUILTIN_SCENARIOS["archers_vs_cavalry"]
    path = tmp_path / "custom.json"
    sc.to_json(path)
    assert Scenario.from_json(path) == sc


def test_records_carry_rules_version() -> None:
    from tactica.battle import RULES_VERSION
    from tactica.eval.runner import GameRecord, run_match
    from tactica.scenario import BUILTIN_SCENARIOS

    rec = run_match("heuristic", "random", BUILTIN_SCENARIOS["skirmish"], seed=1)
    assert rec.rules_version == RULES_VERSION
    # round-trips through dict, and old records (no field) default to 1
    assert GameRecord.from_dict(rec.to_dict()).rules_version == RULES_VERSION
    legacy = rec.to_dict()
    del legacy["rules_version"]
    assert GameRecord.from_dict(legacy).rules_version == 1
