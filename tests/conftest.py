"""Shared fixtures and helpers for the tactica test suite."""
from __future__ import annotations

import pytest

from tactica.actions import xy_cell
from tactica.battle import Battle
from tactica.scenario import ArmySlot, Scenario
from tactica.units import UnitType


def duel_scenario(
    unit0: UnitType, count0: int, cell0: tuple[int, int],
    unit1: UnitType, count1: int, cell1: tuple[int, int],
    obstacles: frozenset[int] = frozenset(),
    deterministic: bool = True,
    name: str = "duel",
) -> Scenario:
    """Minimal two-stack scenario for rules tests."""
    return Scenario(
        name=name,
        army0=(ArmySlot(unit0, count0, xy_cell(*cell0)),),
        army1=(ArmySlot(unit1, count1, xy_cell(*cell1)),),
        obstacles=obstacles,
        deterministic=deterministic,
    )


def battle_of(scenario: Scenario, seed: int = 1) -> Battle:
    return Battle.from_scenario(scenario, seed)


@pytest.fixture
def open_field() -> Scenario:
    from tactica.scenario import OPEN_FIELD
    return OPEN_FIELD
