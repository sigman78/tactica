"""Action encoding tests."""
import pytest

from tactica.actions import (
    N_ACTIONS,
    N_CELLS,
    Action,
    ActionType,
    cell_xy,
    xy_cell,
)


def test_action_space_size() -> None:
    assert N_CELLS == 11 * 9
    assert N_ACTIONS == 5 * N_CELLS


def test_canonical_ids_round_trip() -> None:
    for t in (ActionType.MOVE, ActionType.MELEE_ATTACK, ActionType.RANGED_ATTACK):
        for cell in range(N_CELLS):
            a = Action(t, cell)
            assert a.id == int(t) * N_CELLS + cell
            assert Action.from_id(a.id) == a
    for t in (ActionType.WAIT, ActionType.DEFEND):
        a = Action(t)
        assert a.id == int(t) * N_CELLS
        assert Action.from_id(a.id) == a


def test_wait_defend_ignore_cell() -> None:
    assert Action(ActionType.WAIT, 42) == Action(ActionType.WAIT, 0)
    assert Action(ActionType.DEFEND, 7).id == int(ActionType.DEFEND) * N_CELLS


def test_from_id_bounds() -> None:
    with pytest.raises(ValueError):
        Action.from_id(-1)
    with pytest.raises(ValueError):
        Action.from_id(N_ACTIONS)


def test_from_id_rejects_non_canonical_wait_defend() -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        Action.from_id(int(ActionType.WAIT) * N_CELLS + 5)
    with pytest.raises(ValueError, match="non-canonical"):
        Action.from_id(int(ActionType.DEFEND) * N_CELLS + 98)


def test_cell_xy_round_trip() -> None:
    for cell in range(N_CELLS):
        x, y = cell_xy(cell)
        assert xy_cell(x, y) == cell
