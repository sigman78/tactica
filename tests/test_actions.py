"""Action encoding tests."""
import pytest

from tactica.actions import (
    MELEE_TYPES,
    N_ACTIONS,
    N_CELLS,
    Action,
    ActionType,
    cell_xy,
    is_melee,
    melee_offset,
    melee_type_for_offset,
    xy_cell,
)


def test_action_space_size() -> None:
    assert N_CELLS == 11 * 9
    assert N_ACTIONS == 12 * N_CELLS


def test_eight_melee_types() -> None:
    assert len(MELEE_TYPES) == 8
    assert all(is_melee(t) for t in MELEE_TYPES)
    assert not is_melee(ActionType.MOVE)
    assert not is_melee(ActionType.RANGED_ATTACK)


def test_canonical_ids_round_trip() -> None:
    cell_types = (ActionType.MOVE, ActionType.RANGED_ATTACK, *MELEE_TYPES)
    for t in cell_types:
        for cell in range(N_CELLS):
            a = Action(t, cell)
            assert a.id == int(t) * N_CELLS + cell
            assert Action.from_id(a.id) == a
    for t in (ActionType.WAIT, ActionType.DEFEND):
        a = Action(t)
        assert a.id == int(t) * N_CELLS
        assert Action.from_id(a.id) == a


def test_melee_offset_inverse() -> None:
    seen = set()
    for t in MELEE_TYPES:
        dx, dy = melee_offset(t)
        assert (dx, dy) != (0, 0)
        assert -1 <= dx <= 1 and -1 <= dy <= 1
        assert melee_type_for_offset(dx, dy) == t
        seen.add((dx, dy))
    assert len(seen) == 8  # all eight neighbors, distinct


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
