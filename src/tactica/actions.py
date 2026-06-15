"""Flat fixed-size action space.

``action_id = action_type * (W * H) + target_cell``

WAIT and DEFEND ignore the cell index; their canonical encoding uses
``target_cell = 0`` and only that single id is ever legal/masked.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

BOARD_W = 11
BOARD_H = 9
N_CELLS = BOARD_W * BOARD_H


class ActionType(IntEnum):
    MOVE = 0
    MELEE_N = 1
    MELEE_NE = 2
    MELEE_E = 3
    MELEE_SE = 4
    MELEE_S = 5
    MELEE_SW = 6
    MELEE_W = 7
    MELEE_NW = 8
    RANGED_ATTACK = 9
    WAIT = 10
    DEFEND = 11


# Approach cell offset (dx, dy) relative to the TARGET cell, per melee
# direction. MELEE_W means the attacker stands west of the target and
# strikes east. Order matches the 8-neighborhood.
MELEE_OFFSETS: dict[ActionType, tuple[int, int]] = {
    ActionType.MELEE_N: (0, -1),
    ActionType.MELEE_NE: (1, -1),
    ActionType.MELEE_E: (1, 0),
    ActionType.MELEE_SE: (1, 1),
    ActionType.MELEE_S: (0, 1),
    ActionType.MELEE_SW: (-1, 1),
    ActionType.MELEE_W: (-1, 0),
    ActionType.MELEE_NW: (-1, -1),
}
MELEE_TYPES: tuple[ActionType, ...] = tuple(MELEE_OFFSETS)
_OFFSET_TO_MELEE: dict[tuple[int, int], ActionType] = {
    off: t for t, off in MELEE_OFFSETS.items()
}


def is_melee(t: ActionType) -> bool:
    return t in MELEE_OFFSETS


def melee_offset(t: ActionType) -> tuple[int, int]:
    return MELEE_OFFSETS[t]


def melee_type_for_offset(dx: int, dy: int) -> ActionType:
    return _OFFSET_TO_MELEE[(dx, dy)]


N_ACTIONS = len(ActionType) * N_CELLS


@dataclass(frozen=True)
class Action:
    type: ActionType
    target_cell: int = 0

    def __post_init__(self) -> None:
        if self.type in (ActionType.WAIT, ActionType.DEFEND):
            object.__setattr__(self, "target_cell", 0)

    @property
    def id(self) -> int:
        return int(self.type) * N_CELLS + self.target_cell

    @classmethod
    def from_id(cls, action_id: int) -> "Action":
        if not 0 <= action_id < N_ACTIONS:
            raise ValueError(f"action id {action_id} out of range [0, {N_ACTIONS})")
        action_type = ActionType(action_id // N_CELLS)
        cell = action_id % N_CELLS
        if action_type in (ActionType.WAIT, ActionType.DEFEND) and cell != 0:
            raise ValueError(
                f"non-canonical id {action_id}: {action_type.name} ids must "
                f"use cell 0")
        return cls(action_type, cell)

    def __repr__(self) -> str:
        if self.type in (ActionType.WAIT, ActionType.DEFEND):
            return self.type.name
        x, y = self.target_cell % BOARD_W, self.target_cell // BOARD_W
        return f"{self.type.name}({x},{y})"


def cell_xy(cell: int) -> tuple[int, int]:
    return cell % BOARD_W, cell // BOARD_W


def xy_cell(x: int, y: int) -> int:
    return y * BOARD_W + x
