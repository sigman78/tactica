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
    MELEE_ATTACK = 1
    RANGED_ATTACK = 2
    WAIT = 3
    DEFEND = 4


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
