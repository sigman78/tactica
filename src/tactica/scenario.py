"""Scenario definitions: army placements, obstacles, built-in maps, JSON I/O."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tactica.actions import BOARD_H, BOARD_W, xy_cell
from tactica.units import UNIT_BY_NAME, UnitType


@dataclass(frozen=True)
class ArmySlot:
    unit_type: UnitType
    count: int
    start_cell: int


@dataclass(frozen=True)
class Scenario:
    name: str
    army0: tuple[ArmySlot, ...]
    army1: tuple[ArmySlot, ...]
    obstacles: frozenset[int] = field(default_factory=frozenset)
    deterministic: bool = False

    def __post_init__(self) -> None:
        occupied: set[int] = set(self.obstacles)
        for slot in (*self.army0, *self.army1):
            if not 0 <= slot.start_cell < BOARD_W * BOARD_H:
                raise ValueError(f"{self.name}: cell {slot.start_cell} off board")
            if slot.count < 1:
                raise ValueError(f"{self.name}: stack count must be >= 1")
            if slot.start_cell in occupied:
                raise ValueError(f"{self.name}: cell {slot.start_cell} double-occupied")
            occupied.add(slot.start_cell)
        if not self.army0 or not self.army1:
            raise ValueError(f"{self.name}: both sides need at least one stack")

    def with_deterministic(self, deterministic: bool) -> "Scenario":
        return Scenario(self.name, self.army0, self.army1, self.obstacles, deterministic)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "army0": [[s.unit_type.name, s.count, s.start_cell] for s in self.army0],
            "army1": [[s.unit_type.name, s.count, s.start_cell] for s in self.army1],
            "obstacles": sorted(self.obstacles),
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        def slots(rows: list) -> tuple[ArmySlot, ...]:
            out = []
            for unit, count, cell in rows:
                ut = UnitType[unit] if unit in UnitType.__members__ else UNIT_BY_NAME[unit.lower()]
                out.append(ArmySlot(ut, int(count), int(cell)))
            return tuple(out)

        return cls(
            name=d["name"],
            army0=slots(d["army0"]),
            army1=slots(d["army1"]),
            obstacles=frozenset(int(c) for c in d.get("obstacles", [])),
            deterministic=bool(d.get("deterministic", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _mirror_x(cell_x: int) -> int:
    return BOARD_W - 1 - cell_x


def _symmetric(name: str, slots: list[tuple[UnitType, int, int, int]],
               obstacles: frozenset[int] = frozenset()) -> Scenario:
    """Build a mirror-symmetric scenario from side-0 slots given as (type, count, x, y)."""
    army0 = tuple(ArmySlot(t, n, xy_cell(x, y)) for t, n, x, y in slots)
    army1 = tuple(ArmySlot(t, n, xy_cell(_mirror_x(x), y)) for t, n, x, y in slots)
    return Scenario(name, army0, army1, obstacles)


def _row_cells(x: int, ys: list[int]) -> list[int]:
    return [xy_cell(x, y) for y in ys]


P, A, G, S, C = (UnitType.PIKEMAN, UnitType.ARCHER, UnitType.GRIFFIN,
                 UnitType.SWORDSMAN, UnitType.CAVALRY)

# --- 3 mirror-symmetric scenarios ---------------------------------------

OPEN_FIELD = _symmetric("open_field", [
    (A, 12, 0, 1), (P, 20, 0, 3), (S, 8, 0, 4), (P, 20, 0, 5), (G, 6, 0, 7),
])

CHOKEPOINT = _symmetric(
    "chokepoint",
    [(A, 10, 0, 2), (S, 8, 0, 4), (C, 4, 0, 6)],
    obstacles=frozenset(_row_cells(5, [0, 1, 2, 3, 6, 7, 8])),
)

SKIRMISH = _symmetric("skirmish", [
    (G, 5, 0, 2), (S, 10, 0, 4), (G, 5, 0, 6),
])

# --- 3 asymmetric scenarios ----------------------------------------------

# Shooters dug in behind pikemen vs a cavalry rush.
ARCHERS_VS_CAVALRY = Scenario(
    "archers_vs_cavalry",
    army0=(
        ArmySlot(A, 15, xy_cell(0, 2)),
        ArmySlot(A, 15, xy_cell(0, 6)),
        ArmySlot(P, 25, xy_cell(1, 3)),
        ArmySlot(P, 25, xy_cell(1, 5)),
    ),
    army1=(
        ArmySlot(C, 7, xy_cell(10, 2)),
        ArmySlot(C, 7, xy_cell(10, 6)),
    ),
    obstacles=frozenset([xy_cell(5, 4), xy_cell(6, 3), xy_cell(6, 5)]),
)

# A flying wing against a slow ground line.
GRIFFIN_RAID = Scenario(
    "griffin_raid",
    army0=(
        ArmySlot(G, 10, xy_cell(0, 3)),
        ArmySlot(G, 10, xy_cell(0, 5)),
        ArmySlot(A, 8, xy_cell(0, 4)),
    ),
    army1=(
        ArmySlot(S, 9, xy_cell(10, 2)),
        ArmySlot(P, 30, xy_cell(10, 4)),
        ArmySlot(S, 9, xy_cell(10, 6)),
    ),
    obstacles=frozenset([xy_cell(4, 1), xy_cell(4, 4), xy_cell(4, 7),
                         xy_cell(7, 2), xy_cell(7, 6)]),
)

# Elite few vs numerous chaff.
LAST_STAND = Scenario(
    "last_stand",
    army0=(
        ArmySlot(C, 6, xy_cell(0, 3)),
        ArmySlot(S, 9, xy_cell(0, 5)),
    ),
    army1=(
        ArmySlot(P, 40, xy_cell(10, 1)),
        ArmySlot(P, 40, xy_cell(10, 3)),
        ArmySlot(A, 14, xy_cell(10, 5)),
        ArmySlot(P, 40, xy_cell(10, 7)),
    ),
)

BUILTIN_SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (OPEN_FIELD, CHOKEPOINT, SKIRMISH,
              ARCHERS_VS_CAVALRY, GRIFFIN_RAID, LAST_STAND)
}

SYMMETRIC_SCENARIOS = ("open_field", "chokepoint", "skirmish")


def load_scenario(name_or_path: str, deterministic: bool | None = None) -> Scenario:
    """Load a scenario by built-in name or from a JSON file path."""
    if name_or_path in BUILTIN_SCENARIOS:
        sc = BUILTIN_SCENARIOS[name_or_path]
    elif Path(name_or_path).is_file():
        sc = Scenario.from_json(name_or_path)
    else:
        raise ValueError(
            f"unknown scenario {name_or_path!r}; built-ins: {', '.join(BUILTIN_SCENARIOS)}"
        )
    if deterministic is not None:
        sc = sc.with_deterministic(deterministic)
    return sc


def resolve_scenarios(spec: str, deterministic: bool | None = None) -> list[Scenario]:
    """Resolve a CLI scenario spec: 'all', a name, a path, or a comma list."""
    if spec == "all":
        names = list(BUILTIN_SCENARIOS)
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    return [load_scenario(n, deterministic) for n in names]
