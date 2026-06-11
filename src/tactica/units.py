"""Unit type definitions and the stats table."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class UnitType(IntEnum):
    PIKEMAN = 0
    ARCHER = 1
    GRIFFIN = 2
    SWORDSMAN = 3
    CAVALRY = 4


@dataclass(frozen=True)
class UnitStats:
    name: str
    speed: int
    attack: int
    defense: int
    dmg_min: int
    dmg_max: int
    hp: int
    is_ranged: bool
    is_flyer: bool
    initiative: int

    @property
    def avg_dmg(self) -> float:
        return (self.dmg_min + self.dmg_max) / 2.0

    @property
    def value(self) -> float:
        """Rough per-creature material value used by MCTS eval and agents."""
        return self.hp * 0.5 + self.avg_dmg * (1.0 + 0.05 * self.attack) * 2.0


# Numbers tuned for rough usefulness parity, not balance.
STATS: dict[UnitType, UnitStats] = {
    UnitType.PIKEMAN: UnitStats("Pikeman", 4, 4, 5, 1, 3, 10, False, False, 4),
    UnitType.ARCHER: UnitStats("Archer", 4, 6, 3, 2, 3, 10, True, False, 5),
    UnitType.GRIFFIN: UnitStats("Griffin", 6, 8, 8, 3, 6, 25, False, True, 7),
    UnitType.SWORDSMAN: UnitStats("Swordsman", 5, 10, 12, 6, 9, 35, False, False, 5),
    UnitType.CAVALRY: UnitStats("Cavalry", 7, 15, 15, 10, 15, 50, False, False, 8),
}

UNIT_BY_NAME: dict[str, UnitType] = {s.name.lower(): t for t, s in STATS.items()}

# Single ASCII letter per unit type for board rendering.
GLYPHS: dict[UnitType, str] = {
    UnitType.PIKEMAN: "P",
    UnitType.ARCHER: "A",
    UnitType.GRIFFIN: "G",
    UnitType.SWORDSMAN: "S",
    UnitType.CAVALRY: "C",
}
