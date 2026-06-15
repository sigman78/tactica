"""Unit type definitions and the stats table."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto


class UnitType(IntEnum):
    PIKEMAN = 0
    ARCHER = 1
    GRIFFIN = 2
    SWORDSMAN = 3
    CAVALRY = 4


class Perk(Enum):
    """Named unit specials. Each perk is declared here as data and has
    exactly one implementation point in the engine (battle.py)."""
    CHARGE = auto()         # melee dmg x2 when the strike moved >= 2 cells
    MELEE_PENALTY = auto()  # any melee strike (attack or retaliation) at x0.5


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
    perks: frozenset[Perk] = field(default_factory=frozenset)

    @property
    def avg_dmg(self) -> float:
        return (self.dmg_min + self.dmg_max) / 2.0

    @property
    def value(self) -> float:
        """Rough per-creature material value used by MCTS eval and agents.

        TODO(ideas): perks are invisible here -- CHARGE makes cavalry worth
        more and MELEE_PENALTY makes archers worth less in melee-heavy spots.
        See TODO.md.
        """
        return self.hp * 0.5 + self.avg_dmg * (1.0 + 0.05 * self.attack) * 2.0


# Numbers tuned for rough usefulness parity, not balance.
STATS: dict[UnitType, UnitStats] = {
    UnitType.PIKEMAN: UnitStats("Pikeman", 4, 4, 5, 1, 3, 10, False, False),
    UnitType.ARCHER: UnitStats("Archer", 4, 6, 3, 2, 3, 10, True, False,
                               perks=frozenset({Perk.MELEE_PENALTY})),
    UnitType.GRIFFIN: UnitStats("Griffin", 6, 8, 8, 3, 6, 25, False, True),
    UnitType.SWORDSMAN: UnitStats("Swordsman", 5, 10, 12, 6, 9, 35, False, False),
    UnitType.CAVALRY: UnitStats("Cavalry", 7, 15, 15, 10, 15, 50, False, False,
                                perks=frozenset({Perk.CHARGE})),
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
