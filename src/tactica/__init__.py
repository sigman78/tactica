"""tactica: a HoMM-style tactics AI sandbox."""
from tactica.actions import Action, ActionType, BOARD_H, BOARD_W, N_ACTIONS
from tactica.battle import Battle
from tactica.scenario import Scenario, load_scenario
from tactica.units import STATS, UnitType

__all__ = [
    "Action", "ActionType", "BOARD_H", "BOARD_W", "N_ACTIONS",
    "Battle", "Scenario", "load_scenario", "STATS", "UnitType",
]
__version__ = "0.1.0"
