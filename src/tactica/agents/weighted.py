"""Weight-vector agent: scores candidate actions via linear features.

Same feature set for every action; weights come from a JSON file. The shipped
defaults imitate :class:`HeuristicAgent` (focus fire, kiting, advancing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from tactica.actions import Action, ActionType, is_melee
from tactica.agents.base import Agent
from tactica.battle import (
    DAMAGE_MOD_MAX,
    DAMAGE_MOD_MIN,
    DAMAGE_MOD_PER_POINT,
    MELEE_PENALTY_FACTOR,
    Battle,
    Stack,
    chebyshev,
)
from tactica.agents.heuristic import stack_value
from tactica.units import Perk

FEATURES = (
    "damage_dealt",     # expected damage as a fraction of the target's HP pool
    "kill",             # 1 if the expected damage wipes the target stack
    "damage_received",  # expected retaliation as a fraction of own HP pool
    "target_value",     # target material value / strongest enemy value
    "focus_fire",       # 1 if the target has the lowest HP pool among enemies
    "dist_melee",       # melee units: distance to nearest enemy after acting
    "dist_ranged",      # ranged units: same distance feature (kiting)
    "wait",             # 1 for WAIT
    "defend",           # 1 for DEFEND
)

# Mirrors data/weights_default.json (the aggressive profile: SPRT-validated
# at ~+11.5 elo over the original heuristic-imitating weights, which live on
# as weights/conservative.json).
DEFAULT_WEIGHTS: dict[str, float] = {
    "damage_dealt": 8.0,
    "kill": 5.0,
    "damage_received": -2.0,
    "target_value": 1.0,
    "focus_fire": 1.0,
    "dist_melee": -1.5,
    "dist_ranged": 1.0,
    "wait": -0.10,
    "defend": 0.05,
}


def expected_damage(attacker: Stack, defender: Stack, melee: bool) -> float:
    """Mirror of Battle.compute_damage with the expected roll.

    TODO(ideas): Perk.CHARGE is not modelled -- the damage_dealt feature
    understates a charging cavalry strike. Needs a distance-aware feature
    set (see TODO.md) before the weights can learn it.
    """
    stats = attacker.stats
    base = stats.avg_dmg * attacker.count
    diff = stats.attack - defender.effective_defense()
    factor = min(max(1.0 + DAMAGE_MOD_PER_POINT * diff, DAMAGE_MOD_MIN),
                 DAMAGE_MOD_MAX)
    if melee and Perk.MELEE_PENALTY in stats.perks:
        factor *= MELEE_PENALTY_FACTOR
    return max(1.0, base * factor)


@dataclass
class TurnContext:
    """Per-turn shared state so feature extraction is O(1) per action."""
    stack: Stack
    reach: dict[int, int]
    enemies: dict[int, Stack]
    weakest_hp: int
    max_value: float

    @classmethod
    def from_battle(cls, battle: Battle) -> "TurnContext":
        s = battle.active_stack()
        enemies = {e.cell: e for e in battle.stacks.values()
                   if e.alive and e.side != s.side}
        return cls(
            stack=s,
            reach=battle.reachable(s),
            enemies=enemies,
            weakest_hp=min(e.total_hp for e in enemies.values()),
            max_value=max(stack_value(e) for e in enemies.values()),
        )


def action_features(battle: Battle, action: Action,
                    ctx: TurnContext | None = None) -> dict[str, float]:
    if ctx is None:
        ctx = TurnContext.from_battle(battle)
    s = ctx.stack
    f = dict.fromkeys(FEATURES, 0.0)
    after_cell = s.cell

    if action.type == ActionType.MOVE:
        after_cell = action.target_cell
    elif action.type == ActionType.WAIT:
        f["wait"] = 1.0
    elif action.type == ActionType.DEFEND:
        f["defend"] = 1.0
    else:
        target = ctx.enemies[action.target_cell]
        melee = is_melee(action.type)
        if melee:
            approach = battle.approach_cell(action.target_cell, action.type)
            if approach is not None:
                after_cell = approach
        dealt = expected_damage(s, target, melee)
        f["damage_dealt"] = min(dealt / target.total_hp, 1.0)
        f["kill"] = 1.0 if dealt >= target.total_hp else 0.0
        f["target_value"] = stack_value(target) / ctx.max_value
        f["focus_fire"] = 1.0 if target.total_hp == ctx.weakest_hp else 0.0
        if melee and dealt < target.total_hp and target.retaliations_left > 0:
            received = expected_damage(target, s, melee=True)
            f["damage_received"] = min(received / s.total_hp, 1.0)

    dist = min(chebyshev(after_cell, c) for c in ctx.enemies) / 10.0
    f["dist_ranged" if s.stats.is_ranged else "dist_melee"] = dist
    return f


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    if path is None:
        text = (resources.files("tactica.data") / "weights_default.json").read_text()
    else:
        text = Path(path).read_text()
    weights = dict.fromkeys(FEATURES, 0.0)
    for k, v in json.loads(text).items():
        if k not in weights:
            raise ValueError(f"unknown weight {k!r}; known features: {FEATURES}")
        weights[k] = float(v)
    return weights


class WeightedAgent(Agent):
    name = "weighted"

    def __init__(self, weights: dict[str, float] | str | Path | None = None) -> None:
        if weights is None or isinstance(weights, (str, Path)):
            self.weights_path = str(weights) if weights else "default"
            self.weights = load_weights(weights)
        else:
            self.weights_path = "inline"
            self.weights = {**dict.fromkeys(FEATURES, 0.0), **weights}

    def score(self, battle: Battle, action: Action,
              ctx: TurnContext | None = None) -> float:
        f = action_features(battle, action, ctx)
        return sum(self.weights[k] * f[k] for k in FEATURES)

    def act(self, battle: Battle) -> Action:
        legal = battle.legal_actions()
        ctx = TurnContext.from_battle(battle)
        return max(legal, key=lambda a: (self.score(battle, a, ctx), -a.id))

    def config(self) -> dict:
        return {"name": self.name, "weights": self.weights_path}
