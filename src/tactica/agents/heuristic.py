"""Scripted heuristic agent.

Ranged stacks shoot the lowest-effective-HP target and kite away when an
enemy is adjacent; melee stacks attack the highest-value reachable target or
advance toward it; everyone defends when nothing useful is available.
"""
from __future__ import annotations

from tactica.actions import Action, ActionType
from tactica.agents.base import Agent
from tactica.battle import Battle, Stack, chebyshev


def stack_value(s: Stack) -> float:
    """Material value of a stack: per-creature value scaled by remaining HP."""
    return s.stats.value * s.total_hp / s.stats.hp


def min_enemy_distance(battle: Battle, cell: int, side: int) -> int:
    enemies = [s for s in battle.stacks.values() if s.alive and s.side != side]
    return min(chebyshev(cell, e.cell) for e in enemies)


class HeuristicAgent(Agent):
    name = "heuristic"

    def act(self, battle: Battle) -> Action:
        s = battle.active_stack()
        legal = battle.legal_actions()
        by_type: dict[ActionType, list[Action]] = {}
        for a in legal:
            by_type.setdefault(a.type, []).append(a)

        enemies = {e.cell: e for e in battle.stacks.values()
                   if e.alive and e.side != s.side}

        shots = by_type.get(ActionType.RANGED_ATTACK, [])
        if shots:
            # Focus fire: shoot the target with the lowest remaining HP pool.
            return min(shots, key=lambda a: (enemies[a.target_cell].total_hp,
                                             a.target_cell))

        # One charge-aware default-approach melee Action per reachable target.
        reach = battle.reachable(s)
        melee_by_target: dict[int, Action] = {}
        for e in enemies.values():
            d = battle.default_melee(s, e, reach)
            if d is not None:
                melee_by_target[e.cell] = Action(d, e.cell)

        if s.stats.is_ranged:
            # Shooting is blocked by an adjacent enemy: kite if it gains
            # distance, otherwise melee the weakest adjacent enemy.
            here = min_enemy_distance(battle, s.cell, s.side)
            moves = by_type.get(ActionType.MOVE, [])
            if moves:
                best = max(moves, key=lambda a: (
                    min_enemy_distance(battle, a.target_cell, s.side),
                    -a.target_cell))
                if min_enemy_distance(battle, best.target_cell, s.side) > here:
                    return best
            if melee_by_target:
                return min(melee_by_target.values(),
                           key=lambda a: (enemies[a.target_cell].total_hp,
                                          a.target_cell))
            return Action(ActionType.DEFEND)

        if melee_by_target:
            # Hit the highest-value target.
            return max(melee_by_target.values(),
                       key=lambda a: (stack_value(enemies[a.target_cell]),
                                      -a.target_cell))

        # Advance toward the highest-value enemy.
        target = max(enemies.values(), key=lambda e: (stack_value(e), -e.cell))
        here = chebyshev(s.cell, target.cell)
        moves = by_type.get(ActionType.MOVE, [])
        if moves:
            best = min(moves, key=lambda a: (chebyshev(a.target_cell, target.cell),
                                             a.target_cell))
            if chebyshev(best.target_cell, target.cell) < here:
                return best
        return Action(ActionType.DEFEND)
