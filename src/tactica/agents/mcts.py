"""Flat UCT (UCB1 bandit over root actions) with random rollouts.

Each simulation clones the battle, reseeds the clone's RNG (so chance nodes
are sampled implicitly across simulations), steps the chosen root action,
then runs ``Battle.playout`` -- a fast random rollout policy -- for up to
``rollout_cap`` steps. If the rollout hits the cap without a result, a
material-balance evaluation stands in for the terminal return.
"""
from __future__ import annotations

import math

import numpy as np

from tactica.actions import Action, ActionType
from tactica.agents.base import Agent
from tactica.battle import Battle
from tactica.agents.heuristic import stack_value

ROLLOUT_CAP = 200


def material_eval(battle: Battle) -> tuple[float, float]:
    """Material balance in [-1, 1] per side; stands in for returns()."""
    m = [0.0, 0.0]
    for s in battle.stacks.values():
        if s.alive:
            m[s.side] += stack_value(s)
    total = m[0] + m[1]
    if total <= 0:
        return (0.0, 0.0)
    v0 = (m[0] - m[1]) / total
    return (v0, -v0)


class MCTSAgent(Agent):
    def __init__(self, simulations: int = 32, c_uct: float = 1.4,
                 seed: int = 0, rollout_cap: int = ROLLOUT_CAP) -> None:
        self.simulations = simulations
        self.c_uct = c_uct
        self.rollout_cap = rollout_cap
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.name = f"mcts({simulations})"

    def config(self) -> dict:
        return {"name": self.name, "simulations": self.simulations,
                "c_uct": self.c_uct, "rollout_cap": self.rollout_cap}

    def _rollout(self, b: Battle, player: int) -> float:
        steps = b.playout(max_steps=self.rollout_cap)
        if b.is_terminal():
            value = b.returns()[player]
        else:
            value = material_eval(b)[player]
        # Length discount: a win now beats a win in 150 plies (and a slow
        # loss beats a fast one). Without it, every arm of a winning
        # position rolls out to +1 and the agent dawdles instead of killing.
        return value * (1.0 - 0.3 * steps / self.rollout_cap)

    def _sweep_order(self, root_actions: list[Action]) -> list[int]:
        """Initial-visit order: attack arms first (shuffled), then WAIT and
        DEFEND, then moves (shuffled). With fewer simulations than arms a
        fixed or uniform order would routinely leave every attack untried."""
        attacks, control, moves = [], [], []
        for i, a in enumerate(root_actions):
            if a.type in (ActionType.MELEE_ATTACK, ActionType.RANGED_ATTACK):
                attacks.append(i)
            elif a.type in (ActionType.WAIT, ActionType.DEFEND):
                control.append(i)
            else:
                moves.append(i)
        shuffled = lambda xs: [xs[int(i)] for i in self.rng.permutation(len(xs))]
        return shuffled(attacks) + control + shuffled(moves)

    def act(self, battle: Battle) -> Action:
        root_actions = battle.legal_actions()
        if len(root_actions) == 1:
            return root_actions[0]
        player = battle.current_player()
        n = np.zeros(len(root_actions))
        w = np.zeros(len(root_actions))
        sweep = self._sweep_order(root_actions)
        # Common random numbers across arms: the k-th visit of every arm
        # shares a rollout seed, so single-visit comparisons are paired and
        # the root action's own effect dominates rollout noise.
        visit_seeds: dict[int, int] = {}

        def seed_for(visit: int) -> int:
            if visit not in visit_seeds:
                visit_seeds[visit] = int(self.rng.integers(2**63))
            return visit_seeds[visit]

        for sim in range(self.simulations):
            if sim < len(root_actions):
                idx = sweep[sim]
            else:
                ucb = w / n + self.c_uct * np.sqrt(math.log(sim) / n)
                idx = int(np.argmax(ucb))
            b = battle.clone()
            b.reseed(seed_for(int(n[idx])))
            b.step(root_actions[idx])
            value = self._rollout(b, player)
            n[idx] += 1
            w[idx] += value

        means = np.where(n > 0, w / np.maximum(n, 1), -np.inf)
        return root_actions[int(np.argmax(means))]
