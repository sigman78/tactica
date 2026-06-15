"""Flat UCT (UCB1 bandit over root actions) with random rollouts.

Each simulation clones the battle, reseeds the clone's RNG (so chance nodes
are sampled implicitly across simulations), steps the chosen root action,
then runs ``Battle.playout`` -- a fast random rollout policy -- for up to
``rollout_cap`` steps. If the rollout hits the cap without a result, a
material-balance evaluation stands in for the terminal return.

Directional melee is collapsed to one arm per target (the charge-aware
``default_melee`` direction) by ``_root_arms`` before the bandit sees it, so
the arm count stays at its pre-directional-melee shape.
"""
from __future__ import annotations

import math

import numpy as np

from tactica.actions import Action, ActionType, is_melee
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
                 seed: int = 0, rollout_cap: int = 40) -> None:
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
            if is_melee(a.type) or a.type == ActionType.RANGED_ATTACK:
                attacks.append(i)
            elif a.type in (ActionType.WAIT, ActionType.DEFEND):
                control.append(i)
            else:
                moves.append(i)
        shuffled = lambda xs: [xs[int(i)] for i in self.rng.permutation(len(xs))]
        return shuffled(attacks) + control + shuffled(moves)

    def _root_arms(self, battle: Battle) -> list[Action]:
        """Collapse the 8 melee directions per target into a single
        charge-aware default arm; pass non-melee actions through. Keeps the
        bandit's arm count at the pre-directional-melee shape."""
        s = battle.active_stack()
        reach = battle.reachable(s)
        arms: list[Action] = []
        seen_targets: set[int] = set()
        for a in battle.legal_actions(reach=reach):
            if is_melee(a.type):
                if a.target_cell in seen_targets:
                    continue
                seen_targets.add(a.target_cell)
                target = battle._stack_at(a.target_cell)
                assert target is not None  # a legal melee implies a stack here
                # legal_actions emitted a reachable side for this target, so
                # default_melee (same reach) must find one too.
                d = battle.default_melee(s, target, reach)
                assert d is not None
                arms.append(Action(d, a.target_cell))
            else:
                arms.append(a)
        return arms

    def act(self, battle: Battle) -> Action:
        root_actions = self._root_arms(battle)
        if len(root_actions) == 1:
            return root_actions[0]
        player = battle.current_player()
        n_arms = len(root_actions)
        n = np.zeros(n_arms)
        w = np.zeros(n_arms)
        sweep = self._sweep_order(root_actions)
        # Common random numbers across arms: every visit in round r uses the
        # same rollout seed for every arm, so seed luck is common-mode and
        # cancels out of the cross-arm ranking. (Unpaired allocation -- e.g.
        # plain UCB revisits -- lets the max over many one-sample means be
        # won by a lucky outlier arm, which made *more* simulations play
        # strictly worse.)
        visit_seeds: dict[int, int] = {}

        def seed_for(visit: int) -> int:
            if visit not in visit_seeds:
                visit_seeds[visit] = int(self.rng.integers(2**63))
            return visit_seeds[visit]

        def simulate(idx: int) -> None:
            b = battle.clone()
            b.reseed(seed_for(int(n[idx])))
            b.step(root_actions[idx])
            n[idx] += 1
            w[idx] += self._rollout(b, player)

        # Progressive widening: a one-sample mean per arm is a coin flip, so
        # ranking dozens of arms on single rollouts is a max-of-noise lottery
        # (empirically, 8-sim agents beat 128-sim agents before this).
        # Consider ~sqrt(2*sims) arms in sweep-priority order, so a bigger
        # budget deepens estimates faster than it widens the candidate set.
        considered = max(1, min(int(math.sqrt(2 * self.simulations)), n_arms))
        arms = sweep[:considered]
        rounds = self.simulations // considered
        paired_budget = rounds * considered
        for sim in range(paired_budget):
            simulate(arms[sim % considered])
        # Leftover budget goes to UCB1 over the paired means: it sharpens
        # the leaders' estimates while the paired rounds carry the ranking.
        for sim in range(paired_budget, self.simulations):
            ucb = np.full(n_arms, -np.inf)
            ucb[arms] = w[arms] / n[arms] + self.c_uct * np.sqrt(
                math.log(sim) / n[arms])
            simulate(int(np.argmax(ucb)))

        means = np.where(n > 0, w / np.maximum(n, 1), -np.inf)
        best = max(range(n_arms),
                   key=lambda i: (means[i], n[i], -root_actions[i].id))
        return root_actions[best]
