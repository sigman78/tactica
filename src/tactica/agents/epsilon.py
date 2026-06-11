"""Epsilon-random wrapper around any agent."""
from __future__ import annotations

import numpy as np

from tactica.actions import Action
from tactica.agents.base import Agent
from tactica.battle import Battle


class EpsilonAgent(Agent):
    def __init__(self, inner: Agent, epsilon: float, seed: int = 0) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.inner = inner
        self.epsilon = epsilon
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.name = f"epsilon({inner.name},{epsilon:g})"

    def act(self, battle: Battle) -> Action:
        if self.rng.random() < self.epsilon:
            actions = battle.legal_actions()
            return actions[int(self.rng.integers(len(actions)))]
        return self.inner.act(battle)

    def config(self) -> dict:
        return {"name": self.name, "epsilon": self.epsilon,
                "inner": self.inner.config()}
