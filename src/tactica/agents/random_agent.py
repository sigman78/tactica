"""Uniform random agent."""
from __future__ import annotations

import numpy as np

from tactica.actions import Action
from tactica.agents.base import Agent
from tactica.battle import Battle


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def act(self, battle: Battle) -> Action:
        actions = battle.legal_actions()
        return actions[int(self.rng.integers(len(actions)))]
