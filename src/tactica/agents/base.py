"""Common agent interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from tactica.actions import Action
from tactica.battle import Battle


class Agent(ABC):
    """An agent maps a battle state to an action using only the public API."""

    name: str = "agent"

    @abstractmethod
    def act(self, battle: Battle) -> Action: ...

    def config(self) -> dict:
        """JSON-serializable description for game logs."""
        return {"name": self.name}
