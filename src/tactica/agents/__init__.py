"""Agent ladder and a spec-string factory used by the CLI.

Agent spec grammar (used in ``--p0/--p1/--agents``):

- ``random``
- ``heuristic``
- ``weighted`` or ``weighted:path/to/weights.json``
- ``mcts`` or ``mcts:SIMS`` or ``mcts:SIMS:C_UCT``
- ``epsilon:EPS:INNER_SPEC`` e.g. ``epsilon:0.1:heuristic``
"""
from __future__ import annotations

from tactica.agents.base import Agent
from tactica.agents.epsilon import EpsilonAgent
from tactica.agents.heuristic import HeuristicAgent
from tactica.agents.mcts import MCTSAgent
from tactica.agents.random_agent import RandomAgent
from tactica.agents.weighted import WeightedAgent

__all__ = ["Agent", "EpsilonAgent", "HeuristicAgent", "MCTSAgent",
           "RandomAgent", "WeightedAgent", "make_agent"]


def make_agent(spec: str, seed: int = 0) -> Agent:
    """Build an agent from a CLI spec string. Stochastic agents are seeded
    from ``seed`` so a given (spec, seed) pair is fully reproducible."""
    head, _, rest = spec.partition(":")
    head = head.strip().lower()
    if head == "random":
        return RandomAgent(seed=seed)
    if head == "heuristic":
        return HeuristicAgent()
    if head == "weighted":
        return WeightedAgent(rest or None)
    if head == "mcts":
        parts = [p for p in rest.split(":") if p]
        sims = int(parts[0]) if parts else 32
        c_uct = float(parts[1]) if len(parts) > 1 else 1.4
        kwargs = {"rollout_cap": int(parts[2])} if len(parts) > 2 else {}
        return MCTSAgent(simulations=sims, c_uct=c_uct, seed=seed, **kwargs)
    if head == "epsilon":
        eps_str, _, inner = rest.partition(":")
        if not inner:
            raise ValueError(f"epsilon spec needs an inner agent: {spec!r}")
        return EpsilonAgent(make_agent(inner, seed=seed + 1), float(eps_str),
                            seed=seed)
    raise ValueError(f"unknown agent spec {spec!r}")
