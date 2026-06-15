"""Agent ladder and a spec-string factory used by the CLI.

Agent spec grammar (used in ``--p0/--p1/--agents``):

- ``random``
- ``heuristic``
- ``weighted`` or ``weighted:path/to/weights.json``
- ``mcts`` or ``mcts:SIMS[:C_UCT[:ROLLOUT_CAP]]``; append ``heuristic`` for a
  HeuristicAgent-guided rollout (default ``random``), e.g. ``mcts:64:heuristic``
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
        # A "heuristic"/"random" token sets the rollout policy from any
        # position; the rest are positional SIMS[:C_UCT[:ROLLOUT_CAP]].
        policy = "random"
        nums = []
        for p in rest.split(":"):
            if not p:
                continue
            if p in ("heuristic", "random"):
                policy = p
            else:
                nums.append(p)
        sims = int(nums[0]) if nums else 32
        c_uct = float(nums[1]) if len(nums) > 1 else 1.4
        kwargs = {"rollout_cap": int(nums[2])} if len(nums) > 2 else {}
        return MCTSAgent(simulations=sims, c_uct=c_uct, seed=seed,
                         rollout_policy=policy, **kwargs)
    if head == "epsilon":
        eps_str, _, inner = rest.partition(":")
        if not inner:
            raise ValueError(f"epsilon spec needs an inner agent: {spec!r}")
        return EpsilonAgent(make_agent(inner, seed=seed + 1), float(eps_str),
                            seed=seed)
    raise ValueError(f"unknown agent spec {spec!r}")
