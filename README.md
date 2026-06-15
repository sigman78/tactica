# tactica

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A HoMM-style turn-based tactics sandbox for exploring, testing, and evolving
game-playing AI. The package is three things:

1. a **headless, deterministic rules engine** (11x9 grid, creature stacks,
   speed-based turn order, retaliation, ranged/melee/flyers, unit perks),
2. an **agent ladder** from uniform-random to flat-UCT MCTS,
3. **evaluation tooling** built for statistical rigor: mirrored pairs, common
   random numbers, noise floors, skill curves, SPRT, and byte-identical
   replays.

Simulator speed, determinism, and evaluation rigor are the priorities — game
content is intentionally small (5 unit types, 6 scenarios, no spells/morale).
Heuristic-vs-heuristic runs at ~4500 battles/min single-core.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```
uv sync          # installs numpy + dev/eval groups (pytest, openskill, matplotlib)
uv run pytest    # full test suite, ~4 s
```

pip users: `pip install -e .[eval]`.

## Quick start

```
uv run tactica play --p0 heuristic --p1 random --scenario open_field --seed 1 --render
uv run tactica tournament --agents random,heuristic,mcts --scenarios all --pairs 20
uv run tactica web    # interactive dashboard on http://127.0.0.1:8321
```

One entry point, seven subcommands. Agent specs used everywhere:
`random`, `heuristic`, `weighted[:weights.json]`,
`mcts[:SIMS[:C_UCT[:ROLLOUT_CAP]]]`, and `epsilon:EPS:INNER`.

| Command | What it does |
| --- | --- |
| `tactica play` | run one battle, optionally rendered or logged as a replay |
| `tactica tournament` | round-robin over mirrored pairs: pair-score matrix, per-scenario breakdown, OpenSkill ratings |
| `tactica noise-floor` | agent vs itself — the luck baseline / resolution limit of your experiment |
| `tactica skill-curve` | inject blunders at rate eps and measure the cost: do decisions matter? |
| `tactica sprt` | sequential testing for weight changes, stops as soon as the evidence decides |
| `tactica replay` | re-simulate a logged game and assert the byte-identical state hash |
| `tactica web` | local single-page war room: live runs over SSE, replay viewer, play vs an agent |

Full per-command walkthroughs with example output: [docs/cli.md](docs/cli.md).

## Documentation

- [docs/cli.md](docs/cli.md) — CLI reference: all seven subcommands with
  example runs, plus the web dashboard tour.
- [docs/agents.md](docs/agents.md) — the agent ladder, from `random` to
  `mcts`, including the measured design decisions behind the MCTS agent.
- [docs/simulator.md](docs/simulator.md) — the `Battle` API contract
  (action encoding, observation planes, RNG discipline, state hash) and a
  rules summary.
- [docs/evaluation.md](docs/evaluation.md) — the evaluation methodology:
  mirrored pairs, CRN, noise floors, skill curves, SPRT.
- [docs/extending.md](docs/extending.md) — adding agents, scenarios, unit
  types, and weight profiles.

## Non-goals

No spells, morale/luck, hex grids, GUIs, RL training loops, or external game
engines. The action encoding and `observe()` are designed so a
PettingZoo/Gymnasium wrapper can be added later without touching the core.

## License

[MIT](LICENSE).
