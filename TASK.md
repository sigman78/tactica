# Task: Build "tactica" — a HoMM-style tactics AI sandbox

Build a complete, working Python project in one pass: a headless turn-based tactics
simulator plus an agent ladder, tournament/evaluation tooling, and statistical
analysis commands. This is a research sandbox for exploring, testing, and evolving
game-playing AI — simulator speed, determinism, and evaluation rigor matter more
than game-content richness.

## Tech constraints

- Python 3.11+, managed with **uv** (`pyproject.toml`, `uv.lock`, src layout: `src/tactica/`).
- Core dependencies only: `numpy`. Evaluation extras: `openskill`, `matplotlib`. Dev: `pytest`.
  CLI via stdlib `argparse` (single entry point `tactica` with subcommands). No heavy frameworks.
- Everything must run headless from the CLI. Rendering = ASCII only.
- Type hints everywhere; dataclasses for state objects.

## Architecture (non-negotiable)

The simulator core must be a pure, framework-free rules engine:

```python
class Battle:
    @classmethod
    def from_scenario(cls, scenario: Scenario, seed: int) -> "Battle"
    def clone(self) -> "Battle"                      # cheap deep copy, used by MCTS
    def current_player(self) -> int                  # 0 or 1
    def legal_actions(self) -> list[Action]
    def legal_action_mask(self) -> np.ndarray        # bool, fixed-size action space
    def step(self, action: Action) -> None           # mutates; raises on illegal action
    def is_terminal(self) -> bool
    def returns(self) -> tuple[float, float]         # (+1,-1) / (-1,+1) / (0,0)
    def observe(self) -> np.ndarray                  # stacked feature planes, see below
    def render(self) -> str                          # ASCII board
```

- All randomness flows through ONE `np.random.Generator` owned by the battle, created
  from the seed. No `random` module, no global RNG, anywhere.
- **Deterministic mode**: `Scenario(deterministic=True)` replaces all damage rolls with
  expected values and disables morale/luck. Same API, zero chance nodes.
- Fixed-size flat action space: `action_id = action_type * (W*H) + target_cell`, with
  action types {MOVE, MELEE_ATTACK, RANGED_ATTACK, WAIT, DEFEND}. WAIT/DEFEND ignore the
  cell index. `legal_action_mask()` must align with this encoding exactly.
- `observe()` returns float32 planes (C, H, W): per-side unit-type one-hots, normalized
  stack count, normalized HP of top creature, "active unit" plane, obstacles, plus a
  small global feature vector appended as constant planes (round number, side to move).

## Game rules (keep this scope, do not expand)

- Rectangular grid 11 wide x 9 tall (use square grid with 8-neighborhood movement —
  skip hex math), a few impassable obstacle cells defined per scenario.
- HoMM-style **stacks**: a unit on the board = (unit_type, count, top_hp). Damage kills
  whole creatures from the stack; remainder reduces top creature HP.
- 5 unit types with a stats table (speed, attack, defense, dmg_min, dmg_max, hp,
  is_ranged, initiative): Pikeman, Archer, Griffin (fast flyer — ignores obstacles when
  moving), Swordsman, Cavalry. Tune numbers for rough usefulness parity, not balance.
- HoMM damage formula: `base = dmg_roll * count`, modified ±5% per point of
  attack−defense difference, clamped to [0.3x, 3x].
- Turn order: each round, all living stacks act once, sorted by initiative (ties broken
  by a seeded shuffle at battle start, fixed for the whole battle). Acting stack may
  move up to `speed` cells (BFS reachability respecting obstacles/units), then attack
  an adjacent enemy; or shoot any enemy if ranged and no adjacent enemy (else melee at
  half damage); or WAIT (act later this round, once); or DEFEND (+defense until next turn).
- Melee triggers **one retaliation** per defender per round (ranged attacks don't).
- Battle ends when one side has no stacks, or after 100 rounds (draw).
- `Scenario`: dataclass with name, army lists (unit_type, count, start_cell), obstacles,
  deterministic flag. Ship 6 built-in scenarios: 3 mirror-symmetric, 3 asymmetric
  (e.g., shooters+pikemen vs cavalry rush), loadable by name and from JSON files.

## Agents

Common interface: `Agent.act(battle: Battle) -> Action` (must only use the public API;
search agents use `clone()`).

1. `RandomAgent` — uniform over legal actions.
2. `HeuristicAgent` — scripted: ranged units focus lowest-effective-HP reachable target
   and kite when threatened; melee advance toward / attack highest-value target;
   defend when nothing useful. Must clearly beat random (>90% WR on symmetric maps).
3. `WeightedAgent` — same feature set but scores candidate actions via a weight vector
   (distance, target value, expected damage dealt/received, retaliation cost, focus
   fire). Weights from a JSON file; ship defaults that imitate HeuristicAgent.
4. `EpsilonAgent(inner, epsilon)` — wrapper: with prob ε play a random legal action.
5. `MCTSAgent(simulations, c_uct, seed)` — flat UCT over `clone()`+`step()`, chance
   handled implicitly by sampled rollouts; random rollout policy capped at 200 steps,
   fall back to a material-balance evaluation at the cap.

## Evaluation tooling (this is the heart of the project)

CLI subcommands:

- `tactica play --p0 heuristic --p1 mcts --scenario open_field --seed 42 --render`
  — run one battle, optional ASCII rendering per turn, print result.
- `tactica tournament --agents random,heuristic,weighted,mcts --scenarios all
  --pairs 200 --seed 1 --out results.jsonl`
  — round-robin where the unit of play is a **mirrored pair** (same scenario+seed, sides
  swapped). Use **common random numbers**: pair i of matchup (A,B) on scenario s uses
  seed derived as `hash((base_seed, s, i))` for ALL agent pairs, so comparisons are
  paired across the tournament. Output one JSONL row per game. Print a win-rate matrix
  (pair-level scores), per-scenario breakdown, and OpenSkill ratings with intervals.
- `tactica noise-floor --agent heuristic --scenarios all --pairs 500`
  — agent vs itself, mirrored; report deviation from 50% with 95% CI per scenario.
  This is the luck baseline.
- `tactica skill-curve --agent heuristic --epsilons 0,0.05,0.1,0.2,0.5 --pairs 300`
  — EpsilonAgent(heuristic, ε) vs clean heuristic; print/save WR-vs-ε table and a
  matplotlib plot. This measures how much decisions matter.
- `tactica sprt --candidate weights_new.json --baseline weights_default.json
  --elo0 0 --elo1 10 --alpha 0.05 --beta 0.05`
  — sequential probability ratio test on mirrored pairs (trinomial WDL is fine),
  streaming games until accept/reject; print verdict, n games, LLR trajectory.
- `tactica replay --file replays/xyz.jsonl` — re-render a logged game from
  (scenario, seed, action list); assert the final state hash matches the log.

Every game logs: scenario, seed, agent names/configs, action list, winner, rounds,
final state hash. Replays must reproduce byte-identically.

## Tests (pytest, must pass)

- Rules unit tests: damage formula edge cases, retaliation once per round, WAIT
  ordering, ranged-blocked-by-adjacency, flyer movement, stack kill arithmetic.
- Property tests (randomized, seeded): N random playthroughs where every sampled action
  is legal per the mask, mask agrees with `legal_actions()`, state stays valid, games
  terminate within the round cap.
- Determinism test: same scenario+seed+action-list twice → identical state hash;
  `clone()` then divergent play does not affect the original.
- Replay round-trip test, and a smoke test that `heuristic` beats `random` ≥90% over
  50 pairs on a symmetric scenario.

## Definition of done (verify each before finishing)

1. `uv sync && uv run pytest` — all green.
2. `uv run tactica play --p0 heuristic --p1 random --scenario open_field --seed 1 --render` works.
3. `uv run tactica tournament --agents random,heuristic,mcts --scenarios all --pairs 20 --seed 1`
   completes in a few minutes, prints sane matrix + ratings (random clearly last).
4. `uv run tactica noise-floor --agent heuristic --pairs 50` runs and reports CIs.
5. README.md: project goal, install, every CLI command with example output, the
   simulator API contract, how to add a new agent / scenario / unit type, and a short
   "evaluation methodology" section explaining mirrored pairs, CRN, noise floor, SPRT.

## Non-goals

No spells, morale/luck (beyond the deterministic-mode hook), hex grids, GUIs, RL
training loops, or external game engines. Design the action encoding and `observe()`
so an RL wrapper (PettingZoo/Gymnasium) can be added later without touching the core.
Prioritize: correct rules > reproducibility > simulator speed (target ≥200 full
battles/min for heuristic vs heuristic) > everything else.