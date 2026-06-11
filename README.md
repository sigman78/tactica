# tactica

A HoMM-style turn-based tactics sandbox for exploring, testing, and evolving
game-playing AI. The package is three things:

1. a **headless, deterministic rules engine** (11x9 grid, creature stacks,
   initiative order, retaliation, ranged/melee/flyers),
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

## CLI

One entry point, six subcommands. Agent specs used everywhere:
`random`, `heuristic`, `weighted[:weights.json]`,
`mcts[:SIMS[:C_UCT[:ROLLOUT_CAP]]]`, and `epsilon:EPS:INNER`
(e.g. `epsilon:0.1:heuristic`).

### `tactica play` — run one battle

```
$ uv run tactica play --p0 heuristic --p1 random --scenario open_field --seed 1 --render
Round 1
    0  1  2  3  4  5  6  7  8  9 10
 0  .  .  .  .  .  .  .  .  .  .  .
 1  A  .  .  .  .  .  .  .  .  .  a
 2  .  .  .  .  .  .  .  .  .  .  .
 3  P  .  .  .  .  .  .  .  .  .  p
 4  S  .  .  .  .  .  .  .  .  .  s
 5  P  .  .  .  .  .  .  .  .  .  p
 6  .  .  .  .  .  .  .  .  .  .  .
 7  G  .  .  .  .  .  .  .  .  .  g
 8  .  .  .  .  .  .  .  .  .  .  .
Side 0:  Archer x12 (10/10hp)@(0, 1)  ...  *Griffin x6 (25/25hp)@(0, 7)
Side 1:  Archer x12 (10/10hp)@(10, 1)  ...
[... one board per action ...]
open_field seed=1: side 0 (heuristic) wins after 7 rounds, hash=6770ab9a165d29c1
```

`--deterministic` switches damage to expected values (zero chance nodes);
`--out replays/x.jsonl` writes a replayable game record.

### `tactica tournament` — round-robin over mirrored pairs

```
$ uv run tactica tournament --agents random,heuristic,mcts --scenarios all --pairs 20 --seed 1 --out results.jsonl
720 games (360 mirrored pairs) in 48.7s

Pair-score matrix (row vs column, 0.5 = even, +/- is 95% CI):
                      random     heuristic          mcts
        random             -  0.037+/-0.02  0.067+/-0.03
     heuristic  0.963+/-0.02             -  0.588+/-0.05
          mcts  0.933+/-0.03  0.412+/-0.05             -

Per-scenario pair scores (matchup: scenario=score):
  heuristic vs mcts: archers_vs_cavalry=0.500+/-0.00  chokepoint=0.525+/-0.13  ...
  random vs heuristic: archers_vs_cavalry=0.000+/-0.00  chokepoint=0.000+/-0.00  ...
  random vs mcts: archers_vs_cavalry=0.113+/-0.08  chokepoint=0.000+/-0.00  ...

OpenSkill ratings (PlackettLuce; ordinal = mu - 3*sigma):
                 heuristic  mu= 34.109  sigma= 1.646  ordinal= 29.170  95% mu-range=[30.88, 37.34]
                      mcts  mu= 33.963  sigma= 1.626  ordinal= 29.085  95% mu-range=[30.78, 37.15]
                    random  mu= 12.701  sigma= 2.575  ordinal=  4.977  95% mu-range=[7.65, 17.75]

game log written to results.jsonl
```

The unit of play is a **mirrored pair** (same scenario + seed, sides
swapped); the matrix reports pair-level scores with 95% CIs, plus a
per-scenario breakdown and OpenSkill ratings. One JSONL row per game.
`--workers` defaults to one process per CPU core.

### `tactica noise-floor` — the luck baseline

```
$ uv run tactica noise-floor --agent heuristic --pairs 50
Noise floor: heuristic vs itself, 50 mirrored pairs per scenario (CRN seeds, base=1)

            open_field: game-level side-0 score=0.6000 +/-0.0965 (dev +0.1000)   pair-level=0.5000 +/-0.0000
            chokepoint: game-level side-0 score=0.4400 +/-0.0978 (dev -0.0600)   pair-level=0.5000 +/-0.0000
              skirmish: game-level side-0 score=0.7200 +/-0.0884 (dev +0.2200)   pair-level=0.5000 +/-0.0000
    archers_vs_cavalry: game-level side-0 score=0.0000 +/-0.0000 (dev -0.5000)   pair-level=0.5000 +/-0.0000
          griffin_raid: game-level side-0 score=0.0000 +/-0.0000 (dev -0.5000)   pair-level=0.5000 +/-0.0000
            last_stand: game-level side-0 score=0.2200 +/-0.0816 (dev -0.2800)   pair-level=0.5000 +/-0.0000

               OVERALL: game-level side-0 score=0.3300 +/-0.0377 (dev -0.1700)   pair-level=0.5000 +/-0.0000
```

Runs an agent against itself on mirrored pairs, and the output above is the
whole argument for the paired design: raw game-level scores are badly biased
(the asymmetric maps are one-sided, and even symmetric `skirmish` shows a
+0.22 initiative-driven side advantage), while pair-level scores sit at
exactly 0.500 — the bias cancels inside each pair. For a deterministic agent
the pair-level noise is exactly zero; for stochastic agents
(`noise-floor --agent mcts:8`) it is the true resolution limit of your
experiment.

### `tactica skill-curve` — how much do decisions matter?

```
$ uv run tactica skill-curve --agent heuristic --epsilons 0,0.05,0.1,0.2,0.5 --pairs 300
Skill curve: epsilon(heuristic, eps) vs heuristic, 300 pairs x 6 scenarios per point

  eps=0     score=0.5000 +/-0.0000
  eps=0.05  score=0.4689 +/-0.0081
  eps=0.1   score=0.4306 +/-0.0096
  eps=0.2   score=0.3600 +/-0.0110
  eps=0.5   score=0.2072 +/-0.0114

(0.5 = noise mistakes cost nothing; lower = decisions matter)
plot saved to skill_curve.png
```

Plays `EpsilonAgent(agent, eps)` against the clean agent and plots the score
as a function of the blunder rate. A steep curve means the environment
rewards skill; a flat one means outcomes are luck-dominated.

### `tactica sprt` — sequential testing for weight changes

```
$ uv run tactica sprt --candidate weights/conservative.json --baseline weights/default.json --elo0 0 --elo1 10 --alpha 0.05 --beta 0.05
SPRT: H1 elo>=10.0 vs H0 elo<=0.0, alpha=0.05 beta=0.05
LLR bounds: accept H0 at -2.944, accept H1 at 2.944

  n=   50  WDL=23/0/27  LLR=-0.155
  [... streams mirrored pairs until a bound is crossed or --max-pairs ...]
  n= 1950  WDL=938/0/1012  LLR=-2.941

Verdict: accept H0 (no improvement)
Games: 1960 (980 mirrored pairs)  WDL=943/0/1017  score=0.4811 (~-13.1 elo)
Final LLR: -2.945  (bounds [-2.944, 2.944])
LLR trajectory (last 12 of 980 pairs): -2.94, -2.94, ..., -2.94, -2.94

(That run is how the shipped default weights earned their place: the
aggressive profile was SPRT-accepted at ~+11.5 elo over the original
heuristic-imitating weights, which live on as weights/conservative.json.)
```

Streams mirrored pairs of `WeightedAgent(candidate)` vs
`WeightedAgent(baseline)` and computes the trinomial GSPRT log-likelihood
ratio after each pair, stopping the moment either hypothesis is accepted.

### `tactica replay` — byte-identical re-simulation

```
$ uv run tactica replay --file replays/test.jsonl
griffin_raid seed=9: side 1 (mcts:16) wins after 3 rounds, hash=7b112c15bdf6bff4
replay OK: final state hash matches (7b112c15bdf6bff4)
```

Re-simulates a logged game from (scenario, seed, action list) and asserts
the final state hash matches the log. `--render` replays the ASCII board
turn by turn; `--index N` picks a row from a multi-game log such as
`results.jsonl`.

## Agent ladder

- **random** — uniform over legal actions; the floor every agent must beat.
- **heuristic** — scripted: archers focus the lowest-HP target and kite when
  cornered; melee attacks the highest-value reachable target or advances on
  it; defends when nothing is useful. The strongest shipped agent.
- **weighted** — scores every legal action with a linear feature vector
  (expected damage dealt/received, kill bonus, target value, focus fire,
  distance terms, wait/defend flags) and plays the argmax. The shipped
  default weights are an aggressive profile that was SPRT-accepted at
  ~+11.5 elo over the original heuristic-imitating weights
  (`weights/conservative.json`).
- **epsilon:EPS:INNER** — plays a uniform random legal action with
  probability EPS, otherwise defers to INNER. The measurement probe behind
  `skill-curve`.
- **mcts** — flat UCT with several documented twists that earned their place
  through measurements you can rerun:
  - *CRN-paired rounds*: simulations are allocated in full rounds where
    every arm shares the round's rollout seed, so seed luck is common-mode
    and cancels from the ranking. (Unpaired UCB revisits let the max over
    ~50 one-sample means be won by lucky outlier arms — measurably, 8-sim
    agents beat 128-sim agents until this change.)
  - *Progressive widening*: only ~sqrt(2·sims) arms are considered, attack
    arms first — rollout evaluations rank attacks usefully but are noisy
    and systematically retreat-happy on movement arms.
  - *Rollout-length discount*: values decay ~30% over the rollout cap, so a
    kill now beats the same win 100 plies later. Without it the agent rolls
    out a won position from every arm, ties at +1.0, and dawdles forever.
  - *Short biased rollouts*: `Battle.playout` (attack- and chase-biased
    random policy) capped at 40 steps by default, then the material-balance
    eval. The spec's literal 200-step cap is available as `mcts:S:C:200`,
    but it measures weaker (0.42 +/- 0.12 head-to-head vs the 40-step
    default, and 0.22 vs heuristic instead of 0.35) and runs ~5x slower:
    long random rollouts drown the root action's effect in outcome noise.
  - A known, measured limitation: strength does **not** scale cleanly with
    simulations (128 sims ~ 0.25 vs 8 sims on open_field). Uninformed
    rollouts undervalue initiating melee trades (the attacker always eats
    retaliation; the tempo payoff is beyond the evaluation horizon), so
    extra search converges on that bias instead of fixing it. Better
    rollout policies or tree search are the obvious next experiments —
    this sandbox exists to make those measurable.

## Simulator API contract

The core is a pure, framework-free rules engine — no CLI, logging, or agent
imports. Everything an agent or RL wrapper needs:

```python
class Battle:
    @classmethod
    def from_scenario(cls, scenario: Scenario, seed: int) -> "Battle"
    def clone(self) -> "Battle"                      # cheap deep copy, used by MCTS
    def reseed(self, seed: int) -> None              # fresh chance stream for search clones
    def current_player(self) -> int                  # 0 or 1
    def legal_actions(self) -> list[Action]
    def legal_action_mask(self) -> np.ndarray        # bool, fixed size 5*11*9 = 495
    def step(self, action: Action) -> None           # mutates; raises ValueError on illegal
    def is_terminal(self) -> bool
    def returns(self) -> tuple[float, float]         # (+1,-1) / (-1,+1) / (0,0)
    def observe(self) -> np.ndarray                  # float32 planes (18, 9, 11)
    def render(self) -> str                          # ASCII board
    def playout(self, max_steps=200, ...) -> int     # fast biased-random rollout (search helper)
```

- **Action encoding**: `action_id = action_type * 99 + target_cell` with
  types `{MOVE, MELEE_ATTACK, RANGED_ATTACK, WAIT, DEFEND}`. WAIT/DEFEND are
  canonical at cell 0; non-canonical ids are rejected. The mask aligns with
  this encoding exactly, so an RL policy head of size 495 plugs straight in.
- **Observation planes** (C=18, H=9, W=11): per-side unit-type one-hots
  (10), normalized stack count (2), normalized top-creature HP (2), active
  unit (1), obstacles (1), constant planes for round number and side to move
  (2).
- **Randomness**: a single `np.random.Generator` owned by the battle, seeded
  at construction. No `random` module, no global RNG, anywhere.
  `Scenario(deterministic=True)` replaces damage rolls with expected values —
  the same API with zero chance nodes (the initiative tie-shuffle happens at
  construction and is fixed by the seed).
- **State hash**: `Battle.state_hash()` digests stacks, queue, round, and
  RNG state; replays must reproduce it byte-identically.

### Rules summary

11x9 squares, 8-neighborhood movement (BFS up to `speed`; flyers ignore
obstacles/units in transit but land on free cells). Stacks are
`(unit_type, count, top_hp)`; damage kills whole creatures, the remainder
dents the top one. HoMM damage: `dmg_roll * count`, modified 5% per point of
attack-defense difference, clamped to [0.3x, 3x]. Each round every living
stack acts once in initiative order (ties broken by a shuffle seeded at
battle start); WAIT defers a stack once per round to a reverse-initiative
wait phase; DEFEND grants +2 defense until the stack's next turn. Melee
draws one retaliation per defender per round; ranged attacks don't, but a
shooter with an adjacent enemy cannot shoot and melees at half damage. A
side with no stacks loses; 100 rounds is a draw.

## Extending

**New agent** — subclass `tactica.agents.base.Agent`, implement
`act(battle) -> Action` using only the public API (search agents must use
`clone()` + `reseed()`), and register a spec name in
`tactica.agents.make_agent`. Every CLI command then accepts it.

**New scenario** — either add a `Scenario` to
`tactica.scenario.BUILTIN_SCENARIOS`, or write a JSON file and pass its path
anywhere a scenario name is accepted:

```json
{"name": "my_map",
 "army0": [["ARCHER", 12, 12], ["PIKEMAN", 20, 34]],
 "army1": [["CAVALRY", 6, 21]],
 "obstacles": [48, 49, 50],
 "deterministic": false}
```

(army rows are `[unit, count, cell]` with `cell = y * 11 + x`.)

**New unit type** — add a member to `UnitType` and a `UnitStats` row to
`STATS` in `src/tactica/units.py`, plus a glyph in `GLYPHS`. Scenarios can
use the new unit by name immediately.

**New weights** — copy `weights/default.json`, tune, then validate with
`tactica sprt --candidate yours.json --baseline weights/default.json`.

## Evaluation methodology

Tactics battles are noisy: damage rolls, initiative tiebreaks, and asymmetric
maps can fake or bury an agent improvement. The tooling attacks variance
from several directions:

- **Mirrored pairs.** The unit of play is two games on the same scenario and
  seed with sides swapped. Map asymmetry and side advantage cancel within
  the pair instead of inflating variance across the sample.
- **Common random numbers (CRN).** Pair `i` on scenario `s` uses seed
  `derive_seed(base_seed, s, i)` (a stable sha256 derivation — Python's
  salted `hash()` is never used) for *every* matchup in a tournament. All
  agents face the same battles, so matchup comparisons are paired and much
  tighter than independent sampling at the same game count.
- **Noise floor.** Before believing "A beats B by 3%", run `noise-floor`:
  an agent against itself "should" score 0.500, and the measured deviation
  with its CI is the resolution limit of your experiment. The pair-level
  numbers also demonstrate what mirroring buys: deterministic self-play
  pairs are exact mirrors, so their paired noise is zero.
- **Skill curve.** `skill-curve` injects an `eps` rate of random moves and
  measures the cost. It calibrates how much decisions matter on these maps —
  if eps=0.2 barely moved the score, you couldn't expect agent improvements
  to show up either. (Here a 5% blunder rate already costs ~3 points of
  score and 50% costs ~29, so decisions matter.)
- **SPRT.** For iterating on `WeightedAgent` parameters, the sequential
  probability ratio test streams mirrored pairs and stops as soon as the
  evidence crosses the alpha/beta bounds — usually far earlier than a
  fixed-n experiment with the same error guarantees.

Every game is logged as one JSONL row (scenario, seed, agent configs, action
list, winner, rounds, final state hash), and `tactica replay` re-simulates
any row and asserts the hash — reproducibility is enforced, not assumed.

## Non-goals

No spells, morale/luck, hex grids, GUIs, RL training loops, or external game
engines. The action encoding and `observe()` are designed so a
PettingZoo/Gymnasium wrapper can be added later without touching the core.
