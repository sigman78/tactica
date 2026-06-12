# CLI reference

One entry point, seven subcommands. Agent specs used everywhere:
`random`, `heuristic`, `weighted[:weights.json]`,
`mcts[:SIMS[:C_UCT[:ROLLOUT_CAP]]]`, and `epsilon:EPS:INNER`
(e.g. `epsilon:0.1:heuristic`). See [agents.md](agents.md) for what each
agent does.

## `tactica play` — run one battle

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

## `tactica tournament` — round-robin over mirrored pairs

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

## `tactica noise-floor` — the luck baseline

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

## `tactica skill-curve` — how much do decisions matter?

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

## `tactica sprt` — sequential testing for weight changes

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

## `tactica replay` — byte-identical re-simulation

```
$ uv run tactica replay --file replays/test.jsonl
griffin_raid seed=9: side 1 (mcts:16) wins after 3 rounds, hash=7b112c15bdf6bff4
replay OK: final state hash matches (7b112c15bdf6bff4)
```

Re-simulates a logged game from (scenario, seed, action list) and asserts
the final state hash matches the log. `--render` replays the ASCII board
turn by turn; `--index N` picks a row from a multi-game log such as
`results.jsonl`.

## `tactica web` — interactive dashboard

```
$ uv run tactica web
tactica dashboard on http://127.0.0.1:8321
```

A local single-page war room over the same evaluation tooling (needs the
`web` extra, included in a plain `uv sync`):

- **Experiments from presets** — editable JSON presets in `experiments/`
  (a few ship with the repo: quick smoke, full ladder, the SPRT run that
  promoted the default weights, ...). Tweak in a form or as raw JSON,
  save back, run.
- **Live runs over SSE** — tournament pair-score matrix fills in as a
  heatmap while games stream back; SPRT draws its LLR trajectory against
  the accept bounds in real time; skill-curve and noise-floor plot as
  each point lands. Progress, throughput, streaming logs, cooperative
  cancel. Charts are hand-rolled SVG; the page works fully offline.
- **Final reports** — OpenSkill rating bars, per-scenario breakdowns,
  verdict banners; tournaments can also write the usual JSONL game log.
- **Replay viewer** — step/scrub/autoplay any logged game (from a
  dashboard run or any `*.jsonl` on disk) on an SVG board with stack
  counts, top-creature HP and the active stack highlighted. Every replay
  is hash-verified against the log before it renders, same as
  `tactica replay`.
- **Play vs an agent** — playtest any agent spec yourself, turn by turn,
  on the same board: click a highlighted cell to move, click an enemy in
  reach to attack (hover for the average-damage forecast and retaliation
  warning), Wait/Defend buttons, live initiative queue and a narrated
  battle log with damage and kills. Finished battles save as
  hash-verified replays (`replays/human-*.jsonl`) you can re-watch or
  feed back through `tactica replay`.

Units render as pixel-art portraits (`src/tactica/web/static/units/`),
pre-generated with Gemini image generation via
`scripts/gen_unit_icons.py` (needs `GEMINI_API_KEY`; the shipped icons
are committed, so regeneration is only needed to restyle them).

The dashboard is a thin layer: jobs reuse the exact CRN seed derivation,
mirrored-pair runner and statistics as the CLI, so a result you see in the
browser is byte-for-byte the result you would get from the equivalent
command.
