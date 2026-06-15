# TODO / Ideas

Deferred work items, mostly fallout from the perk system (CHARGE,
MELEE_PENALTY) and the speed-derived turn order.

## Agent charge-awareness — remaining work

Damage previews now model `Perk.CHARGE`: `battle.expected_damage` (shared by
`WeightedAgent` and the dashboard) takes `moved`, and `action_features` feeds
it the chosen approach's BFS distance, so the `damage_dealt` / `kill` features
value charging sides correctly. `HeuristicAgent` charges via `default_melee`.
What's left:

- **Re-validate the shipped weights via SPRT.** DONE (ordering check): on the
  charge-aware code, `tactica sprt --candidate weights/default.json --baseline
  weights/conservative.json` gives default ~+15.8 elo over conservative
  (WDL 1567/2/1431 over 3000 games, LLR +2.68 toward the +2.944 H1 bound) --
  consistent with the documented ~+11.5, so the charge change preserved the
  weight ordering. NOT yet checked: absolute non-regression vs a *pre-charge
  build* (the weights file is unchanged; only the code is, so SPRT can't pit
  old code against new). If desired, snapshot the pre-charge engine, regenerate
  its default agent, and SPRT new-vs-old. Going forward, ladder win-rates are
  guarded by `tests/test_strength_regression.py` (banded; re-bless on
  intentional changes).
- Optionally add an explicit feature (`charge_available` or raw `cells_moved`)
  so the linear model can value setting up a charge directly, not just via the
  doubled `damage_dealt`.
- Flanking / next-turn-exposure features (pick the approach side that minimizes
  retaliation next round), the original motivation for first-class directional
  melee in the search agents.

## Perk-aware value function

`UnitStats.value` (units.py) and `stack_value` (agents/heuristic.py) ignore
perks: CHARGE makes cavalry worth more than its raw stats, MELEE_PENALTY
makes archers worth less when fights go to melee. Affects MCTS eval,
heuristic targeting, and the `target_value` feature. Needs empirical
calibration (e.g. fit values from tournament outcomes) rather than a
hand-tuned multiplier.

## Speed buffs/debuffs (haste/slow)

Turn order is now derived from speed (HoMM3 model). When buffs arrive:

- put modifiers on `Stack` (e.g. `Stack.effective_speed()`), never on the
  frozen shared `UnitStats`;
- route `_order_key` / `_wait_order_key` and `reachable()` through the
  effective value;
- decided rule: the queue is built at round start only — a mid-round speed
  change affects the *next* round's order, never the current queue;
- buffs become part of battle state: include them in `clone()`,
  `state_hash()`, and `observe()` (new feature plane).

## Directional melee follow-ups

- **MCTS doesn't search approach sides.** `MCTSAgent._root_arms` collapses the
  8 melee directions to one charge-aware `default_melee` arm per target, because
  flat-UCT with short random rollouts has no positional eval to rank sides
  beyond charge. A nested per-direction scorer (or intra-target progressive
  widening) is the upgrade, gated on positional eval existing first.
- **Double `reachable()` per melee step.** `Battle.step` validates (which calls
  `reachable`) and then calls `reachable` again to compute `moved`. Harmless on
  the 99-cell board and the MCTS hot path (`playout`) short-circuits adjacent
  strikes, but could be threaded through if step throughput ever matters.

## MCTS strength: rollout policy (prototyped)

Measured: flat-UCT with the default biased-random rollout is **at or below the
heuristic regardless of depth** (vs heuristic: 32 sims 0.30, 512 0.42, 1024
0.25 -- small samples, trend flat-to-down). Ruled out the two-player
sign/perspective bug: flat UCT has no minimax backup to negate, the rollout
value is taken consistently from the root player's side, and `mcts` beats
`random` 0.83 (a sign-bugged search could not). The ceiling is the rollout +
leaf evaluator.

Prototyped the fix: `MCTSAgent(rollout_policy="heuristic")` (epsilon-greedy
HeuristicAgent rollout for both sides) **beats the heuristic 0.71 at just 64
sims** on open_field+skirmish, where random-rollout mcts:512 managed 0.42.
Remaining work:

- It is ~10s/game (the heuristic runs every rollout step) -- a strength
  experiment, not a default. Profile / cap rollout length, or memoize.
- Tune `rollout_epsilon` and sims; SPRT it. Currently only constructible
  programmatically -- wire a spec form into `make_agent` to use it from the
  tournament/SPRT CLI.
- A positional leaf eval (beyond material) is the complementary lever; pairs
  with the "MCTS doesn't search approach sides" item above.
