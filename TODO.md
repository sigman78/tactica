# TODO / Ideas

Deferred work items, mostly fallout from the perk system (CHARGE,
MELEE_PENALTY) and the speed-derived turn order.

## Agent charge-awareness — remaining work

Damage previews now model `Perk.CHARGE`: `battle.expected_damage` (shared by
`WeightedAgent` and the dashboard) takes `moved`, and `action_features` feeds
it the chosen approach's BFS distance, so the `damage_dealt` / `kill` features
value charging sides correctly. `HeuristicAgent` charges via `default_melee`.
What's left:

- **Re-validate the shipped weights via SPRT.** `weights_default.json` was
  tuned before charge was modelled (and before the float->int change in
  `expected_damage`), so `WeightedAgent` now scores melee differently. Run
  `tactica sprt --candidate weights/default.json --baseline <pre-charge copy>`
  and retune if it regressed.
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
