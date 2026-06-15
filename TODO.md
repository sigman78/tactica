# TODO / Ideas

Deferred work items, mostly fallout from the perk system (CHARGE,
MELEE_PENALTY) and the speed-derived turn order.

## Distance-aware agent features (training)

`WeightedAgent`'s `expected_damage` (agents/weighted.py) does not model
`Perk.CHARGE`, so the `damage_dealt` / `kill` / `damage_received` features
understate charging cavalry strikes and the weights can't learn to set up
charges. Fixing it needs:

- thread the melee approach distance (already returned by
  `Battle._melee_approach`) into `action_features`;
- possibly a new feature (`charge_available` or raw `cells_moved`) so the
  linear model can value distance directly;
- retune/revalidate weights via SPRT afterwards — the current shipped
  weights predate charge.

The scripted `HeuristicAgent` is equally blind: it picks melee targets by
stack value only and will happily attack from adjacent instead of backing
off to charge. MCTS discovers charges through rollouts, so it is the
benchmark to compare against.

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
