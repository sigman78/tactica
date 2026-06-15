# Extending

**New agent** — subclass `tactica.agents.base.Agent`, implement
`act(battle) -> Action` using only the public API (search agents must use
`clone()` + `reseed()`, see [simulator.md](simulator.md)), and register a
spec name in `tactica.agents.make_agent`. Every CLI command then accepts it.

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
use the new unit by name immediately. Specials are declared via the
`perks` field (e.g. `perks=frozenset({Perk.CHARGE})`).

**New perk** — add a member to `Perk` in `units.py`, then implement it at
the matching hook in `battle.py`. Damage perks go in `battle.damage_factor`
**only** — `Battle.compute_damage` (the live roll), `battle.expected_damage`
(the agent/UI average-roll preview), and therefore `WeightedAgent` and the
dashboard all delegate to it, so there is a single implementation point and
no mirrors to keep in sync. Perks that change action *legality* need a hook
in `Battle.legal_actions` instead.

**Directional melee** — melee is encoded as one action type per approach
side (`MELEE_N…MELEE_NW`); `Action(MELEE_<dir>, target_cell)` strikes the
target from `target_cell + offset[dir]`. Agents that just want "attack this
target" call `Battle.default_melee(attacker, target)` for a charge-aware
direction. Flat-UCT MCTS collapses the 8 directions to that single default
arm per target (it has no positional eval to rank sides); the action space
stays first-class for the human UI and any future learned policy.

**New weights** — copy `weights/default.json`, tune, then validate with
`tactica sprt --candidate yours.json --baseline weights/default.json`
(see [evaluation.md](evaluation.md) for why SPRT and not a fixed-n run).
