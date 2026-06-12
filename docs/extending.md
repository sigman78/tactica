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
use the new unit by name immediately.

**New weights** — copy `weights/default.json`, tune, then validate with
`tactica sprt --candidate yours.json --baseline weights/default.json`
(see [evaluation.md](evaluation.md) for why SPRT and not a fixed-n run).
