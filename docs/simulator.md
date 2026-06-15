# Simulator API contract

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

- **Action encoding**: `action_id = action_type * 99 + target_cell` with 12
  types `{MOVE, MELEE_N…MELEE_NW (8 directional melee), RANGED_ATTACK, WAIT,
  DEFEND}`. A melee action names the target's cell and the type names the
  approach side; the engine resolves the approach cell (`target + offset[dir]`)
  and sets the charge distance from it. WAIT/DEFEND are canonical at cell 0;
  non-canonical ids are rejected. The mask aligns with this encoding exactly,
  so an RL policy head of size 1188 plugs straight in.
- **Observation planes** (C=18, H=9, W=11): per-side unit-type one-hots
  (10), normalized stack count (2), normalized top-creature HP (2), active
  unit (1), obstacles (1), constant planes for round number and side to move
  (2).
- **Randomness**: a single `np.random.Generator` owned by the battle, seeded
  at construction. No `random` module, no global RNG, anywhere.
  `Scenario(deterministic=True)` replaces damage rolls with expected values —
  the same API with zero chance nodes (the turn-order tie-shuffle happens at
  construction and is fixed by the seed).
- **State hash**: `Battle.state_hash()` digests stacks, queue, round, and
  RNG state; replays must reproduce it byte-identically.

## Rules summary

11x9 squares, 8-neighborhood movement (BFS up to `speed`; flyers ignore
obstacles/units in transit but land on free cells). Stacks are
`(unit_type, count, top_hp)`; damage kills whole creatures, the remainder
dents the top one. HoMM damage: `dmg_roll * count`, modified 5% per point of
attack-defense difference, clamped to [0.3x, 3x]. Each round every living
stack acts once in speed order, HoMM3-style — turn order is derived from
speed, computed at round start only (ties broken by a shuffle seeded at
battle start); WAIT defers a stack once per round to a reverse-speed wait
phase; DEFEND grants +2 defense until the stack's next turn. Melee draws
one retaliation per defender per round; ranged attacks don't, but a shooter
with an adjacent enemy cannot shoot. A side with no stacks loses; 100
rounds is a draw.

Unit specials are **perks**: data flags on `UnitStats` (`Perk` enum in
`units.py`), each implemented at exactly one hook in `battle.py`.
Current perks: `MELEE_PENALTY` (archer) — any melee strike, attack or
retaliation, at x0.5; `CHARGE` (cavalry) — melee damage x2 when the
attacker travelled >= 2 cells as part of the attack action (BFS path
length, so walls count; retaliations and stand-and-fight strikes count as
0 cells).
