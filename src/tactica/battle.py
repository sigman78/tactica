"""The battle engine: a pure, framework-free turn-based rules core.

Rules summary (see TASK.md / README for the contract):

- 11x9 square grid, 8-neighborhood (Chebyshev) movement, BFS reachability.
- HoMM-style stacks: (unit_type, count, top_hp); damage kills whole creatures.
- Each round all living stacks act once in initiative order; ties are broken
  by a shuffle seeded at battle start and fixed for the whole battle.
- WAIT defers a stack (once per round) to a wait phase processed after all
  non-waiters, in *reverse* initiative order. DEFEND grants +2 defense until
  the stack's next turn.
- Melee triggers one retaliation per defender per round; ranged attacks don't.
- A ranged unit with an adjacent enemy cannot shoot; any melee strike by a
  ranged unit (attack or retaliation) does half damage.
- Battle ends when a side is wiped out, or in a draw after 100 rounds.

All randomness flows through one ``np.random.Generator`` owned by the battle.
``Scenario(deterministic=True)`` replaces damage rolls with expected values,
leaving zero chance nodes during play (the initiative tie-shuffle happens at
construction time and is fixed by the seed).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from tactica.actions import (
    BOARD_H,
    BOARD_W,
    N_ACTIONS,
    N_CELLS,
    Action,
    ActionType,
    cell_xy,
)
from tactica.scenario import Scenario
from tactica.units import GLYPHS, STATS, UnitStats, UnitType

ROUND_LIMIT = 100
DEFEND_BONUS = 2
DAMAGE_MOD_PER_POINT = 0.05
DAMAGE_MOD_MIN = 0.3
DAMAGE_MOD_MAX = 3.0
RANGED_MELEE_PENALTY = 0.5


@dataclass
class Stack:
    uid: int
    side: int
    unit_type: UnitType
    count: int
    top_hp: int
    cell: int
    has_waited: bool = False
    defending: bool = False
    retaliations_left: int = 1

    @property
    def stats(self) -> UnitStats:
        return STATS[self.unit_type]

    @property
    def alive(self) -> bool:
        return self.count > 0

    @property
    def total_hp(self) -> int:
        return (self.count - 1) * self.stats.hp + self.top_hp

    def effective_defense(self) -> int:
        return self.stats.defense + (DEFEND_BONUS if self.defending else 0)


def adjacent_cells(cell: int) -> list[int]:
    x, y = cell_xy(cell)
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_W and 0 <= ny < BOARD_H:
                out.append(ny * BOARD_W + nx)
    return out


def chebyshev(a: int, b: int) -> int:
    ax, ay = cell_xy(a)
    bx, by = cell_xy(b)
    return max(abs(ax - bx), abs(ay - by))


class Battle:
    """Mutable battle state. Use :meth:`clone` for search."""

    def __init__(self) -> None:  # use from_scenario / clone
        self.scenario: Scenario
        self.rng: np.random.Generator
        self.stacks: dict[int, Stack] = {}
        self.round: int = 0
        self.queue: list[int] = []
        self.waiters: list[int] = []
        self.tiebreak: dict[int, int] = {}
        self._draw: bool = False
        self._n_alive: list[int] = [0, 0]

    # ------------------------------------------------------------------ #
    # Construction

    @classmethod
    def from_scenario(cls, scenario: Scenario, seed: int) -> "Battle":
        b = cls()
        b.scenario = scenario
        b.rng = np.random.Generator(np.random.PCG64(seed))
        uid = 0
        for side, army in ((0, scenario.army0), (1, scenario.army1)):
            for slot in army:
                stats = STATS[slot.unit_type]
                b.stacks[uid] = Stack(uid, side, slot.unit_type, slot.count,
                                      stats.hp, slot.start_cell)
                b._n_alive[side] += 1
                uid += 1
        # Initiative tiebreak: one seeded shuffle, fixed for the whole battle.
        perm = b.rng.permutation(len(b.stacks))
        b.tiebreak = {u: int(perm[i]) for i, u in enumerate(b.stacks)}
        b._start_round()
        return b

    def reseed(self, seed: int) -> None:
        """Replace the battle RNG. Search agents call this on clones so each
        simulation samples fresh chance outcomes instead of replaying the
        original RNG stream."""
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def clone(self) -> "Battle":
        b = Battle()
        b.scenario = self.scenario
        b.rng = np.random.Generator(np.random.PCG64())
        b.rng.bit_generator.state = self.rng.bit_generator.state
        b.stacks = {u: replace(s) for u, s in self.stacks.items()}
        b.round = self.round
        b.queue = list(self.queue)
        b.waiters = list(self.waiters)
        b.tiebreak = self.tiebreak  # immutable after construction
        b._draw = self._draw
        b._n_alive = list(self._n_alive)
        return b

    # ------------------------------------------------------------------ #
    # Turn / round machinery

    def _living(self, side: int | None = None) -> list[Stack]:
        return [s for s in self.stacks.values()
                if s.alive and (side is None or s.side == side)]

    def _order_key(self, uid: int) -> tuple[int, int]:
        return (-self.stacks[uid].stats.initiative, self.tiebreak[uid])

    def _wait_order_key(self, uid: int) -> tuple[int, int]:
        return (self.stacks[uid].stats.initiative, -self.tiebreak[uid])

    def _start_round(self) -> None:
        if self.round >= ROUND_LIMIT:
            self._draw = True
            self.queue = []
            self.waiters = []
            return
        self.round += 1
        for s in self._living():
            s.retaliations_left = 1
            s.has_waited = False
        self.queue = sorted((s.uid for s in self._living()), key=self._order_key)
        self.waiters = []

    def _refill_queue(self) -> None:
        """After an action: drop dead, advance phases/rounds as needed."""
        self.queue = [u for u in self.queue if self.stacks[u].alive]
        self.waiters = [u for u in self.waiters if self.stacks[u].alive]
        while not self.is_terminal() and not self.queue:
            if self.waiters:
                self.queue = sorted(self.waiters, key=self._wait_order_key)
                self.waiters = []
            else:
                self._start_round()

    def active_stack(self) -> Stack:
        if self.is_terminal():
            raise RuntimeError("battle is over; no active stack")
        return self.stacks[self.queue[0]]

    def current_player(self) -> int:
        return self.active_stack().side

    # ------------------------------------------------------------------ #
    # Terminal state

    def is_terminal(self) -> bool:
        return self._draw or self._n_alive[0] == 0 or self._n_alive[1] == 0

    def winner(self) -> int | None:
        """0 / 1, or None for a draw or an unfinished battle."""
        if not self.is_terminal():
            return None
        if self._n_alive[0] and not self._n_alive[1]:
            return 0
        if self._n_alive[1] and not self._n_alive[0]:
            return 1
        return None

    def returns(self) -> tuple[float, float]:
        if not self.is_terminal():
            raise RuntimeError("returns() on a non-terminal battle")
        w = self.winner()
        if w == 0:
            return (1.0, -1.0)
        if w == 1:
            return (-1.0, 1.0)
        return (0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Movement

    def occupied_cells(self) -> set[int]:
        return {s.cell for s in self._living()}

    def reachable(self, stack: Stack) -> dict[int, int]:
        """Cells the stack can end its move on -> distance. Includes its own
        cell at distance 0. Flyers ignore obstacles and units in transit but
        must land on a free cell."""
        blocked = self.scenario.obstacles | self.occupied_cells()
        speed = stack.stats.speed
        start = stack.cell
        if stack.stats.is_flyer:
            out = {start: 0}
            for cell in range(N_CELLS):
                d = chebyshev(start, cell)
                if 0 < d <= speed and cell not in blocked:
                    out[cell] = d
            return out
        out = {start: 0}
        frontier = [start]
        for dist in range(1, speed + 1):
            nxt: list[int] = []
            for c in frontier:
                for n in adjacent_cells(c):
                    if n not in out and n not in blocked:
                        out[n] = dist
                        nxt.append(n)
            if not nxt:
                break
            frontier = nxt
        return out

    def _melee_approach(self, attacker: Stack, target: Stack,
                        reach: dict[int, int] | None = None) -> int | None:
        """Cell the attacker strikes from, or None if unreachable.
        Deterministic: stay put if already adjacent, else the reachable
        adjacent cell with minimal (distance, cell index). ``reach`` is
        computed lazily when not supplied."""
        if chebyshev(attacker.cell, target.cell) == 1:
            return attacker.cell
        if reach is None:
            reach = self.reachable(attacker)
        options = [(reach[c], c) for c in adjacent_cells(target.cell) if c in reach and c != attacker.cell]
        return min(options)[1] if options else None

    def _enemy_adjacent(self, stack: Stack) -> bool:
        adj = set(adjacent_cells(stack.cell))
        return any(e.cell in adj for e in self._living(1 - stack.side))

    # ------------------------------------------------------------------ #
    # Legal actions

    def legal_actions(self) -> list[Action]:
        s = self.active_stack()
        reach = self.reachable(s)
        enemies = self._living(1 - s.side)
        actions: list[Action] = []
        for cell, dist in reach.items():
            if dist > 0:
                actions.append(Action(ActionType.MOVE, cell))
        for e in enemies:
            if self._melee_approach(s, e, reach) is not None:
                actions.append(Action(ActionType.MELEE_ATTACK, e.cell))
        if s.stats.is_ranged and not self._enemy_adjacent(s):
            for e in enemies:
                actions.append(Action(ActionType.RANGED_ATTACK, e.cell))
        if not s.has_waited:
            actions.append(Action(ActionType.WAIT))
        actions.append(Action(ActionType.DEFEND))
        return actions

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for a in self.legal_actions():
            mask[a.id] = True
        return mask

    # ------------------------------------------------------------------ #
    # Damage

    def _damage_roll(self, stats: UnitStats) -> float:
        if self.scenario.deterministic:
            return (stats.dmg_min + stats.dmg_max) / 2.0
        return float(self.rng.integers(stats.dmg_min, stats.dmg_max + 1))

    def compute_damage(self, attacker: Stack, defender: Stack,
                       melee: bool) -> int:
        stats = attacker.stats
        base = self._damage_roll(stats) * attacker.count
        diff = stats.attack - defender.effective_defense()
        factor = min(max(1.0 + DAMAGE_MOD_PER_POINT * diff, DAMAGE_MOD_MIN),
                     DAMAGE_MOD_MAX)
        if melee and stats.is_ranged:
            factor *= RANGED_MELEE_PENALTY
        return max(1, int(base * factor))

    def _apply_damage(self, target: Stack, damage: int) -> None:
        pool = target.total_hp - damage
        if pool <= 0:
            target.count = 0
            target.top_hp = 0
            self._n_alive[target.side] -= 1
            return
        hp = target.stats.hp
        target.count = (pool + hp - 1) // hp
        target.top_hp = pool - (target.count - 1) * hp

    def _melee_strike(self, attacker: Stack, defender: Stack,
                      retaliation: bool) -> None:
        self._apply_damage(defender, self.compute_damage(attacker, defender, melee=True))
        if (not retaliation and defender.alive and defender.retaliations_left > 0):
            defender.retaliations_left -= 1
            self._melee_strike(defender, attacker, retaliation=True)

    # ------------------------------------------------------------------ #
    # Stepping

    def step(self, action: Action) -> None:
        if self.is_terminal():
            raise RuntimeError("battle is over")
        s = self.active_stack()
        self._validate(s, action)
        s.defending = False  # defend bonus lasts until the stack's next turn

        if action.type == ActionType.WAIT:
            s.has_waited = True
            self.queue.pop(0)
            self.waiters.append(s.uid)
            self._refill_queue()
            return

        if action.type == ActionType.DEFEND:
            s.defending = True
        elif action.type == ActionType.MOVE:
            s.cell = action.target_cell
        elif action.type == ActionType.MELEE_ATTACK:
            target = self._stack_at(action.target_cell)
            assert target is not None
            approach = self._melee_approach(s, target)
            assert approach is not None
            s.cell = approach
            self._melee_strike(s, target, retaliation=False)
        elif action.type == ActionType.RANGED_ATTACK:
            target = self._stack_at(action.target_cell)
            assert target is not None
            self._apply_damage(target, self.compute_damage(s, target, melee=False))

        if self.queue and self.queue[0] == s.uid:
            self.queue.pop(0)
        self._refill_queue()

    def playout(self, max_steps: int = 200, attack_bias: float = 0.6,
                chase_bias: float = 0.6) -> int:
        """Play a fast, biased-random rollout in place; returns steps taken.

        The rollout policy samples legal actions cheaply: single-cell moves,
        attacks on adjacent enemies (or any shot when unblocked), and DEFEND.
        With probability ``attack_bias`` an available attack is taken, and
        moves prefer closing on the nearest enemy with probability
        ``chase_bias`` -- otherwise pure random walks diffuse for the whole
        cap and rollouts carry no signal. All sampling flows through the
        battle's own RNG; search agents reseed clones first (see
        :meth:`reseed`)."""
        rng = self.rng
        obstacles = self.scenario.obstacles
        steps = 0
        while steps < max_steps and not self.is_terminal():
            s = self.active_stack()
            living = [st for st in self.stacks.values() if st.count > 0]
            occupied = {st.cell for st in living}
            enemies = [e for e in living if e.side != s.side]
            adj = adjacent_cells(s.cell)
            adj_set = set(adj)
            melee = [e.cell for e in enemies if e.cell in adj_set]
            if s.stats.is_ranged and not melee:
                attacks = [Action(ActionType.RANGED_ATTACK, e.cell)
                           for e in enemies]
            else:
                attacks = [Action(ActionType.MELEE_ATTACK, c) for c in melee]
            if attacks and rng.random() < attack_bias:
                action = attacks[int(rng.integers(len(attacks)))]
            else:
                moves = [c for c in adj
                         if c not in occupied and c not in obstacles]
                if moves and not attacks and rng.random() < chase_bias:
                    target = min(enemies,
                                 key=lambda e: chebyshev(s.cell, e.cell)).cell
                    action = Action(ActionType.MOVE,
                                    min(moves, key=lambda c: chebyshev(c, target)))
                else:
                    k = int(rng.integers(len(moves) + len(attacks) + 1))
                    if k < len(moves):
                        action = Action(ActionType.MOVE, moves[k])
                    elif k < len(moves) + len(attacks):
                        action = attacks[k - len(moves)]
                    else:
                        action = Action(ActionType.DEFEND)
            self.step(action)
            steps += 1
        return steps

    def _stack_at(self, cell: int) -> Stack | None:
        for s in self.stacks.values():
            if s.cell == cell and s.count > 0:
                return s
        return None

    def _validate(self, s: Stack, action: Action) -> None:
        def illegal(why: str) -> None:
            raise ValueError(f"illegal action {action!r} for stack {s.uid}: {why}")

        t = action.type
        if t == ActionType.WAIT:
            if s.has_waited:
                illegal("already waited this round")
            return
        if t == ActionType.DEFEND:
            return
        if not 0 <= action.target_cell < N_CELLS:
            illegal("cell off board")
        if t == ActionType.MOVE:
            if action.target_cell == s.cell:
                illegal("cell not reachable")
            # Fast path: an adjacent free cell is always reachable (speed >= 1).
            if (chebyshev(s.cell, action.target_cell) == 1
                    and action.target_cell not in self.scenario.obstacles
                    and self._stack_at(action.target_cell) is None):
                return
            if action.target_cell not in self.reachable(s):
                illegal("cell not reachable")
            return
        target = self._stack_at(action.target_cell)
        if target is None or target.side == s.side:
            illegal("no enemy stack on target cell")
        if t == ActionType.MELEE_ATTACK:
            if self._melee_approach(s, target) is None:
                illegal("cannot reach a cell adjacent to target")
        elif t == ActionType.RANGED_ATTACK:
            if not s.stats.is_ranged:
                illegal("unit is not ranged")
            if self._enemy_adjacent(s):
                illegal("blocked: enemy adjacent")

    # ------------------------------------------------------------------ #
    # Observation

    def observe(self) -> np.ndarray:
        """Float32 feature planes (C, H, W), C = 18:

        0-4   side-0 unit-type one-hots      5-9   side-1 unit-type one-hots
        10/11 normalized stack count (s0/s1) 12/13 normalized top-creature HP
        14    active-unit plane              15    obstacles
        16    round / ROUND_LIMIT (const)    17    side to move (const)
        """
        planes = np.zeros((18, BOARD_H, BOARD_W), dtype=np.float32)
        for s in self._living():
            x, y = cell_xy(s.cell)
            planes[s.side * 5 + int(s.unit_type), y, x] = 1.0
            planes[10 + s.side, y, x] = min(s.count, 100) / 100.0
            planes[12 + s.side, y, x] = s.top_hp / s.stats.hp
        if not self.is_terminal():
            ax, ay = cell_xy(self.active_stack().cell)
            planes[14, ay, ax] = 1.0
            planes[17, :, :] = float(self.current_player())
        for c in self.scenario.obstacles:
            x, y = cell_xy(c)
            planes[15, y, x] = 1.0
        planes[16, :, :] = self.round / ROUND_LIMIT
        return planes

    # ------------------------------------------------------------------ #
    # Hashing / rendering

    def state_hash(self) -> str:
        parts = [str(self.round), ",".join(map(str, self.queue)),
                 ",".join(map(str, self.waiters)), str(int(self._draw))]
        for u in sorted(self.stacks):
            s = self.stacks[u]
            parts.append(f"{u}:{s.side}:{int(s.unit_type)}:{s.count}:{s.top_hp}:"
                         f"{s.cell}:{int(s.has_waited)}:{int(s.defending)}:"
                         f"{s.retaliations_left}")
        bg = self.rng.bit_generator.state
        rng_state = bg["state"]
        # has_uint32/uinteger: PCG64's buffered half-word also shapes future
        # rolls, so it belongs in the hash.
        parts.append(f"{rng_state['state']}:{rng_state['inc']}:"
                     f"{bg['has_uint32']}:{bg['uinteger']}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def render(self) -> str:
        grid = [["." for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        for c in self.scenario.obstacles:
            x, y = cell_xy(c)
            grid[y][x] = "#"
        for s in self._living():
            x, y = cell_xy(s.cell)
            glyph = GLYPHS[s.unit_type]
            grid[y][x] = glyph if s.side == 0 else glyph.lower()
        lines = [f"Round {self.round}"]
        header = "   " + " ".join(f"{x:2d}" for x in range(BOARD_W))
        lines.append(header)
        for y in range(BOARD_H):
            lines.append(f"{y:2d}  " + "  ".join(grid[y]))
        for side in (0, 1):
            descr = []
            active_uid = self.queue[0] if self.queue else -1
            for s in sorted(self._living(side), key=lambda s: s.uid):
                mark = "*" if s.uid == active_uid else " "
                flags = "".join(f for f, on in (("d", s.defending), ("w", s.has_waited)) if on)
                descr.append(f"{mark}{s.stats.name} x{s.count} ({s.top_hp}/{s.stats.hp}hp)"
                             f"@{cell_xy(s.cell)}{(' [' + flags + ']') if flags else ''}")
            lines.append(f"Side {side}: " + ("  ".join(descr) if descr else "wiped out"))
        if self.is_terminal():
            w = self.winner()
            lines.append(f"RESULT: {'draw' if w is None else f'side {w} wins'}")
        return "\n".join(lines)
