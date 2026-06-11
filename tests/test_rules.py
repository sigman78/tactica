"""Rules unit tests: damage formula, retaliation, WAIT ordering, ranged
blocking, flyer movement, stack-kill arithmetic, defend timing."""
from __future__ import annotations

import dataclasses

import pytest

from tactica.actions import Action, ActionType, xy_cell
from tactica.battle import Battle, Stack
from tactica.scenario import ArmySlot, Scenario
from tactica.units import STATS, UnitType
from tests.conftest import battle_of, duel_scenario

P, A, G, S, C = (UnitType.PIKEMAN, UnitType.ARCHER, UnitType.GRIFFIN,
                 UnitType.SWORDSMAN, UnitType.CAVALRY)


def det_battle(**kwargs) -> Battle:
    return battle_of(duel_scenario(**kwargs))


def make_stack(uid: int, side: int, ut: UnitType, count: int,
               cell: int = 0) -> Stack:
    return Stack(uid, side, ut, count, STATS[ut].hp, cell)


# --------------------------------------------------------------------- #
# Damage formula


class TestDamageFormula:
    def battle(self) -> Battle:
        return det_battle(unit0=S, count0=1, cell0=(0, 0),
                          unit1=P, count1=1, cell1=(10, 8))

    def test_expected_value_with_attack_advantage(self) -> None:
        b = self.battle()
        # Swordsman avg 7.5, atk 10 vs def 5 -> x1.25 -> floor(9.375) = 9
        dmg = b.compute_damage(make_stack(0, 0, S, 1), make_stack(1, 1, P, 1),
                               melee=True)
        assert dmg == 9

    def test_defense_advantage_reduces_damage(self) -> None:
        b = self.battle()
        # Pikeman avg 2.0, atk 4 vs swordsman def 12 -> x0.6 -> floor(1.2) = 1
        dmg = b.compute_damage(make_stack(0, 0, P, 1), make_stack(1, 1, S, 1),
                               melee=True)
        assert dmg == 1

    def test_minimum_damage_is_one(self) -> None:
        b = self.battle()
        defender = make_stack(1, 1, S, 1)
        defender.defending = True  # def 14: pikeman 2.0 * 0.5 = 1.0 -> 1
        dmg = b.compute_damage(make_stack(0, 0, P, 1), defender, melee=True)
        assert dmg == 1

    def test_defend_bonus_lowers_damage(self) -> None:
        b = self.battle()
        attacker = make_stack(0, 0, S, 1)
        plain = make_stack(1, 1, P, 1)
        defending = make_stack(2, 1, P, 1)
        defending.defending = True
        # def 5 -> 9.375 -> 9; def 7 -> x1.15 -> 8.625 -> 8
        assert b.compute_damage(attacker, plain, melee=True) == 9
        assert b.compute_damage(attacker, defending, melee=True) == 8

    def test_ranged_unit_melees_at_half_damage(self) -> None:
        b = self.battle()
        archer, pike = make_stack(0, 0, A, 4), make_stack(1, 1, P, 1)
        # ranged: avg 2.5*4=10, atk 6 vs def 5 -> x1.05 -> 10.5 -> 10
        assert b.compute_damage(archer, pike, melee=False) == 10
        # melee: halved -> 5.25 -> 5
        assert b.compute_damage(archer, pike, melee=True) == 5

    def test_modifier_clamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        b = self.battle()
        monkeypatch.setitem(STATS, C, dataclasses.replace(STATS[C], attack=200))
        monkeypatch.setitem(STATS, P, dataclasses.replace(STATS[P], defense=300))
        # attack 200 vs archer def 3: would be x10.85, clamps to x3
        dmg = b.compute_damage(make_stack(0, 0, C, 1), make_stack(1, 1, A, 1),
                               melee=True)
        assert dmg == int(12.5 * 3.0)
        # archer atk 6 vs pikeman def 300: clamps to x0.3 -> 2.5*0.3 -> max(1,0) = 1
        dmg = b.compute_damage(make_stack(0, 0, A, 1), make_stack(1, 1, P, 1),
                               melee=False)
        assert dmg == 1

    def test_stochastic_roll_within_bounds(self) -> None:
        sc = duel_scenario(unit0=S, count0=1, cell0=(0, 0),
                           unit1=P, count1=1, cell1=(10, 8),
                           deterministic=False)
        stats = STATS[S]
        lo = int(stats.dmg_min * 1.25)
        hi = int(stats.dmg_max * 1.25)
        seen = set()
        for seed in range(40):
            b = battle_of(sc, seed)
            dmg = b.compute_damage(make_stack(0, 0, S, 1),
                                   make_stack(1, 1, P, 1), melee=True)
            assert lo <= dmg <= hi
            seen.add(dmg)
        assert len(seen) > 1  # rolls actually vary


# --------------------------------------------------------------------- #
# Stack kill arithmetic


class TestStackArithmetic:
    def test_partial_kill_with_remainder(self) -> None:
        b = det_battle(unit0=P, count0=20, cell0=(0, 0),
                       unit1=P, count1=1, cell1=(10, 8))
        target = make_stack(9, 1, P, 20)  # 200 hp pool
        b._apply_damage(target, 25)  # kills 2, 5 into the next
        assert (target.count, target.top_hp) == (18, 5)

    def test_exact_kill_boundary(self) -> None:
        b = det_battle(unit0=P, count0=20, cell0=(0, 0),
                       unit1=P, count1=1, cell1=(10, 8))
        target = make_stack(9, 1, P, 3)
        b._apply_damage(target, 10)  # exactly one creature
        assert (target.count, target.top_hp) == (2, 10)

    def test_damage_only_dents_top_creature(self) -> None:
        b = det_battle(unit0=P, count0=20, cell0=(0, 0),
                       unit1=P, count1=1, cell1=(10, 8))
        target = make_stack(9, 1, P, 5)
        b._apply_damage(target, 7)
        assert (target.count, target.top_hp) == (5, 3)

    def test_overkill_wipes_stack(self) -> None:
        b = det_battle(unit0=P, count0=20, cell0=(0, 0),
                       unit1=P, count1=1, cell1=(10, 8))
        target = make_stack(9, 1, P, 2)
        b._apply_damage(target, 9999)
        assert target.count == 0
        assert not target.alive


# --------------------------------------------------------------------- #
# Retaliation


def three_stack_battle() -> Battle:
    """Two side-0 swordsmen flanking one side-1 swordsman, all adjacent.
    Swordsman mirror keeps initiative ties resolved by the seeded shuffle."""
    sc = Scenario(
        name="retal",
        army0=(ArmySlot(S, 5, xy_cell(4, 4)), ArmySlot(S, 5, xy_cell(6, 4))),
        army1=(ArmySlot(S, 5, xy_cell(5, 4)),),
        deterministic=True,
    )
    return Battle.from_scenario(sc, seed=3)


class TestRetaliation:
    def test_melee_retaliates_once_per_round(self) -> None:
        b = three_stack_battle()
        defender = b.stacks[2]
        attackers = [b.stacks[0], b.stacks[1]]
        hp_before = {s.uid: s.total_hp for s in attackers}
        hits = 0
        # Drive a full round: every side-0 stack melees the side-1 stack.
        for _ in range(3):
            if b.is_terminal():
                break
            s = b.active_stack()
            if s.side == 0:
                b.step(Action(ActionType.MELEE_ATTACK, defender.cell))
                hits += 1
            else:
                b.step(Action(ActionType.DEFEND))
        assert hits == 2
        damaged = [s for s in attackers if s.total_hp < hp_before[s.uid]]
        assert len(damaged) == 1  # only the first attacker ate a retaliation

    def test_retaliation_resets_next_round(self) -> None:
        b = three_stack_battle()
        defender = b.stacks[2]
        rounds_seen = []
        retaliations = 0
        hp = {0: b.stacks[0].total_hp, 1: b.stacks[1].total_hp}
        for _ in range(40):
            if b.is_terminal() or b.round > 2:
                break
            s = b.active_stack()
            if s.side == 0 and defender.alive:
                before = s.total_hp
                b.step(Action(ActionType.MELEE_ATTACK, defender.cell))
                if s.total_hp < before:
                    retaliations += 1
                    rounds_seen.append(b.round)
            else:
                b.step(Action(ActionType.DEFEND))
        assert retaliations == 2  # one per round, across two rounds
        assert sorted(set(rounds_seen)) == rounds_seen  # distinct rounds

    def test_ranged_attack_draws_no_retaliation(self) -> None:
        b = battle_of(duel_scenario(unit0=A, count0=5, cell0=(0, 4),
                                    unit1=S, count1=5, cell1=(10, 4)))
        # Archer has higher initiative (5 vs... swordsman 5) -- tie. Find archer turn.
        while b.active_stack().unit_type != A:
            b.step(Action(ActionType.DEFEND))
        archer = b.active_stack()
        hp_before = archer.total_hp
        target = next(s for s in b.stacks.values() if s.side != archer.side)
        b.step(Action(ActionType.RANGED_ATTACK, target.cell))
        assert archer.total_hp == hp_before
        assert target.retaliations_left == 1


# --------------------------------------------------------------------- #
# WAIT ordering


class TestWaitOrdering:
    def setup_battle(self) -> Battle:
        # Cavalry init 8 acts before pikeman init 4.
        return battle_of(duel_scenario(unit0=C, count0=1, cell0=(0, 0),
                                       unit1=P, count1=1, cell1=(10, 8)))

    def test_waiter_acts_later_same_round(self) -> None:
        b = self.setup_battle()
        cav, pike = b.stacks[0], b.stacks[1]
        assert b.active_stack() is cav
        b.step(Action(ActionType.WAIT))
        assert b.round == 1
        assert b.active_stack() is pike
        b.step(Action(ActionType.DEFEND))
        # Back to the waiting cavalry, still round 1.
        assert b.round == 1
        assert b.active_stack() is cav
        b.step(Action(ActionType.DEFEND))
        assert b.round == 2

    def test_wait_only_once_per_round(self) -> None:
        b = self.setup_battle()
        b.step(Action(ActionType.WAIT))
        b.step(Action(ActionType.DEFEND))  # pikeman
        # Cavalry again, in the wait phase: WAIT must be illegal now.
        legal_types = {a.type for a in b.legal_actions()}
        assert ActionType.WAIT not in legal_types
        with pytest.raises(ValueError):
            b.step(Action(ActionType.WAIT))

    def test_wait_phase_runs_in_reverse_initiative(self) -> None:
        b = self.setup_battle()
        cav, pike = b.stacks[0], b.stacks[1]
        b.step(Action(ActionType.WAIT))  # cavalry waits
        b.step(Action(ActionType.WAIT))  # pikeman waits
        # Reverse initiative: pikeman (4) before cavalry (8).
        assert b.active_stack() is pike
        b.step(Action(ActionType.DEFEND))
        assert b.active_stack() is cav

    def test_wait_flag_resets_each_round(self) -> None:
        b = self.setup_battle()
        b.step(Action(ActionType.WAIT))
        b.step(Action(ActionType.DEFEND))
        b.step(Action(ActionType.DEFEND))
        assert b.round == 2
        assert ActionType.WAIT in {a.type for a in b.legal_actions()}


# --------------------------------------------------------------------- #
# Ranged blocking


class TestRangedBlocking:
    def test_shot_blocked_by_adjacent_enemy(self) -> None:
        b = battle_of(duel_scenario(unit0=A, count0=5, cell0=(5, 4),
                                    unit1=S, count1=5, cell1=(6, 4)))
        while b.active_stack().unit_type != A:
            b.step(Action(ActionType.DEFEND))
        types = {a.type for a in b.legal_actions()}
        assert ActionType.RANGED_ATTACK not in types
        assert ActionType.MELEE_ATTACK in types
        with pytest.raises(ValueError):
            b.step(Action(ActionType.RANGED_ATTACK, xy_cell(6, 4)))

    def test_shot_available_at_distance_any_range(self) -> None:
        b = battle_of(duel_scenario(unit0=A, count0=5, cell0=(0, 0),
                                    unit1=S, count1=5, cell1=(10, 8)))
        while b.active_stack().unit_type != A:
            b.step(Action(ActionType.DEFEND))
        shots = [a for a in b.legal_actions()
                 if a.type == ActionType.RANGED_ATTACK]
        assert [a.target_cell for a in shots] == [xy_cell(10, 8)]

    def test_melee_unit_never_shoots(self) -> None:
        b = battle_of(duel_scenario(unit0=S, count0=5, cell0=(0, 0),
                                    unit1=A, count1=5, cell1=(10, 8)))
        while b.active_stack().unit_type != S:
            b.step(Action(ActionType.DEFEND))
        assert all(a.type != ActionType.RANGED_ATTACK for a in b.legal_actions())


# --------------------------------------------------------------------- #
# Flyer movement


class TestFlyerMovement:
    WALL = frozenset(xy_cell(3, y) for y in range(9))  # full column

    def test_flyer_crosses_obstacle_wall(self) -> None:
        b = battle_of(duel_scenario(unit0=G, count0=1, cell0=(0, 4),
                                    unit1=S, count1=1, cell1=(10, 4),
                                    obstacles=self.WALL))
        while b.active_stack().unit_type != G:
            b.step(Action(ActionType.DEFEND))
        targets = {a.target_cell for a in b.legal_actions()
                   if a.type == ActionType.MOVE}
        beyond_wall = {c for c in targets if c % 11 > 3}
        assert beyond_wall  # griffin lands past the wall
        assert not targets & self.WALL  # but never on an obstacle

    def test_walker_blocked_by_obstacle_wall(self) -> None:
        b = battle_of(duel_scenario(unit0=C, count0=1, cell0=(0, 4),
                                    unit1=S, count1=1, cell1=(10, 4),
                                    obstacles=self.WALL))
        while b.active_stack().unit_type != C:
            b.step(Action(ActionType.DEFEND))
        targets = {a.target_cell for a in b.legal_actions()
                   if a.type == ActionType.MOVE}
        assert targets
        assert all(c % 11 < 3 for c in targets)  # everything stays west

    def test_flyer_range_is_chebyshev_speed(self) -> None:
        b = battle_of(duel_scenario(unit0=G, count0=1, cell0=(0, 0),
                                    unit1=S, count1=1, cell1=(10, 8)))
        while b.active_stack().unit_type != G:
            b.step(Action(ActionType.DEFEND))
        speed = STATS[G].speed
        targets = {a.target_cell for a in b.legal_actions()
                   if a.type == ActionType.MOVE}
        assert all(max(c % 11, c // 11) <= speed for c in targets)
        assert xy_cell(speed, speed) in targets


# --------------------------------------------------------------------- #
# Defend timing


def test_defend_lasts_until_own_next_turn() -> None:
    # Swordsman (init 5) acts before pikeman (init 4) every round.
    b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(5, 4),
                                unit1=P, count1=1, cell1=(6, 4)))
    sword, pike = b.stacks[0], b.stacks[1]
    b.step(Action(ActionType.DEFEND))  # swordsman
    b.step(Action(ActionType.DEFEND))  # pikeman
    assert pike.defending
    # Round 2: swordsman strikes the still-defending pikeman: 8 not 9 damage.
    assert b.active_stack() is sword
    b.step(Action(ActionType.MELEE_ATTACK, pike.cell))
    assert pike.top_hp == 10 - 8
    # Pikeman's own turn clears its defend flag.
    assert b.active_stack() is pike
    b.step(Action(ActionType.MELEE_ATTACK, sword.cell))
    assert not pike.defending


# --------------------------------------------------------------------- #
# Terminal conditions


def test_battle_ends_when_side_wiped() -> None:
    b = battle_of(duel_scenario(unit0=C, count0=10, cell0=(5, 4),
                                unit1=P, count1=1, cell1=(6, 4)))
    while b.active_stack().unit_type != C:
        b.step(Action(ActionType.DEFEND))
    b.step(Action(ActionType.MELEE_ATTACK, xy_cell(6, 4)))  # 125+ dmg vs 10 hp
    assert b.is_terminal()
    assert b.winner() == 0
    assert b.returns() == (1.0, -1.0)


def test_draw_at_round_cap() -> None:
    b = battle_of(duel_scenario(unit0=S, count0=5, cell0=(0, 0),
                                unit1=S, count1=5, cell1=(10, 8)))
    steps = 0
    while not b.is_terminal():
        b.step(Action(ActionType.DEFEND))
        steps += 1
        assert steps < 1000
    assert b.round == 100
    assert b.winner() is None
    assert b.returns() == (0.0, 0.0)
