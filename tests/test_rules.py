"""Rules unit tests: damage formula, retaliation, WAIT ordering, ranged
blocking, flyer movement, stack-kill arithmetic, defend timing."""
from __future__ import annotations

import dataclasses

import pytest

from tactica.actions import (
    Action,
    ActionType,
    MELEE_TYPES,
    is_melee,
    xy_cell,
)
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


def default_melee_action(b, target_cell: int) -> Action:
    """The battle's charge-aware default approach against the stack on
    ``target_cell``, as a concrete directional Action."""
    s = b.active_stack()
    target = b._stack_at(target_cell)
    return Action(b.default_melee(s, target), target_cell)


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


class TestDirectionalMelee:
    def test_all_reachable_sides_are_legal(self) -> None:
        b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(0, 4),
                                    unit1=P, count1=1, cell1=(3, 4)))
        while b.active_stack().unit_type != S:
            b.step(Action(ActionType.DEFEND))
        target = xy_cell(3, 4)
        melee = {a.type for a in b.legal_actions()
                 if is_melee(a.type) and a.target_cell == target}
        assert melee == set(MELEE_TYPES)

    def test_offboard_side_is_illegal(self) -> None:
        b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(5, 4),
                                    unit1=P, count1=1, cell1=(0, 0)))
        while b.active_stack().unit_type != S:
            b.step(Action(ActionType.DEFEND))
        target = xy_cell(0, 0)
        dirs = {a.type for a in b.legal_actions()
                if is_melee(a.type) and a.target_cell == target}
        assert dirs == {ActionType.MELEE_SE, ActionType.MELEE_S,
                        ActionType.MELEE_E}

    def test_strike_ends_attacker_on_chosen_side(self) -> None:
        b = det_battle(unit0=S, count0=1, cell0=(0, 4),
                       unit1=P, count1=10, cell1=(5, 4))
        assert b.active_stack().unit_type == S
        b.step(Action(ActionType.MELEE_W, xy_cell(5, 4)))
        assert b.stacks[0].cell == xy_cell(4, 4)

    def test_direction_controls_charge(self) -> None:
        far = det_battle(unit0=C, count0=1, cell0=(0, 4),
                         unit1=P, count1=10, cell1=(5, 4))
        far.step(Action(ActionType.MELEE_W, xy_cell(5, 4)))
        charged = far.stacks[1].total_hp

        near = det_battle(unit0=C, count0=1, cell0=(4, 4),
                          unit1=P, count1=10, cell1=(5, 4))
        near.step(Action(ActionType.MELEE_W, xy_cell(5, 4)))
        not_charged = near.stacks[1].total_hp
        assert charged < not_charged

    def test_unreachable_side_raises(self) -> None:
        wall = frozenset(xy_cell(6, y) for y in range(9))
        b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(0, 4),
                                    unit1=P, count1=1, cell1=(5, 4),
                                    obstacles=wall))
        while b.active_stack().unit_type != S:
            b.step(Action(ActionType.DEFEND))
        with pytest.raises(ValueError):
            b.step(Action(ActionType.MELEE_E, xy_cell(5, 4)))

    def test_default_melee_prefers_charge_for_cavalry(self) -> None:
        b = det_battle(unit0=C, count0=1, cell0=(0, 4),
                       unit1=P, count1=10, cell1=(5, 4))
        s = b.active_stack()
        target = b.stacks[1]
        d = b.default_melee(s, target)
        approach = b.approach_cell(target.cell, d)
        assert b.reachable(s)[approach] >= 2

    def test_default_melee_minimal_for_non_charger(self) -> None:
        b = det_battle(unit0=S, count0=1, cell0=(4, 4),
                       unit1=P, count1=10, cell1=(5, 4))
        s = b.active_stack()
        target = b.stacks[1]
        d = b.default_melee(s, target)
        assert b.approach_cell(target.cell, d) == s.cell

    def test_default_melee_none_when_target_fully_walled(self) -> None:
        # Pikeman boxed in by obstacles on all 8 sides: no approach is
        # reachable, so default_melee returns None and no melee is legal.
        walls = frozenset(
            xy_cell(5 + dx, 4 + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))
        b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(0, 4),
                                    unit1=P, count1=1, cell1=(5, 4),
                                    obstacles=walls))
        while b.active_stack().unit_type != S:
            b.step(Action(ActionType.DEFEND))
        s = b.active_stack()
        target = b._stack_at(xy_cell(5, 4))
        assert b.default_melee(s, target) is None
        assert not any(is_melee(a.type) for a in b.legal_actions())


# --------------------------------------------------------------------- #
# Turn order


class TestSpeedTurnOrder:
    def test_faster_unit_acts_first_for_every_seed(self) -> None:
        # Swordsman (speed 5) strictly outpaces archer (speed 4): no seed's
        # tiebreak shuffle may ever put the archer first.
        for seed in range(20):
            b = battle_of(duel_scenario(unit0=A, count0=1, cell0=(0, 0),
                                        unit1=S, count1=1, cell1=(10, 8)),
                          seed)
            assert b.active_stack().unit_type == S, f"seed {seed}"


# --------------------------------------------------------------------- #
# Perks


class TestPerks:
    def battle(self) -> Battle:
        return det_battle(unit0=C, count0=1, cell0=(0, 0),
                          unit1=P, count1=10, cell1=(10, 8))

    def test_charge_doubles_melee_damage_when_moved_two_plus(self) -> None:
        b = self.battle()
        cav, pike = make_stack(0, 0, C, 1), make_stack(1, 1, P, 1)
        # Cavalry avg 12.5, atk 15 vs def 5 -> x1.5 -> 18; charged x2 -> 37
        assert b.compute_damage(cav, pike, melee=True, moved=0) == 18
        assert b.compute_damage(cav, pike, melee=True, moved=2) == 37
        assert b.compute_damage(cav, pike, melee=True, moved=7) == 37

    def test_no_charge_under_two_cells(self) -> None:
        b = self.battle()
        cav, pike = make_stack(0, 0, C, 1), make_stack(1, 1, P, 1)
        assert b.compute_damage(cav, pike, melee=True, moved=1) == 18

    def test_charge_requires_the_perk(self) -> None:
        b = self.battle()
        sword, pike = make_stack(0, 0, S, 1), make_stack(1, 1, P, 1)
        # Swordsman has no CHARGE: distance moved is irrelevant.
        assert (b.compute_damage(sword, pike, melee=True, moved=5)
                == b.compute_damage(sword, pike, melee=True, moved=0) == 9)

    def test_melee_penalty_is_perk_gated(self,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        b = self.battle()
        # Strip the archer's MELEE_PENALTY perk: melee damage is no longer
        # halved even though the unit is still ranged.
        monkeypatch.setitem(STATS, A,
                            dataclasses.replace(STATS[A], perks=frozenset()))
        archer, pike = make_stack(0, 0, A, 4), make_stack(1, 1, P, 1)
        assert b.compute_damage(archer, pike, melee=True) == 10

    def test_charge_applies_through_melee_attack_action(self) -> None:
        # Cavalry walks 4 cells into the strike: 37 charged damage, not 18.
        b = det_battle(unit0=C, count0=1, cell0=(0, 4),
                       unit1=P, count1=10, cell1=(5, 4))
        assert b.active_stack().unit_type == C  # speed 7 vs 4
        pike = b.stacks[1]
        b.step(default_melee_action(b, pike.cell))
        assert pike.total_hp == 100 - 37

    def test_adjacent_attack_does_not_charge(self) -> None:
        b = det_battle(unit0=C, count0=1, cell0=(4, 4),
                       unit1=P, count1=10, cell1=(5, 4))
        assert b.active_stack().unit_type == C
        pike = b.stacks[1]
        b.step(Action(ActionType.MELEE_W, pike.cell))
        assert pike.total_hp == 100 - 18

    def test_retaliation_never_charges(self) -> None:
        # Cavalry mirror: the attacker charges in (25 = 12 x2), but the
        # defender's retaliation is struck standing still (12).
        b = det_battle(unit0=C, count0=1, cell0=(0, 4),
                       unit1=C, count1=1, cell1=(4, 4))
        attacker = b.active_stack()
        defender = next(s for s in b.stacks.values() if s is not attacker)
        b.step(default_melee_action(b, defender.cell))
        assert defender.total_hp == 50 - 25
        assert attacker.total_hp == 50 - 12


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
    Swordsman mirror keeps speed ties resolved by the seeded shuffle."""
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
                b.step(default_melee_action(b, defender.cell))
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
                b.step(default_melee_action(b, defender.cell))
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
        # Swordsman (speed 5) acts before archer (4); skip to the archer turn.
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
        # Cavalry (speed 7) acts before pikeman (speed 4).
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

    def test_wait_phase_runs_in_reverse_speed(self) -> None:
        b = self.setup_battle()
        cav, pike = b.stacks[0], b.stacks[1]
        b.step(Action(ActionType.WAIT))  # cavalry waits
        b.step(Action(ActionType.WAIT))  # pikeman waits
        # Reverse speed: pikeman (4) before cavalry (7).
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
        assert any(is_melee(t) for t in types)
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
    # Swordsman (speed 5) acts before pikeman (speed 4) every round.
    b = battle_of(duel_scenario(unit0=S, count0=1, cell0=(5, 4),
                                unit1=P, count1=1, cell1=(6, 4)))
    sword, pike = b.stacks[0], b.stacks[1]
    b.step(Action(ActionType.DEFEND))  # swordsman
    b.step(Action(ActionType.DEFEND))  # pikeman
    assert pike.defending
    # Round 2: swordsman strikes the still-defending pikeman: 8 not 9 damage.
    assert b.active_stack() is sword
    b.step(default_melee_action(b, pike.cell))
    assert pike.top_hp == 10 - 8
    # Pikeman's own turn clears its defend flag.
    assert b.active_stack() is pike
    b.step(default_melee_action(b, sword.cell))
    assert not pike.defending


# --------------------------------------------------------------------- #
# Terminal conditions


def test_battle_ends_when_side_wiped() -> None:
    b = battle_of(duel_scenario(unit0=C, count0=10, cell0=(5, 4),
                                unit1=P, count1=1, cell1=(6, 4)))
    while b.active_stack().unit_type != C:
        b.step(Action(ActionType.DEFEND))
    b.step(default_melee_action(b, xy_cell(6, 4)))  # 125+ dmg vs 10 hp
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
