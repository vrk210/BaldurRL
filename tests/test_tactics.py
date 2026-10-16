"""Deterministic contracts for M5/M6 mechanics primitives and rules."""

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from combat.characters import Rank, Role, TacticalEnemy, TacticalFighter, Targeting
from combat.damage import DamageSpec
from combat.resources import Resource
from combat.rolls import (
    RollMode,
    combine_roll_mode,
    resolve_attack_with_mode,
    roll_attack,
    roll_d20_with_mode,
)
from combat.tactical_env import make_tactical_ally, make_tactical_enemy
from combat.tactics import (
    advance_available,
    archer_engaged,
    begin_tactical_turn,
    can_reach,
    choose_enemy_target,
    end_tactical_turn,
    enemy_attack_mode,
    front_broken,
    heal_character,
    heal_target,
    run_enemy_turn,
    use_advance,
    use_disengage,
    use_dodge,
    use_tactical_attack,
    use_tactical_cleave,
    use_trip,
)

HEAL = DamageSpec(2, 6, 2)


def fixed_rng(*rolls: int) -> Generator:
    """Exact rolls in order, checking each draw's exclusive upper bound."""
    values = iter(rolls)

    def integers(low: int, high: int) -> np.int64:
        try:
            result = next(values)
        except StopIteration as exc:
            raise AssertionError("Unexpected extra die roll") from exc
        assert low <= result < high, f"Roll {result} is outside [{low}, {high})"
        return np.int64(result)

    rng = Mock(spec=Generator)
    rng.integers.side_effect = integers
    return cast(Generator, rng)


def ally(stage: str = "m5", hp: int = 20) -> TacticalFighter:
    fighter = make_tactical_ally(stage)
    fighter.hp = hp
    return fighter


def enemy(
    role: Role = Role.BRUTE, *, stage: str = "m5", slot: int = 0, hp: int | None = None,
    targeting: Targeting = Targeting.WEAKEST, ac: int = 14, attack: int = 5, max_hp: int = 26,
) -> TacticalEnemy:
    built = make_tactical_enemy(stage, slot, role, targeting, max_hp, ac, attack, 3)
    if hp is not None:
        built.hp = hp
    return built


def m6_line() -> tuple[list[TacticalFighter], list[TacticalEnemy]]:
    """Brute in front (slot 0); Archer and Healer in the back rank."""
    allies = [ally("m6"), ally("m6")]
    enemies = [
        enemy(Role.BRUTE, stage="m6", slot=0, max_hp=24),
        enemy(Role.ARCHER, stage="m6", slot=1, max_hp=12, ac=13),
        enemy(Role.HEALER, stage="m6", slot=2, max_hp=14, ac=12),
    ]
    return allies, enemies


# --- advantage and disadvantage ---------------------------------------------------


@pytest.mark.parametrize(("advantage", "disadvantage", "mode"), [
    (False, False, RollMode.NORMAL),
    (True, False, RollMode.ADVANTAGE),
    (False, True, RollMode.DISADVANTAGE),
    (True, True, RollMode.NORMAL),
])
def test_roll_modes_cancel_and_never_stack(advantage, disadvantage, mode) -> None:
    assert combine_roll_mode(advantage, disadvantage) is mode


def test_roll_d20_with_mode_draws_and_keeps() -> None:
    assert roll_d20_with_mode(fixed_rng(7), RollMode.NORMAL) == (7, (7,))
    assert roll_d20_with_mode(fixed_rng(3, 17), RollMode.ADVANTAGE) == (17, (3, 17))
    assert roll_d20_with_mode(fixed_rng(3, 17), RollMode.DISADVANTAGE) == (3, (3, 17))


def test_natural_rolls_use_the_kept_die() -> None:
    crit = roll_attack(0, 30, fixed_rng(2, 20), RollMode.ADVANTAGE)
    assert crit.hit and crit.critical and crit.d20_roll == 20 and crit.d20_rolls == (2, 20)
    fumble = roll_attack(100, 5, fixed_rng(20, 1), RollMode.DISADVANTAGE)
    assert not fumble.hit and not fumble.critical and fumble.total_attack == 101
    exact = roll_attack(5, 15, fixed_rng(10))
    assert exact.hit and not exact.critical


def test_moded_attack_applies_damage_and_crit_doubles_dice() -> None:
    attacker, target = ally(), enemy(hp=26)
    result = resolve_attack_with_mode(attacker, target, fixed_rng(4, 20, 3, 6), RollMode.ADVANTAGE)
    assert (result.mode, result.d20_rolls, result.critical, result.damage) == ("advantage", (4, 20), True, 12)
    assert target.hp == 14
    miss = resolve_attack_with_mode(attacker, target, fixed_rng(19, 2), RollMode.DISADVANTAGE)
    assert not miss.hit and miss.damage == 0 and target.hp == 14
    target.hp = 3
    capped = resolve_attack_with_mode(attacker, target, fixed_rng(15, 8), RollMode.NORMAL)
    assert capped.damage == 3 and target.hp == 0


def test_heal_character_caps_and_rejects_dead() -> None:
    target = enemy(hp=20, max_hp=26)
    assert heal_character(target, 10) == 6 and target.hp == 26
    target.hp = 0
    with pytest.raises(ValueError, match="dead"):
        heal_character(target, 5)
    with pytest.raises(ValueError):
        heal_character(enemy(hp=5), -1)


# --- Trip, Prone, attacks, Cleave, Dodge ------------------------------------------


def test_trip_success_spends_bonus_and_charge_and_knocks_prone() -> None:
    attacker, target = ally(), enemy(ac=14)
    result = use_trip(1, attacker, target, [target], fixed_rng(9))  # 9 + 5 = 14 meets AC 14
    assert result.success and result.total == 14 and target.prone
    assert target.last_attacker == 1
    assert attacker.resources.get(Resource.BONUS_ACTION) == 0
    assert attacker.resources.get(Resource.TRIP) == 1
    with pytest.raises(ValueError, match="TRIP"):
        use_trip(1, attacker, target, [target], fixed_rng(20))  # already Prone


def test_failed_trip_still_records_attacker_and_spends() -> None:
    attacker, target = ally(), enemy(ac=30)
    result = use_trip(0, attacker, target, [target], fixed_rng(19))
    assert not result.success and not target.prone and target.last_attacker == 0
    assert attacker.resources.get(Resource.TRIP) == 1
    armored = enemy(ac=40)
    assert use_trip(0, ally(), armored, [armored], fixed_rng(20)).success and armored.prone  # natural 20
    lucky = enemy(ac=2)
    assert not use_trip(0, ally(), lucky, [lucky], fixed_rng(1)).success  # natural 1


def test_trip_needs_charges_and_bonus_action() -> None:
    attacker = ally()
    attacker.resources.set(Resource.TRIP, 0)
    with pytest.raises(ValueError, match="TRIP"):
        use_trip(0, attacker, enemy(), [enemy()], fixed_rng(10))
    attacker.resources.set(Resource.TRIP, 2)
    attacker.resources.set(Resource.BONUS_ACTION, 0)
    with pytest.raises(ValueError, match="TRIP"):
        use_trip(0, attacker, enemy(), [enemy()], fixed_rng(10))


def test_attacks_against_prone_targets_have_advantage() -> None:
    attacker, target = ally(), enemy(hp=26)
    target.prone = True
    result = use_tactical_attack(0, attacker, target, [target], fixed_rng(2, 12, 5))
    assert result.mode == "advantage" and result.d20_rolls == (2, 12) and result.damage == 8
    assert target.last_attacker == 0 and attacker.resources.get(Resource.ACTION) == 0


def test_prone_and_dodging_impose_disadvantage_on_enemy_attacks() -> None:
    brute, archer, fighter = enemy(Role.BRUTE), enemy(Role.ARCHER), ally()
    assert enemy_attack_mode(brute, fighter, engaged_archer=False) is RollMode.NORMAL
    brute.prone = True
    assert enemy_attack_mode(brute, fighter, engaged_archer=False) is RollMode.DISADVANTAGE
    fighter.dodging = True
    assert enemy_attack_mode(brute, fighter, engaged_archer=False) is RollMode.DISADVANTAGE
    brute.prone = fighter.dodging = False
    assert enemy_attack_mode(brute, fighter, engaged_archer=True) is RollMode.NORMAL
    assert enemy_attack_mode(archer, fighter, engaged_archer=True) is RollMode.DISADVANTAGE


def test_cleave_rolls_per_target_with_prone_advantage() -> None:
    attacker = ally()
    enemies = [enemy(Role.BRUTE, slot=0, hp=26), enemy(Role.ARCHER, slot=1, hp=0), enemy(Role.HEALER, slot=2, hp=14)]
    enemies[2].prone = True
    # Brute: normal hit d20 15, damage 8+3=11 -> 5; Healer: advantage (1, 18), damage 4+3=7 -> 3.
    results = use_tactical_cleave(1, attacker, enemies, fixed_rng(15, 8, 1, 18, 4))
    assert results[1] is None
    assert (results[0].mode, results[0].damage) == ("normal", 5)
    assert (results[2].mode, results[2].d20_rolls, results[2].damage) == ("advantage", (1, 18), 3)
    assert [e.last_attacker for e in enemies] == [1, None, 1]
    assert attacker.resources.get(Resource.CLEAVE) == 0


def test_dodge_spends_action_once() -> None:
    fighter = ally()
    use_dodge(fighter)
    assert fighter.dodging and fighter.resources.get(Resource.ACTION) == 0
    fighter.resources.set(Resource.ACTION, 1)
    with pytest.raises(ValueError, match="DODGE"):
        use_dodge(fighter)


def test_turn_lifecycle_clears_conditions() -> None:
    fighter, brute = ally("m6"), enemy(stage="m6")
    fighter.dodging = fighter.disengaged = True
    fighter.resources.set(Resource.MOVEMENT, 0)
    fighter.resources.set(Resource.TRIP, 1)
    begin_tactical_turn(fighter)
    assert not fighter.dodging and not fighter.disengaged
    assert fighter.resources.get(Resource.MOVEMENT) == 1 and fighter.resources.get(Resource.TRIP) == 1
    brute.prone = True
    brute.resources.set(Resource.REACTION, 0)
    begin_tactical_turn(brute)
    assert brute.prone and brute.resources.get(Resource.REACTION) == 1
    end_tactical_turn(brute)
    assert not brute.prone


# --- enemy behavior -----------------------------------------------------------------


def test_weakest_and_retaliate_targeting() -> None:
    allies = [ally(hp=12), ally(hp=9)]
    weakest = enemy(targeting=Targeting.WEAKEST)
    assert choose_enemy_target(weakest, allies, fixed_rng()) == 1
    allies[1].hp = 12
    assert choose_enemy_target(weakest, allies, fixed_rng()) == 0  # tie -> lowest slot
    vengeful = enemy(targeting=Targeting.RETALIATE)
    assert choose_enemy_target(vengeful, allies, fixed_rng()) == 0  # no attacker yet -> weakest
    vengeful.last_attacker = 1
    allies[1].hp = 20
    assert choose_enemy_target(vengeful, allies, fixed_rng()) == 1
    allies[1].hp = 0
    assert choose_enemy_target(vengeful, allies, fixed_rng()) == 0  # dead attacker -> weakest


def test_random_targeting_draws_only_with_two_living_allies() -> None:
    allies = [ally(), ally()]
    chaotic = enemy(targeting=Targeting.RANDOM)
    assert choose_enemy_target(chaotic, allies, fixed_rng(1)) == 1
    assert choose_enemy_target(chaotic, allies, fixed_rng(0)) == 0
    allies[0].hp = 0
    assert choose_enemy_target(chaotic, allies, fixed_rng()) == 1  # no draw


def test_heal_target_threshold_and_ties() -> None:
    enemies = [enemy(slot=0, max_hp=26, hp=14), enemy(Role.ARCHER, slot=1, max_hp=15, hp=8),
               enemy(Role.HEALER, slot=2, max_hp=14, hp=7)]
    assert heal_target(enemies) == 2  # 2*14 > 26; 2*8 > 15; 2*7 <= 14
    enemies[0].hp = 13  # 2*13 <= 26, missing 13 > 7
    assert heal_target(enemies) == 0
    enemies[1].max_hp, enemies[1].hp = 26, 13  # same missing HP as slot 0 -> lowest slot
    assert heal_target(enemies) == 0
    for target in enemies:
        target.hp = target.max_hp
    assert heal_target(enemies) is None


def test_healer_heals_then_attacks_when_out_of_charges() -> None:
    allies = [ally(), ally(hp=11)]
    enemies = [enemy(slot=0, max_hp=26, hp=10), enemy(Role.ARCHER, slot=1, max_hp=12),
               enemy(Role.HEALER, slot=2, max_hp=14)]
    healer = enemies[2]
    result = run_enemy_turn(2, allies, enemies, fixed_rng(3, 4), heal=HEAL, engaged_archer=False)
    assert (result.kind, result.target_slot, result.healed) == ("heal", 0, 9)
    assert enemies[0].hp == 19 and healer.resources.get(Resource.HEAL) == 2
    healer.resources.set(Resource.HEAL, 0)
    healer.resources.set(Resource.ACTION, 1)
    enemies[0].hp = 10
    result = run_enemy_turn(2, allies, enemies, fixed_rng(15, 4), heal=HEAL, engaged_archer=False)
    assert result.kind == "attack" and result.target_slot == 1 and allies[1].hp == 11 - 7


def test_non_healers_attack_and_dead_enemies_skip() -> None:
    allies = [ally(), ally()]
    enemies = [enemy(slot=0, max_hp=26, hp=5), enemy(Role.ARCHER, slot=1, max_hp=12, hp=0),
               enemy(Role.HEALER, slot=2, max_hp=14)]
    result = run_enemy_turn(0, allies, enemies, fixed_rng(1), heal=HEAL, engaged_archer=False)
    assert result.kind == "attack" and not result.attack.hit
    assert run_enemy_turn(1, allies, enemies, fixed_rng(), heal=HEAL, engaged_archer=False) is None


# --- M6 ranks, reach, Advance, Disengage ------------------------------------------


def test_reach_front_broken_and_archer_engagement() -> None:
    allies, enemies = m6_line()
    brute, archer, healer = enemies
    assert can_reach(allies[0], brute, enemies) and not can_reach(allies[0], archer, enemies)
    assert not archer_engaged(allies, enemies) and not front_broken(enemies)
    allies[1].deep = True
    assert can_reach(allies[1], healer, enemies) and can_reach(allies[1], brute, enemies)
    assert archer_engaged(allies, enemies)
    allies[1].deep = False
    brute.hp = 0
    assert front_broken(enemies) and can_reach(allies[0], archer, enemies)
    assert archer_engaged(allies, enemies)
    assert not advance_available(allies[0], enemies)


def test_m5_enemies_are_all_front_rank() -> None:
    enemies = [enemy(role, slot=slot) for slot, role in enumerate(Role)]
    assert all(e.rank is Rank.FRONT for e in enemies)
    assert all(can_reach(ally(), e, enemies) for e in enemies)


def test_advance_provokes_one_opportunity_attack_and_moves_deep() -> None:
    allies, enemies = m6_line()
    attacks = use_advance(allies[0], allies, enemies, fixed_rng(15, 6))
    assert len(attacks) == 1 and attacks[0].enemy_slot == 0
    assert attacks[0].attack.damage == 9 and allies[0].hp == 11 and allies[0].deep
    assert enemies[0].resources.get(Resource.REACTION) == 0
    assert allies[0].resources.get(Resource.MOVEMENT) == 0
    # The Brute's Reaction is spent, so the second ally advances freely.
    assert use_advance(allies[1], allies, enemies, fixed_rng()) == ()
    assert allies[1].deep


def test_opportunity_attacks_use_roll_modes_and_can_kill() -> None:
    allies, enemies = m6_line()
    enemies[0].prone = True
    allies[0].hp = 4
    attacks = use_advance(allies[0], allies, enemies, fixed_rng(3, 18, 4))
    assert attacks[0].attack.mode == "disadvantage" and attacks[0].attack.d20_rolls == (3, 18)
    allies[1].hp = 5
    enemies[0].prone = False
    enemies[0].resources.set(Resource.REACTION, 1)
    attacks = use_advance(allies[1], allies, enemies, fixed_rng(12, 5))
    assert attacks[0].attack.damage == 5 and not allies[1].alive and not allies[1].deep


def test_disengage_prevents_opportunity_attacks() -> None:
    allies, enemies = m6_line()
    use_disengage(allies[0], enemies)
    assert allies[0].disengaged and allies[0].resources.get(Resource.ACTION) == 0
    assert use_advance(allies[0], allies, enemies, fixed_rng()) == ()
    assert allies[0].deep and enemies[0].resources.get(Resource.REACTION) == 1
    with pytest.raises(ValueError, match="DISENGAGE"):
        use_disengage(allies[0], enemies)  # already deep
    with pytest.raises(ValueError, match="ADVANCE"):
        use_advance(allies[0], allies, enemies, fixed_rng())


def test_cleave_reach_is_fixed_before_the_first_roll() -> None:
    allies, enemies = m6_line()
    enemies[0].hp = 1  # the Brute dies to the first Cleave roll
    results = use_tactical_cleave(0, allies[0], enemies, fixed_rng(15, 2))
    assert results[0].damage == 1 and front_broken(enemies)
    assert results[1] is None and results[2] is None


def test_back_rank_attacks_and_trips_need_reach() -> None:
    allies, enemies = m6_line()
    with pytest.raises(ValueError, match="ATTACK"):
        use_tactical_attack(0, allies[0], enemies[1], enemies, fixed_rng(15, 4))
    with pytest.raises(ValueError, match="TRIP"):
        use_trip(0, allies[0], enemies[2], enemies, fixed_rng(15))
