"""Deterministic tests for the M0 combat rules."""

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from combat.characters import Fighter, Goblin
from combat.mechanics import (
    resolve_attack,
    roll_damage,
    use_action_surge,
    use_fighter_attack,
    use_second_wind,
)


def fixed_rng(*rolls: int) -> Generator:
    """Supply exact Generator.integers results with exclusive upper bounds."""
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


def make_fighter(**overrides: int | bool) -> Fighter:
    values: dict[str, str | int | bool] = {
        "name": "Fighter",
        "max_hp": 20,
        "hp": 20,
        "armor_class": 16,
        "attack_bonus": 5,
        "damage_dice_count": 1,
        "damage_die_size": 8,
        "damage_bonus": 3,
    }
    values.update(overrides)
    return Fighter(**values)


def make_goblin(**overrides: int) -> Goblin:
    values: dict[str, str | int] = {
        "name": "Goblin",
        "max_hp": 15,
        "hp": 15,
        "armor_class": 15,
        "attack_bonus": 4,
        "damage_dice_count": 1,
        "damage_die_size": 6,
        "damage_bonus": 2,
    }
    values.update(overrides)
    return Goblin(**values)


def test_natural_one_misses_despite_large_attack_bonus() -> None:
    fighter = make_fighter(attack_bonus=100)
    goblin = make_goblin()

    result = resolve_attack(fighter, goblin, fixed_rng(1))

    assert result.d20_roll == 1
    assert result.total_attack == 101
    assert not result.hit
    assert not result.critical
    assert result.damage == 0
    assert goblin.hp == 15


def test_natural_twenty_hits_and_crits_despite_high_ac() -> None:
    fighter = make_fighter()
    goblin = make_goblin(armor_class=100)

    result = resolve_attack(fighter, goblin, fixed_rng(20, 2, 4))

    assert result.d20_roll == 20
    assert result.total_attack == 25
    assert result.hit
    assert result.critical
    assert result.damage == 9  # 2d8 + 3
    assert goblin.hp == 6


@pytest.mark.parametrize(
    ("d20", "expected_hit", "expected_hp"),
    [(9, False, 15), (10, True, 8)],
)
def test_ordinary_attack_uses_total_against_ac(
    d20: int, expected_hit: bool, expected_hp: int
) -> None:
    fighter = make_fighter()
    goblin = make_goblin()
    rng = fixed_rng(d20, 4) if expected_hit else fixed_rng(d20)

    result = resolve_attack(fighter, goblin, rng)

    assert result.total_attack == d20 + fighter.attack_bonus
    assert result.hit is expected_hit
    assert not result.critical
    assert goblin.hp == expected_hp


def test_critical_doubles_dice_but_not_damage_bonus() -> None:
    fighter = make_fighter(damage_dice_count=2, damage_die_size=6)

    assert roll_damage(fighter, fixed_rng(2, 4)) == 9  # 2d6 + 3
    assert roll_damage(fighter, fixed_rng(2, 4, 5, 1), critical=True) == 15  # 4d6 + 3


def test_successful_attack_reduces_hp_without_going_below_zero() -> None:
    fighter = make_fighter()
    goblin = make_goblin(hp=4)

    result = resolve_attack(fighter, goblin, fixed_rng(10, 8))

    assert result.hit
    assert result.damage == 4
    assert goblin.hp == 0
    assert not goblin.alive


def test_dead_character_is_not_alive() -> None:
    fighter = make_fighter(hp=0)
    assert not fighter.alive


def test_fighter_attack_consumes_action() -> None:
    fighter = make_fighter()
    goblin = make_goblin()

    use_fighter_attack(fighter, goblin, fixed_rng(10, 3))

    assert not fighter.action_available
    assert goblin.hp == 9


def test_fighter_cannot_attack_without_action() -> None:
    fighter = make_fighter(action_available=False)
    goblin = make_goblin()

    with pytest.raises(ValueError, match="ATTACK is not legal"):
        use_fighter_attack(fighter, goblin, fixed_rng())

    assert goblin.hp == 15


def test_fighter_cannot_attack_dead_goblin() -> None:
    fighter = make_fighter()
    goblin = make_goblin(hp=0)

    with pytest.raises(ValueError, match="ATTACK is not legal"):
        use_fighter_attack(fighter, goblin, fixed_rng())

    assert fighter.action_available


def test_second_wind_consumes_bonus_action_and_resource() -> None:
    fighter = make_fighter(hp=10)

    restored = use_second_wind(fighter, fixed_rng(4))

    assert restored == 6  # 1d10 + 2
    assert fighter.hp == 16
    assert not fighter.bonus_action_available
    assert not fighter.second_wind_available
    with pytest.raises(ValueError, match="SECOND_WIND is not legal"):
        use_second_wind(fighter, fixed_rng())


def test_second_wind_cannot_be_used_at_full_hp() -> None:
    fighter = make_fighter()

    with pytest.raises(ValueError, match="SECOND_WIND is not legal"):
        use_second_wind(fighter, fixed_rng())

    assert fighter.bonus_action_available
    assert fighter.second_wind_available


def test_second_wind_cannot_exceed_max_hp() -> None:
    fighter = make_fighter(hp=19)

    assert use_second_wind(fighter, fixed_rng(10)) == 1
    assert fighter.hp == fighter.max_hp


def test_second_wind_cannot_be_used_without_bonus_action() -> None:
    fighter = make_fighter(hp=10, bonus_action_available=False)

    with pytest.raises(ValueError, match="SECOND_WIND is not legal"):
        use_second_wind(fighter, fixed_rng())

    assert fighter.hp == 10
    assert fighter.second_wind_available


def test_action_surge_restores_action_and_is_once_per_encounter() -> None:
    fighter = make_fighter(action_available=False)

    use_action_surge(fighter)

    assert fighter.action_available
    assert not fighter.action_surge_available
    with pytest.raises(ValueError, match="ACTION_SURGE is not legal"):
        use_action_surge(fighter)
    assert fighter.action_available


def test_numpy_integer_rolls_return_python_ints() -> None:
    fighter = make_fighter()
    goblin = make_goblin()

    result = resolve_attack(fighter, goblin, fixed_rng(10, 4))

    assert all(type(value) is int for value in (result.d20_roll, result.total_attack, result.damage))
    assert type(goblin.hp) is int
    assert type(roll_damage(fighter, fixed_rng(4))) is int

    fighter.hp = 10
    assert type(use_second_wind(fighter, fixed_rng(4))) is int
    assert type(fighter.hp) is int


def test_seeded_generators_reproduce_attack_outcomes() -> None:
    first_fighter, first_goblin = make_fighter(), make_goblin()
    second_fighter, second_goblin = make_fighter(), make_goblin()

    first_result = resolve_attack(first_fighter, first_goblin, np.random.default_rng(0))
    second_result = resolve_attack(second_fighter, second_goblin, np.random.default_rng(0))

    assert first_result.hit
    assert first_result == second_result
    assert first_goblin.hp == second_goblin.hp
