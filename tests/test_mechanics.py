"""M0 state, ability, resource, and combat contracts."""

from dataclasses import FrozenInstanceError
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from combat.abilities import M0_ABILITIES, TargetType
from combat.actions import Action
from combat.characters import Fighter, Goblin
from combat.damage import DamageSpec
from combat.mechanics import (
    can_use_ability,
    resolve_attack,
    roll_damage,
    use_action_surge,
    use_attack,
    use_fighter_attack,
    use_second_wind,
)
from combat.resources import Resource, ResourcePool


def fixed_rng(*rolls: int) -> Generator:
    """Return exact NumPy scalar rolls, checking each die's exclusive upper bound."""
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


def fighter(*, hp: int = 20, attack_bonus: int = 5, damage: DamageSpec | None = None) -> Fighter:
    return Fighter("Fighter", 20, hp, 16, attack_bonus, damage or DamageSpec(1, 8, 3))


def goblin(*, hp: int = 15, armor_class: int = 15) -> Goblin:
    return Goblin("Goblin", 15, hp, armor_class, 4, DamageSpec(1, 6, 2))


@pytest.mark.parametrize(
    ("d20", "bonus", "ac", "damage_rolls", "hit", "critical", "damage", "total"),
    [
        (1, 100, 15, (), False, False, 0, 101),
        (20, 5, 100, (2, 4), True, True, 9, 25),
        (9, 5, 15, (), False, False, 0, 14),
        (10, 5, 15, (4,), True, False, 7, 15),
    ],
)
def test_attack_resolution_boundaries(
    d20: int,
    bonus: int,
    ac: int,
    damage_rolls: tuple[int, ...],
    hit: bool,
    critical: bool,
    damage: int,
    total: int,
) -> None:
    attacker, defender = fighter(attack_bonus=bonus), goblin(armor_class=ac)
    result = resolve_attack(attacker, defender, fixed_rng(d20, *damage_rolls))

    assert (result.d20_roll, result.total_attack) == (d20, total)
    assert (result.hit, result.critical, result.damage) == (hit, critical, damage)
    assert defender.hp == 15 - damage


def test_damage_and_alive_boundaries() -> None:
    attacker = fighter(damage=DamageSpec(2, 6, 3))
    assert roll_damage(attacker, fixed_rng(2, 4)) == 9  # 2d6 + 3
    assert roll_damage(attacker, fixed_rng(2, 4, 5, 1), critical=True) == 15  # 4d6 + 3

    defender = goblin(hp=4)
    result = resolve_attack(fighter(), defender, fixed_rng(10, 8))
    assert result.damage == 4
    assert defender.hp == 0
    assert not defender.alive
    assert fighter(hp=1).alive

    with pytest.raises(FrozenInstanceError):
        attacker.damage.bonus = 4


@pytest.mark.parametrize(
    ("action", "costs", "target"),
    [
        (Action.ATTACK, {Resource.ACTION: 1}, TargetType.ENEMY),
        (Action.SECOND_WIND, {Resource.BONUS_ACTION: 1, Resource.SECOND_WIND: 1}, TargetType.SELF),
        (Action.ACTION_SURGE, {Resource.ACTION_SURGE: 1}, TargetType.SELF),
    ],
)
def test_known_abilities_and_shared_costs(
    action: Action, costs: dict[Resource, int], target: TargetType
) -> None:
    actor = fighter(hp=10)
    spec = M0_ABILITIES[action]
    assert spec.action is action
    assert dict(spec.costs) == costs
    assert spec.target is target
    assert can_use_ability(actor, action)
    before = {resource: actor.resources.get(resource) for resource in costs}
    actor.resources.spend(spec.costs)
    assert all(actor.resources.get(resource) == before[resource] - amount for resource, amount in costs.items())
    assert not can_use_ability(actor, action)

    assert actor.known_abilities == frozenset({Action.ATTACK, Action.SECOND_WIND, Action.ACTION_SURGE})
    enemy = goblin()
    assert enemy.known_abilities == frozenset({Action.ATTACK})
    enemy.resources.set(Resource.BONUS_ACTION, 1)
    enemy.resources.set(Resource.SECOND_WIND, 1)
    assert not can_use_ability(enemy, Action.SECOND_WIND)
    assert Action.END_TURN not in M0_ABILITIES
    with pytest.raises(TypeError):
        spec.costs[Resource.ACTION] = 2


@pytest.mark.parametrize(
    ("action", "missing"),
    [
        (Action.ATTACK, Resource.ACTION),
        (Action.SECOND_WIND, Resource.BONUS_ACTION),
        (Action.SECOND_WIND, Resource.SECOND_WIND),
        (Action.ACTION_SURGE, Resource.ACTION_SURGE),
    ],
)
def test_missing_resources_reject_ability_without_mutation(action: Action, missing: Resource) -> None:
    actor = fighter(hp=10)
    actor.resources.set(missing, 0)
    before = {resource: actor.resources.get(resource) for resource in Resource}
    assert not can_use_ability(actor, action)

    with pytest.raises(ValueError, match=f"{action.name} is not legal"):
        if action is Action.ATTACK:
            use_fighter_attack(actor, goblin(), fixed_rng())
        elif action is Action.SECOND_WIND:
            use_second_wind(actor, fixed_rng())
        else:
            use_action_surge(actor)

    assert actor.hp == 10
    assert {resource: actor.resources.get(resource) for resource in Resource} == before


def test_attack_target_rule_and_action_consumption() -> None:
    actor = fighter()
    dead_target = goblin(hp=0)
    with pytest.raises(ValueError, match="ATTACK is not legal"):
        use_fighter_attack(actor, dead_target, fixed_rng())
    assert actor.resources.get(Resource.ACTION) == 1

    live_target = goblin()
    use_fighter_attack(actor, live_target, fixed_rng(10, 3))
    assert actor.resources.get(Resource.ACTION) == 0
    assert live_target.hp == 9
    with pytest.raises(ValueError, match="ATTACK is not legal"):
        use_fighter_attack(actor, live_target, fixed_rng())

    enemy = goblin()
    result = use_attack(enemy, fighter(), fixed_rng(12, 2))
    assert result.hit
    assert enemy.resources.get(Resource.ACTION) == 0


@pytest.mark.parametrize(("starting_hp", "d10", "restored", "ending_hp"), [(10, 4, 6, 16), (19, 10, 1, 20)])
def test_second_wind_unique_effect(starting_hp: int, d10: int, restored: int, ending_hp: int) -> None:
    actor = fighter(hp=starting_hp)
    assert use_second_wind(actor, fixed_rng(d10)) == restored
    assert actor.hp == ending_hp
    assert actor.resources.get(Resource.BONUS_ACTION) == 0
    assert actor.resources.get(Resource.SECOND_WIND) == 0
    with pytest.raises(ValueError, match="SECOND_WIND is not legal"):
        use_second_wind(actor, fixed_rng())

    full_hp_actor = fighter()
    assert can_use_ability(full_hp_actor, Action.SECOND_WIND)  # Costs alone are payable.
    with pytest.raises(ValueError, match="SECOND_WIND is not legal"):
        use_second_wind(full_hp_actor, fixed_rng())
    assert full_hp_actor.resources.get(Resource.SECOND_WIND) == 1


@pytest.mark.parametrize("starting_action", [0, 1])
def test_action_surge_adds_one_action(starting_action: int) -> None:
    actor = fighter()
    actor.resources.set(Resource.ACTION, starting_action)
    use_action_surge(actor)
    assert actor.resources.get(Resource.ACTION) == starting_action + 1
    assert actor.resources.get(Resource.ACTION_SURGE) == 0
    with pytest.raises(ValueError, match="ACTION_SURGE is not legal"):
        use_action_surge(actor)


def test_resource_pool_rejects_negative_and_spends_atomically() -> None:
    with pytest.raises(ValueError, match="negative"):
        ResourcePool({Resource.ACTION: -1})
    pool = ResourcePool({Resource.ACTION: 1, Resource.BONUS_ACTION: 1})
    assert pool.get(Resource.SECOND_WIND) == 0
    assert not pool.has({Resource.ACTION: 1, Resource.BONUS_ACTION: 2})
    with pytest.raises(ValueError, match="Insufficient"):
        pool.spend({Resource.ACTION: 1, Resource.BONUS_ACTION: 2})
    assert pool.get(Resource.ACTION) == pool.get(Resource.BONUS_ACTION) == 1
    with pytest.raises(ValueError, match="negative"):
        pool.set(Resource.ACTION, -1)
    with pytest.raises(ValueError, match="negative"):
        pool.has({Resource.ACTION: -1})


def test_resource_pool_gain_adds_without_allowing_negative_amounts() -> None:
    pool = ResourcePool({Resource.ACTION: 1})
    pool.gain(Resource.ACTION)
    assert pool.get(Resource.ACTION) == 2
    pool.gain(Resource.SECOND_WIND, 2)
    assert pool.get(Resource.SECOND_WIND) == 2
    pool.gain(Resource.ACTION, 0)
    assert pool.get(Resource.ACTION) == 2
    with pytest.raises(ValueError, match="negative"):
        pool.gain(Resource.ACTION, -1)
    assert pool.get(Resource.ACTION) == 2


def test_seeded_rng_and_python_int_boundaries() -> None:
    first_attacker, first_defender = fighter(), goblin()
    second_attacker, second_defender = fighter(), goblin()
    first = resolve_attack(first_attacker, first_defender, np.random.default_rng(0))
    second = resolve_attack(second_attacker, second_defender, np.random.default_rng(0))
    assert first.hit and first == second
    assert first_defender.hp == second_defender.hp
    assert all(type(value) is int for value in (first.d20_roll, first.total_attack, first.damage, first_defender.hp))
    assert type(roll_damage(fighter(), fixed_rng(4))) is int
    actor = fighter(hp=10)
    assert type(use_second_wind(actor, fixed_rng(4))) is int
    assert type(actor.hp) is int
