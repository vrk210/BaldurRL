"""Per-actor targeting and legality without turns, dice, or episodes."""

import pytest

from combat.actions import Action
from combat.actors import ActorRef, CombatRoster, Side
from combat.characters import Fighter, Goblin
from combat.damage import DamageSpec
from combat.legality import DecisionSpec, legal_mask_for, resolve_target
from combat.resources import Resource


def _fighter(hp: int = 20) -> Fighter:
    return Fighter("Fighter", 20, hp, 16, 5, DamageSpec(1, 8, 3))


def _goblin(hp: int = 15) -> Goblin:
    return Goblin("Goblin", 15, hp, 15, 4, DamageSpec(1, 6, 2))


def _m1b_decisions() -> tuple[DecisionSpec, ...]:
    return (
        DecisionSpec(Action.ATTACK, 0),
        DecisionSpec(Action.ATTACK, 1),
        DecisionSpec(Action.SECOND_WIND),
        DecisionSpec(Action.ACTION_SURGE),
        DecisionSpec(Action.END_TURN),
    )


def test_resolve_target_indexes_the_opposing_side() -> None:
    roster = CombatRoster(allies=(_fighter(),), enemies=(_goblin(), _goblin()))
    assert resolve_target(ActorRef(Side.ALLY, 0), 0, roster) == ActorRef(Side.ENEMY, 0)
    assert resolve_target(ActorRef(Side.ALLY, 0), 1, roster) == ActorRef(Side.ENEMY, 1)
    assert resolve_target(ActorRef(Side.ALLY, 0), None, roster) == ActorRef(Side.ENEMY, 0)
    assert resolve_target(ActorRef(Side.ENEMY, 0), 0, roster) == ActorRef(Side.ALLY, 0)
    assert resolve_target(ActorRef(Side.ENEMY, 1), None, roster) == ActorRef(Side.ALLY, 0)


def test_resolve_target_rejects_out_of_range_slots() -> None:
    roster = CombatRoster(allies=(_fighter(),), enemies=(_goblin(),))
    with pytest.raises(IndexError, match="out of range"):
        resolve_target(ActorRef(Side.ALLY, 0), 1, roster)
    with pytest.raises(IndexError, match="out of range"):
        resolve_target(ActorRef(Side.ALLY, 0), -1, roster)


def test_fighter_mask_matches_current_m1b_expectations() -> None:
    roster = CombatRoster(allies=(_fighter(),), enemies=(_goblin(), _goblin()))
    decisions = _m1b_decisions()
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions) == [
        True, True, False, True, True,  # Second Wind needs missing HP.
    ]
    roster.allies[0].hp = 10
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions)[2] is True
    roster.allies[0].resources.set(Resource.ACTION, 0)
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions)[:2] == [False, False]
    roster.enemies[0].hp = 0
    roster.allies[0].resources.set(Resource.ACTION, 1)
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions)[:2] == [False, True]


def test_goblin_mask_comes_from_its_own_class_kit() -> None:
    roster = CombatRoster(allies=(_fighter(),), enemies=(_goblin(), _goblin()))
    decisions = _m1b_decisions()
    assert legal_mask_for(ActorRef(Side.ENEMY, 0), roster, decisions) == [
        # Goblin knows ATTACK only; ATTACK_1 names a second ally that
        # does not exist, so it is illegal rather than an error.
        True, False, False, False, True,
    ]
    roster.enemies[0].resources.set(Resource.ACTION, 0)
    assert legal_mask_for(ActorRef(Side.ENEMY, 0), roster, decisions) == [
        False, False, False, False, True,
    ]


def test_second_wind_uses_the_acting_character_hp() -> None:
    roster = CombatRoster(allies=(_fighter(), _fighter(hp=10)), enemies=(_goblin(),))
    decisions = _m1b_decisions()
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions)[2] is False
    assert legal_mask_for(ActorRef(Side.ALLY, 1), roster, decisions)[2] is True


def test_dead_actor_and_defeated_side_yield_all_false() -> None:
    decisions = _m1b_decisions()
    dead_actor = CombatRoster(allies=(_fighter(hp=0),), enemies=(_goblin(),))
    assert legal_mask_for(ActorRef(Side.ALLY, 0), dead_actor, decisions) == [False] * 5
    no_opponents = CombatRoster(allies=(_fighter(),), enemies=(_goblin(hp=0),))
    assert legal_mask_for(ActorRef(Side.ALLY, 0), no_opponents, decisions) == [False] * 5


def test_cleave_needs_known_ability_and_a_living_opponent() -> None:
    decisions = _m1b_decisions()[:2] + (DecisionSpec(Action.CLEAVE),) + _m1b_decisions()[2:]
    fighter = _fighter()
    fighter.known_abilities = fighter.known_abilities | {Action.CLEAVE}
    fighter.resources.set(Resource.CLEAVE, 1)
    roster = CombatRoster(allies=(fighter,), enemies=(_goblin(), _goblin(hp=0)))
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions)[2] is True
    roster.enemies[0].hp = 0
    assert legal_mask_for(ActorRef(Side.ALLY, 0), roster, decisions) == [False] * 6
    fresh = CombatRoster(allies=(_fighter(),), enemies=(_goblin(), _goblin()))
    assert legal_mask_for(ActorRef(Side.ALLY, 0), fresh, decisions)[2] is False
