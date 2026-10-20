"""The resources each known ability spends, as immutable tables."""

from types import MappingProxyType
from typing import Mapping

from .actions import Action
from .resources import Resource

AbilityCosts = Mapping[Action, Mapping[Resource, int]]


def _frozen(costs: AbilityCosts) -> AbilityCosts:
    return MappingProxyType({action: MappingProxyType(dict(cost)) for action, cost in costs.items()})


# M0–M4: the Fighter's abilities, Cleave for M2 onward, and the Goblin's attack.
ABILITY_COSTS = _frozen({
    Action.ATTACK: {Resource.ACTION: 1},
    Action.SECOND_WIND: {Resource.BONUS_ACTION: 1, Resource.SECOND_WIND: 1},
    Action.ACTION_SURGE: {Resource.ACTION_SURGE: 1},
    Action.CLEAVE: {Resource.ACTION: 1, Resource.CLEAVE: 1},
})

# M5/M6 simulator choices (combat/tactical_env.py and combat/tactics.py): Trip spends the Bonus Action
# (shared with Second Wind) and one per-encounter Trip charge; Dodge and
# Disengage spend the Action; Advance spends the per-turn Movement; an enemy
# Healer's heal spends its Action and one heal charge.
TACTICAL_ABILITY_COSTS = _frozen({
    **ABILITY_COSTS,
    Action.TRIP: {Resource.BONUS_ACTION: 1, Resource.TRIP: 1},
    Action.DODGE: {Resource.ACTION: 1},
    Action.HEAL: {Resource.ACTION: 1, Resource.HEAL: 1},
    Action.ADVANCE: {Resource.MOVEMENT: 1},
    Action.DISENGAGE: {Resource.ACTION: 1},
})
