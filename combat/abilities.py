"""Immutable definitions for M0 character abilities."""

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping

from .actions import Action
from .resources import Resource


class TargetType(Enum):
    SELF = auto()
    ENEMY = auto()
    FRIENDLY = auto()


@dataclass(frozen=True)
class AbilitySpec:
    action: Action
    costs: Mapping[Resource, int]
    target: TargetType

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", MappingProxyType(dict(self.costs)))


M0_ABILITIES: Mapping[Action, AbilitySpec] = MappingProxyType({
    Action.ATTACK: AbilitySpec(Action.ATTACK, {Resource.ACTION: 1}, TargetType.ENEMY),
    Action.SECOND_WIND: AbilitySpec(
        Action.SECOND_WIND,
        {Resource.BONUS_ACTION: 1, Resource.SECOND_WIND: 1},
        TargetType.SELF,
    ),
    Action.ACTION_SURGE: AbilitySpec(Action.ACTION_SURGE, {Resource.ACTION_SURGE: 1}, TargetType.SELF),
})

M2_ABILITIES: Mapping[Action, AbilitySpec] = MappingProxyType({
    **M0_ABILITIES,
    Action.CLEAVE: AbilitySpec(
        Action.CLEAVE, {Resource.ACTION: 1, Resource.CLEAVE: 1}, TargetType.ENEMY
    ),
})

# M5/M6 simulator choices (docs/STAGES_SPEC.md): Trip spends the Bonus Action
# (shared with Second Wind) and one per-encounter Trip charge; Dodge and
# Disengage spend the Action; Advance spends the per-turn Movement; an enemy
# Healer's heal spends its Action and one heal charge.
TACTICAL_ABILITIES: Mapping[Action, AbilitySpec] = MappingProxyType({
    **M2_ABILITIES,
    Action.TRIP: AbilitySpec(
        Action.TRIP, {Resource.BONUS_ACTION: 1, Resource.TRIP: 1}, TargetType.ENEMY
    ),
    Action.DODGE: AbilitySpec(Action.DODGE, {Resource.ACTION: 1}, TargetType.SELF),
    Action.HEAL: AbilitySpec(
        Action.HEAL, {Resource.ACTION: 1, Resource.HEAL: 1}, TargetType.FRIENDLY
    ),
    Action.ADVANCE: AbilitySpec(Action.ADVANCE, {Resource.MOVEMENT: 1}, TargetType.SELF),
    Action.DISENGAGE: AbilitySpec(Action.DISENGAGE, {Resource.ACTION: 1}, TargetType.SELF),
})
