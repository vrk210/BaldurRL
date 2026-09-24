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
