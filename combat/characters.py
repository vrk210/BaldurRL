"""Character state for combat."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .actions import Action
from .damage import DamageSpec
from .resources import Resource, ResourcePool


@dataclass
class Character:
    name: str
    max_hp: int
    hp: int
    armor_class: int
    attack_bonus: int
    damage: DamageSpec
    resources: ResourcePool = field(default_factory=ResourcePool)
    known_abilities: frozenset[Action] = frozenset()
    turn_refresh: Mapping[Resource, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def alive(self) -> bool:
        return self.hp > 0


def refresh_turn_resources(actor: Character) -> None:
    """Restore per-turn resources at the start of the actor's turn."""
    for resource, amount in actor.turn_refresh.items():
        actor.resources.set(resource, amount)


@dataclass
class Fighter(Character):
    resources: ResourcePool = field(default_factory=lambda: ResourcePool({
        Resource.ACTION: 1,
        Resource.BONUS_ACTION: 1,
        Resource.SECOND_WIND: 1,
        Resource.ACTION_SURGE: 1,
    }))
    known_abilities: frozenset[Action] = frozenset({
        Action.ATTACK,
        Action.SECOND_WIND,
        Action.ACTION_SURGE,
    })
    turn_refresh: Mapping[Resource, int] = field(
        default_factory=lambda: MappingProxyType({
            Resource.ACTION: 1,
            Resource.BONUS_ACTION: 1,
        })
    )


@dataclass
class Goblin(Character):
    resources: ResourcePool = field(default_factory=lambda: ResourcePool({Resource.ACTION: 1}))
    known_abilities: frozenset[Action] = frozenset({Action.ATTACK})
    turn_refresh: Mapping[Resource, int] = field(
        default_factory=lambda: MappingProxyType({Resource.ACTION: 1})
    )
