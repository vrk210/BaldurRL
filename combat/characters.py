"""Character state for combat."""

from dataclasses import dataclass, field

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

    @property
    def alive(self) -> bool:
        return self.hp > 0


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


@dataclass
class Goblin(Character):
    resources: ResourcePool = field(default_factory=lambda: ResourcePool({Resource.ACTION: 1}))
    known_abilities: frozenset[Action] = frozenset({Action.ATTACK})
