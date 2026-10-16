"""Character state for combat."""

from dataclasses import dataclass, field
from enum import Enum
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


class Role(Enum):
    """M5/M6 enemy role (simulator choice); values are the observation one-hot order."""

    BRUTE = 0
    ARCHER = 1
    HEALER = 2


class Targeting(Enum):
    """M5/M6 enemy target-selection policy; values are the observation one-hot order."""

    WEAKEST = 0
    RETALIATE = 1
    RANDOM = 2


class Rank(Enum):
    """M6 enemy rank; M5 places every enemy in the front rank."""

    FRONT = 0
    BACK = 1


@dataclass
class TacticalFighter(Fighter):
    """M5/M6 Fighter state with per-turn conditions and an M6 position.

    ``dodging`` lasts until the start of this ally's next turn; ``deep`` means
    it moved past the enemy front rank (M6); ``disengaged`` lasts until the end
    of its current turn (M6).
    """

    dodging: bool = False
    deep: bool = False
    disengaged: bool = False


@dataclass
class TacticalEnemy(Character):
    """M5/M6 enemy state: role, targeting policy, rank, and conditions.

    ``last_attacker`` is the ally slot that most recently made an attack or
    Trip roll against this enemy, or ``None``.
    """

    resources: ResourcePool = field(default_factory=lambda: ResourcePool({Resource.ACTION: 1}))
    known_abilities: frozenset[Action] = frozenset({Action.ATTACK})
    turn_refresh: Mapping[Resource, int] = field(
        default_factory=lambda: MappingProxyType({Resource.ACTION: 1})
    )
    role: Role = Role.BRUTE
    targeting: Targeting = Targeting.WEAKEST
    rank: Rank = Rank.FRONT
    prone: bool = False
    last_attacker: int | None = None
