"""Stable combatant references and roster organization."""

from dataclasses import dataclass
from enum import Enum

from .characters import Character


class Side(Enum):
    ALLY = "ally"
    ENEMY = "enemy"


@dataclass(frozen=True)
class ActorRef:
    """Immutable semantic address such as ALLY/0 or ENEMY/1.

    Holds no Character pointer; resolve through CombatRoster.get.
    """

    side: Side
    slot: int

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("Actor slot cannot be negative")


@dataclass(frozen=True)
class CombatRoster:
    """Shallow container for stable ally/enemy tuples.

    State organization only; combat rules live in mechanics.py.
    """

    allies: tuple[Character, ...]
    enemies: tuple[Character, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allies", tuple(self.allies))
        object.__setattr__(self, "enemies", tuple(self.enemies))

    def get(self, ref: ActorRef) -> Character:
        if ref.side is Side.ALLY:
            return self.allies[ref.slot]
        return self.enemies[ref.slot]

    def is_alive(self, ref: ActorRef) -> bool:
        return self.get(ref).alive

    def living_allies(self) -> tuple[Character, ...]:
        return tuple(ally for ally in self.allies if ally.alive)

    def living_enemies(self) -> tuple[Character, ...]:
        return tuple(enemy for enemy in self.enemies if enemy.alive)

    def side_defeated(self, side: Side) -> bool:
        members = self.allies if side is Side.ALLY else self.enemies
        return not any(member.alive for member in members)
