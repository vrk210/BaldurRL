"""Wire contract for semantic game snapshots and events.

These types describe observed facts, not simulator state or executable rules.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("World coordinates must be finite JSON numbers")


@dataclass(frozen=True)
class ResourceValue:
    resource_id: str
    amount: float
    level: int | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.amount):
            raise ValueError("Resource amount must be a finite JSON number")


@dataclass(frozen=True)
class AbilitySnapshot:
    ability_id: str
    known: bool = True
    visible: bool | None = None


@dataclass(frozen=True)
class EntitySnapshot:
    entity_id: str
    name: str | None = None
    hp: int | None = None
    max_hp: int | None = None
    armor_class: int | None = None
    position: Vec3 | None = None
    alive: bool | None = None
    resources: tuple[ResourceValue, ...] = ()
    abilities: tuple[AbilitySnapshot, ...] = ()


@dataclass(frozen=True)
class CombatSnapshot:
    combat_id: str | None = None
    round_number: int | None = None
    active_entity_id: str | None = None
    participant_ids: tuple[str, ...] = ()


class Legality(Enum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    UNKNOWN = "unknown"


class TargetKind(Enum):
    NONE = "none"
    SELF = "self"
    ENTITY = "entity"
    POSITION = "position"


@dataclass(frozen=True)
class DecisionCandidate:
    ability_id: str
    target_kind: TargetKind
    legality: Legality
    target_entity_id: str | None = None
    target_position: Vec3 | None = None
    legality_source: str | None = None
    legality_reason: str | None = None


def _position_dict(position: Vec3 | None) -> dict[str, float] | None:
    return None if position is None else {"x": position.x, "y": position.y, "z": position.z}


@dataclass(frozen=True)
class GameSnapshot:
    schema_version: int
    sequence: int
    timestamp_ms: int | None = None
    controlled_entity_id: str | None = None
    combat: CombatSnapshot | None = None
    entities: tuple[EntitySnapshot, ...] = ()
    candidate_decisions: tuple[DecisionCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible primitives and arrays in stable field order."""
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "controlled_entity_id": self.controlled_entity_id,
            "combat": None if self.combat is None else {
                "combat_id": self.combat.combat_id,
                "round_number": self.combat.round_number,
                "active_entity_id": self.combat.active_entity_id,
                "participant_ids": list(self.combat.participant_ids),
            },
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "hp": entity.hp,
                    "max_hp": entity.max_hp,
                    "armor_class": entity.armor_class,
                    "position": _position_dict(entity.position),
                    "alive": entity.alive,
                    "resources": [
                        {"resource_id": resource.resource_id, "amount": resource.amount, "level": resource.level}
                        for resource in entity.resources
                    ],
                    "abilities": [
                        {"ability_id": ability.ability_id, "known": ability.known, "visible": ability.visible}
                        for ability in entity.abilities
                    ],
                }
                for entity in self.entities
            ],
            "candidate_decisions": [
                {
                    "ability_id": candidate.ability_id,
                    "target_kind": candidate.target_kind.value,
                    "target_entity_id": candidate.target_entity_id,
                    "target_position": _position_dict(candidate.target_position),
                    "legality": candidate.legality.value,
                    "legality_source": candidate.legality_source,
                    "legality_reason": candidate.legality_reason,
                }
                for candidate in self.candidate_decisions
            ],
        }


class EventKind(Enum):
    COMBAT_STARTED = "combat_started"
    COMBAT_ENDED = "combat_ended"
    ENTERED_COMBAT = "entered_combat"
    ROUND_STARTED = "round_started"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    ABILITY_USED = "ability_used"
    ABILITY_USED_ON_TARGET = "ability_used_on_target"
    ABILITY_USED_AT_POSITION = "ability_used_at_position"
    DAMAGE = "damage"
    MISS = "miss"
    CRITICAL_HIT = "critical_hit"
    DIED = "died"


@dataclass(frozen=True)
class GameEvent:
    schema_version: int
    sequence: int
    kind: EventKind
    timestamp_ms: int | None = None
    combat_id: str | None = None
    round_number: int | None = None
    actor_id: str | None = None
    target_id: str | None = None
    ability_id: str | None = None
    target_position: Vec3 | None = None
    story_action_id: int | None = None
    damage: int | None = None
    damage_type: str | None = None
    critical: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible primitives in stable field order."""
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "kind": self.kind.value,
            "combat_id": self.combat_id,
            "round_number": self.round_number,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "ability_id": self.ability_id,
            "target_position": _position_dict(self.target_position),
            "story_action_id": self.story_action_id,
            "damage": self.damage,
            "damage_type": self.damage_type,
            "critical": self.critical,
        }
