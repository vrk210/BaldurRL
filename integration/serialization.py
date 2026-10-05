"""Explicit version-one JSON parsing for passive BG3 observations."""

from math import isfinite
from typing import Any, Mapping

from .schema import (
    SCHEMA_VERSION, AbilitySnapshot, CombatSnapshot, DecisionCandidate,
    EntitySnapshot, EventKind, GameEvent, GameSnapshot, Legality,
    ResourceValue, TargetKind, Vec3,
)


Record = GameSnapshot | GameEvent


def record_to_dict(record: Record) -> dict[str, Any]:
    if not isinstance(record, (GameSnapshot, GameEvent)):
        raise TypeError("Expected GameSnapshot or GameEvent")
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record.schema_version!r}")
    return record.to_dict()


def _object(value: Any, path: str, allowed: set[str], required: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise ValueError(f"{path} missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
    return value


def _typed(value: Any, kind: type, path: str, *, optional: bool = False) -> Any:
    if value is None and optional:
        return None
    if type(value) is not kind:
        raise ValueError(f"{path} must be {'null or ' if optional else ''}{kind.__name__}")
    return value


def _number(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{path} must be a finite JSON number")
    return float(value)


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array")
    return value


def _enum(value: Any, kind: type, path: str) -> Any:
    _typed(value, str, path)
    try:
        return kind(value)
    except ValueError as exc:
        raise ValueError(f"{path} has unknown value {value!r}") from exc


def _vec3(value: Any, path: str) -> Vec3 | None:
    if value is None:
        return None
    obj = _object(value, path, {"x", "y", "z"}, {"x", "y", "z"})
    return Vec3(*(_number(obj[key], f"{path}.{key}") for key in ("x", "y", "z")))


def _resource(value: Any, path: str) -> ResourceValue:
    obj = _object(value, path, {"resource_id", "amount", "level"}, {"resource_id", "amount"})
    return ResourceValue(
        _typed(obj["resource_id"], str, f"{path}.resource_id"),
        _number(obj["amount"], f"{path}.amount"),
        _typed(obj.get("level"), int, f"{path}.level", optional=True),
    )


def _ability(value: Any, path: str) -> AbilitySnapshot:
    obj = _object(value, path, {"ability_id", "known", "visible"}, {"ability_id"})
    return AbilitySnapshot(
        _typed(obj["ability_id"], str, f"{path}.ability_id"),
        _typed(obj.get("known", True), bool, f"{path}.known"),
        _typed(obj.get("visible"), bool, f"{path}.visible", optional=True),
    )


def _entity(value: Any, path: str) -> EntitySnapshot:
    fields = {"entity_id", "name", "hp", "max_hp", "armor_class", "position", "alive", "resources", "abilities"}
    obj = _object(value, path, fields, {"entity_id"})
    return EntitySnapshot(
        entity_id=_typed(obj["entity_id"], str, f"{path}.entity_id"),
        name=_typed(obj.get("name"), str, f"{path}.name", optional=True),
        hp=_typed(obj.get("hp"), int, f"{path}.hp", optional=True),
        max_hp=_typed(obj.get("max_hp"), int, f"{path}.max_hp", optional=True),
        armor_class=_typed(obj.get("armor_class"), int, f"{path}.armor_class", optional=True),
        position=_vec3(obj.get("position"), f"{path}.position"),
        alive=_typed(obj.get("alive"), bool, f"{path}.alive", optional=True),
        resources=tuple(_resource(item, f"{path}.resources[{i}]") for i, item in enumerate(_array(obj.get("resources", []), f"{path}.resources"))),
        abilities=tuple(_ability(item, f"{path}.abilities[{i}]") for i, item in enumerate(_array(obj.get("abilities", []), f"{path}.abilities"))),
    )


def _combat(value: Any) -> CombatSnapshot | None:
    if value is None:
        return None
    obj = _object(value, "combat", {"combat_id", "round_number", "active_entity_id", "participant_ids"})
    participants = _array(obj.get("participant_ids", []), "combat.participant_ids")
    return CombatSnapshot(
        combat_id=_typed(obj.get("combat_id"), str, "combat.combat_id", optional=True),
        round_number=_typed(obj.get("round_number"), int, "combat.round_number", optional=True),
        active_entity_id=_typed(obj.get("active_entity_id"), str, "combat.active_entity_id", optional=True),
        participant_ids=tuple(_typed(item, str, f"combat.participant_ids[{i}]") for i, item in enumerate(participants)),
    )


def _candidate(value: Any, path: str) -> DecisionCandidate:
    fields = {"ability_id", "target_kind", "legality", "target_entity_id", "target_position", "legality_source", "legality_reason"}
    obj = _object(value, path, fields, {"ability_id", "target_kind", "legality"})
    return DecisionCandidate(
        ability_id=_typed(obj["ability_id"], str, f"{path}.ability_id"),
        target_kind=_enum(obj["target_kind"], TargetKind, f"{path}.target_kind"),
        legality=_enum(obj["legality"], Legality, f"{path}.legality"),
        target_entity_id=_typed(obj.get("target_entity_id"), str, f"{path}.target_entity_id", optional=True),
        target_position=_vec3(obj.get("target_position"), f"{path}.target_position"),
        legality_source=_typed(obj.get("legality_source"), str, f"{path}.legality_source", optional=True),
        legality_reason=_typed(obj.get("legality_reason"), str, f"{path}.legality_reason", optional=True),
    )


def parse_record(value: Mapping[str, Any]) -> Record:
    """Parse one v1 wire record; absent optional facts retain schema defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("record must be a JSON object")
    base = _object(value, "record", set(value), {"record_type", "schema_version", "sequence"})
    version = _typed(base["schema_version"], int, "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {version}")
    sequence = _typed(base["sequence"], int, "sequence")
    if sequence < 1:
        raise ValueError("sequence must be a positive integer")
    record_type = _typed(base["record_type"], str, "record_type")
    common = {"record_type", "schema_version", "sequence", "timestamp_ms"}
    if record_type == "snapshot":
        obj = _object(value, "snapshot", common | {"controlled_entity_id", "combat", "entities", "candidate_decisions"})
        return GameSnapshot(
            schema_version=version,
            sequence=sequence,
            timestamp_ms=_typed(obj.get("timestamp_ms"), int, "timestamp_ms", optional=True),
            controlled_entity_id=_typed(obj.get("controlled_entity_id"), str, "controlled_entity_id", optional=True),
            combat=_combat(obj.get("combat")),
            entities=tuple(_entity(item, f"entities[{i}]") for i, item in enumerate(_array(obj.get("entities", []), "entities"))),
            candidate_decisions=tuple(_candidate(item, f"candidate_decisions[{i}]") for i, item in enumerate(_array(obj.get("candidate_decisions", []), "candidate_decisions"))),
        )
    if record_type == "event":
        fields = {"kind", "combat_id", "round_number", "actor_id", "target_id", "ability_id", "target_position", "story_action_id", "damage", "damage_type", "critical"}
        obj = _object(value, "event", common | fields, {"kind"})
        return GameEvent(
            schema_version=version,
            sequence=sequence,
            kind=_enum(obj["kind"], EventKind, "kind"),
            timestamp_ms=_typed(obj.get("timestamp_ms"), int, "timestamp_ms", optional=True),
            combat_id=_typed(obj.get("combat_id"), str, "combat_id", optional=True),
            round_number=_typed(obj.get("round_number"), int, "round_number", optional=True),
            actor_id=_typed(obj.get("actor_id"), str, "actor_id", optional=True),
            target_id=_typed(obj.get("target_id"), str, "target_id", optional=True),
            ability_id=_typed(obj.get("ability_id"), str, "ability_id", optional=True),
            target_position=_vec3(obj.get("target_position"), "target_position"),
            story_action_id=_typed(obj.get("story_action_id"), int, "story_action_id", optional=True),
            damage=_typed(obj.get("damage"), int, "damage", optional=True),
            damage_type=_typed(obj.get("damage_type"), str, "damage_type", optional=True),
            critical=_typed(obj.get("critical"), bool, "critical", optional=True),
        )
    raise ValueError(f"Unknown record_type: {record_type!r}")
