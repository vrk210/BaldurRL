import json

import pytest

from integration.schema import (
    SCHEMA_VERSION, AbilitySnapshot, CombatSnapshot, DecisionCandidate,
    EntitySnapshot, EventKind, GameEvent, GameSnapshot, Legality,
    ResourceValue, TargetKind, Vec3,
)
from integration.serialization import parse_record, record_to_dict


def test_full_snapshot_round_trip() -> None:
    snapshot = GameSnapshot(
        SCHEMA_VERSION, 3, 123, "player-1",
        CombatSnapshot("combat-1", 2, "player-1", ("player-1", "enemy-1")),
        (EntitySnapshot("player-1", "Player", 12, 20, 16, Vec3(1, 2, 3), True,
                        (ResourceValue("resource-action", 1.5, 2),),
                        (AbilitySnapshot("ability-1", True, None),)),),
        (DecisionCandidate("ability-1", TargetKind.ENTITY, Legality.UNKNOWN,
                           target_entity_id="enemy-1", legality_source="test"),),
    )
    wire = record_to_dict(snapshot)
    assert wire["record_type"] == "snapshot"
    assert wire["candidate_decisions"][0]["legality"] == "unknown"
    assert parse_record(json.loads(json.dumps(wire, allow_nan=False))) == snapshot
    assert record_to_dict(snapshot) == wire


def test_full_event_round_trip() -> None:
    event = GameEvent(SCHEMA_VERSION, 4, EventKind.ABILITY_USED_AT_POSITION,
                      target_position=Vec3(4, 5, 6), actor_id="player-1",
                      ability_id="ability-1", story_action_id=9, critical=False)
    wire = record_to_dict(event)
    assert wire["record_type"] == "event"
    assert wire["kind"] == "ability_used_at_position"
    assert parse_record(json.loads(json.dumps(wire, allow_nan=False))) == event


def test_missing_optional_fields_remain_unknown() -> None:
    snapshot = parse_record({"record_type": "snapshot", "schema_version": 1, "sequence": 1,
                             "entities": [{"entity_id": "unknown"}]})
    assert snapshot == GameSnapshot(1, 1, entities=(EntitySnapshot("unknown"),))
    event = parse_record({"record_type": "event", "schema_version": 1, "sequence": 2,
                          "kind": "miss"})
    assert event == GameEvent(1, 2, EventKind.MISS)


@pytest.mark.parametrize("change,match", [
    ({"schema_version": 2}, "Unsupported schema_version"),
    ({"record_type": "other"}, "Unknown record_type"),
    ({"record_type": None}, "record_type"),
    ({"sequence": True}, "sequence"),
    ({"sequence": 0}, "sequence"),
    ({"entities": {}}, "entities"),
    ({"entities": [{"entity_id": 4}]}, "entity_id"),
    ({"entities": [{"entity_id": "x", "resources": [{"resource_id": "r", "amount": float("nan") }]}]}, "amount"),
    ({"candidate_decisions": [{"ability_id": "a", "target_kind": "bad", "legality": "unknown"}]}, "target_kind"),
    ({"unexpected": 1}, "unknown field"),
])
def test_malformed_snapshot_rejected(change: dict, match: str) -> None:
    wire = {"record_type": "snapshot", "schema_version": 1, "sequence": 1} | change
    with pytest.raises(ValueError, match=match):
        parse_record(wire)


def test_malformed_event_rejected() -> None:
    wire = {"record_type": "event", "schema_version": 1, "sequence": 2, "kind": "bad"}
    with pytest.raises(ValueError, match="kind"):
        parse_record(wire)
