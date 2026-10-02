"""The integration wire contract stays independent of simulator state."""

import json
from dataclasses import FrozenInstanceError

import pytest

from integration.schema import (
    SCHEMA_VERSION,
    AbilitySnapshot,
    CombatSnapshot,
    DecisionCandidate,
    EntitySnapshot,
    EventKind,
    GameEvent,
    GameSnapshot,
    Legality,
    ResourceValue,
    TargetKind,
    Vec3,
)


@pytest.mark.parametrize(
    "instance",
    [
        Vec3(1.0, 2.0, 3.0),
        ResourceValue("resource-from-game", 2.5, 3),
        AbilitySnapshot("ability-from-game"),
        EntitySnapshot("entity-from-game"),
        CombatSnapshot(),
        DecisionCandidate("ability-from-game", TargetKind.NONE, Legality.UNKNOWN),
        GameSnapshot(SCHEMA_VERSION, 1),
        GameEvent(SCHEMA_VERSION, 2, EventKind.TURN_STARTED),
    ],
)
def test_core_objects_are_frozen(instance: object) -> None:
    with pytest.raises(FrozenInstanceError):
        instance.new_field = "changed"


def test_resource_and_unknown_entity_facts() -> None:
    resource = ResourceValue("unmapped-resource-id", 2.5, level=4)
    entity = EntitySnapshot("entity-7", resources=(resource,))

    assert (resource.resource_id, resource.amount, resource.level) == ("unmapped-resource-id", 2.5, 4)
    assert entity.name is entity.hp is entity.max_hp is entity.armor_class is None
    assert entity.position is entity.alive is None
    assert entity.abilities == ()


def test_nonfinite_numbers_cannot_enter_wire_values() -> None:
    with pytest.raises(ValueError):
        ResourceValue("resource-id", float("nan"))
    with pytest.raises(ValueError):
        Vec3(0.0, float("inf"), 0.0)


@pytest.mark.parametrize("legality", list(Legality))
def test_candidate_represents_each_legality(legality: Legality) -> None:
    candidate = DecisionCandidate("ability-1", TargetKind.ENTITY, legality, target_entity_id="enemy-1")
    assert candidate.legality is legality


def test_snapshot_serialization_is_deterministic_json() -> None:
    snapshot = GameSnapshot(
        schema_version=SCHEMA_VERSION,
        sequence=8,
        timestamp_ms=None,
        controlled_entity_id="fighter-1",
        combat=CombatSnapshot("combat-1", 2, "fighter-1", ("fighter-1", "enemy-1")),
        entities=(
            EntitySnapshot(
                "fighter-1",
                hp=11,
                max_hp=20,
                position=Vec3(1.0, 2.0, 3.0),
                resources=(ResourceValue("resource-id", 1.5, 2),),
                abilities=(AbilitySnapshot("ability-id", visible=None),),
            ),
        ),
        candidate_decisions=(
            DecisionCandidate("ability-id", TargetKind.ENTITY, Legality.LEGAL, target_entity_id="enemy-1"),
            DecisionCandidate(
                "position-ability", TargetKind.POSITION, Legality.UNKNOWN,
                target_position=Vec3(4.0, 5.0, 6.0), legality_reason="range not checked",
            ),
            DecisionCandidate("blocked-ability", TargetKind.NONE, Legality.ILLEGAL),
        ),
    )

    payload = snapshot.to_dict()
    assert payload == snapshot.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["schema_version"] == 1
    assert payload["combat"]["participant_ids"] == ["fighter-1", "enemy-1"]
    assert payload["entities"][0]["resources"] == [
        {"resource_id": "resource-id", "amount": 1.5, "level": 2}
    ]
    assert payload["candidate_decisions"][0]["target_kind"] == "entity"
    assert payload["candidate_decisions"][0]["target_entity_id"] == "enemy-1"
    assert payload["candidate_decisions"][0]["legality"] == "legal"
    assert payload["candidate_decisions"][1]["target_kind"] == "position"
    assert payload["candidate_decisions"][1]["target_position"] == {"x": 4.0, "y": 5.0, "z": 6.0}
    assert payload["candidate_decisions"][1]["legality"] == "unknown"
    assert payload["candidate_decisions"][2]["legality"] == "illegal"


def test_event_serialization_is_deterministic_json() -> None:
    event = GameEvent(
        schema_version=SCHEMA_VERSION,
        sequence=9,
        kind=EventKind.ABILITY_USED_AT_POSITION,
        combat_id="combat-1",
        actor_id="fighter-1",
        ability_id="ability-id",
        target_position=Vec3(4.0, 5.0, 6.0),
        story_action_id=42,
    )

    payload = event.to_dict()
    assert payload == event.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["schema_version"] == 1
    assert payload["kind"] == "ability_used_at_position"
    assert payload["target_position"] == {"x": 4.0, "y": 5.0, "z": 6.0}
    assert payload["story_action_id"] == 42
    assert [kind.value for kind in EventKind] == [
        "combat_started", "combat_ended", "entered_combat", "round_started", "turn_started",
        "turn_ended", "ability_used", "ability_used_on_target", "ability_used_at_position",
        "damage", "miss", "critical_hit", "died",
    ]
