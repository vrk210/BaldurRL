from dataclasses import replace
import json

import pytest

from integration.commands import (
    ActionCommand, ActionType, AcknowledgementStatus as Status,
    CommandAcknowledgement, parse_acknowledgement, parse_command, validate_command,
)
from integration.schema import (
    CombatSnapshot, DecisionCandidate, GameSnapshot, Legality, TargetKind, Vec3,
)


def command(**changes) -> ActionCommand:
    base = ActionCommand(1, "command-001", 1, "player", ActionType.USE_ABILITY,
                         TargetKind.ENTITY, "combat", "BG3-original-ID", "enemy")
    return replace(base, **changes)


def snapshot(candidate: DecisionCandidate | None = None, **changes) -> GameSnapshot:
    candidate = candidate or DecisionCandidate("BG3-original-ID", TargetKind.ENTITY,
                                              Legality.LEGAL, target_entity_id="enemy")
    base = GameSnapshot(1, 1, controlled_entity_id="player",
                        combat=CombatSnapshot("combat", 1, "player"),
                        candidate_decisions=(candidate,))
    return replace(base, **changes)


@pytest.mark.parametrize("kind,fields", [
    (TargetKind.NONE, {}), (TargetKind.SELF, {}),
    (TargetKind.ENTITY, {"target_entity_id": "enemy"}),
    (TargetKind.POSITION, {"target_position": Vec3(1.25, -2, 3)}),
])
def test_command_serialization_and_exact_target_matching(kind, fields) -> None:
    selected = command(target_kind=kind, target_entity_id=None, **fields) if kind is not TargetKind.ENTITY else command(**fields)
    wire = json.loads(json.dumps(selected.to_dict(), allow_nan=False))
    assert wire["ability_id"] == "BG3-original-ID"
    assert parse_command(wire) == selected
    candidate = DecisionCandidate(selected.ability_id, kind, Legality.LEGAL,
                                  target_entity_id=selected.target_entity_id,
                                  target_position=selected.target_position)
    validate_command(selected, snapshot(candidate))


def test_end_turn_has_no_ability_or_target() -> None:
    selected = command(action_type=ActionType.END_TURN, ability_id=None,
                       target_kind=TargetKind.NONE, target_entity_id=None)
    assert parse_command(selected.to_dict()) == selected
    validate_command(selected, snapshot(candidate_decisions=()))
    with pytest.raises(ValueError, match="verified active combat"):
        validate_command(replace(selected, expected_combat_id=None), snapshot(combat=None))


@pytest.mark.parametrize("changes,match", [
    ({"schema_version": 2}, "schema_version"),
    ({"schema_version": True}, "schema_version"),
    ({"command_id": "../escape"}, "command_id"),
    ({"command_id": ""}, "command_id"),
    ({"snapshot_sequence": True}, "snapshot_sequence"),
    ({"snapshot_sequence": 0}, "snapshot_sequence"),
    ({"actor_entity_id": " "}, "actor_entity_id"),
    ({"expected_combat_id": 1}, "expected_combat_id"),
    ({"action_type": "attack"}, "action_type"),
    ({"action_type": ActionType.USE_ABILITY}, "JSON string"),
    ({"ability_id": None}, "ability_id"),
    ({"ability_id": ""}, "ability_id"),
    ({"target_kind": "unknown"}, "target_kind"),
    ({"target_kind": "self"}, "none/self"),
    ({"target_kind": "none"}, "none/self"),
    ({"target_entity_id": None}, "entity target"),
    ({"target_position": {"x": 1, "y": 2, "z": 3}}, "entity target"),
    ({"target_position": {"x": 1}}, "exactly x, y, z"),
    ({"target_position": {"x": True, "y": 2, "z": 3}}, "finite JSON numbers"),
    ({"target_position": {"x": float("inf"), "y": 2, "z": 3}}, "finite JSON numbers"),
    ({"target_kind": "position", "target_entity_id": None}, "position target"),
    ({"action_type": "end_turn"}, "end_turn"),
    ({"unexpected": 2}, "unknown field"),
    ({"record_type": "event"}, "record_type"),
])
def test_strict_command_validation(changes, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_command(command().to_dict() | changes)


def test_missing_field_and_non_object() -> None:
    wire = command().to_dict()
    del wire["actor_entity_id"]
    with pytest.raises(ValueError, match="missing field.*actor_entity_id"):
        parse_command(wire)
    with pytest.raises(ValueError, match="JSON object"):
        parse_command([])


@pytest.mark.parametrize("legality", [Legality.ILLEGAL, Legality.UNKNOWN])
def test_unknown_never_becomes_legal(legality) -> None:
    candidate = DecisionCandidate("BG3-original-ID", TargetKind.ENTITY, legality,
                                  target_entity_id="enemy", legality_source="test")
    with pytest.raises(ValueError, match="LEGAL"):
        validate_command(command(), snapshot(candidate))


@pytest.mark.parametrize("changes,match", [
    ({"sequence": 2}, "Stale"),
    ({"controlled_entity_id": "other"}, "controlled"),
    ({"controlled_entity_id": None}, "controlled"),
    ({"combat": CombatSnapshot("other", 1, "player")}, "combat_id mismatch"),
    ({"combat": CombatSnapshot("combat", 1, "other")}, "active combat actor"),
    ({"combat": CombatSnapshot("combat", 1, None)}, "active combat actor"),
    ({"candidate_decisions": ()}, "LEGAL"),
])
def test_current_snapshot_validation(changes, match) -> None:
    with pytest.raises(ValueError, match=match):
        validate_command(command(), snapshot(**changes))


def test_missing_expected_combat_is_rejected_when_known() -> None:
    with pytest.raises(ValueError, match="combat_id mismatch"):
        validate_command(command(expected_combat_id=None), snapshot())
    validate_command(command(expected_combat_id=None), snapshot(combat=None))


@pytest.mark.parametrize("changes", [
    {"target_entity_id": "other"}, {"ability_id": "other"},
    {"target_kind": TargetKind.SELF, "target_entity_id": None},
    {"target_kind": TargetKind.POSITION, "target_entity_id": None, "target_position": Vec3(1, 2, 3)},
])
def test_changed_intent_does_not_match_candidate(changes) -> None:
    with pytest.raises(ValueError, match="LEGAL"):
        validate_command(command(**changes), snapshot())


def test_position_match_is_exact_and_duplicate_candidates_are_ambiguous() -> None:
    candidate = DecisionCandidate("BG3-original-ID", TargetKind.POSITION, Legality.LEGAL,
                                  target_position=Vec3(1, 2, 3))
    with pytest.raises(ValueError, match="LEGAL"):
        validate_command(command(target_kind=TargetKind.POSITION, target_entity_id=None,
                                 target_position=Vec3(1, 2, 3.1)), snapshot(candidate))
    observed = snapshot()
    with pytest.raises(ValueError, match="ambiguous"):
        validate_command(command(), replace(observed, candidate_decisions=observed.candidate_decisions * 2))


@pytest.mark.parametrize("status", list(Status))
def test_acknowledgement_round_trip_and_semantics(status) -> None:
    index = 1 if status in (Status.ACCEPTED, Status.REJECTED) else 2
    ack = CommandAcknowledgement(1, "command-001", index, status, "explicit evidence or reason", 5)
    assert parse_acknowledgement(json.loads(json.dumps(ack.to_dict()))) == ack
    assert ack.confirmed_resolved == (status is Status.RESOLVED)
    assert ack.releases_in_flight == (status in (Status.RESOLVED, Status.REJECTED, Status.FAILED))


@pytest.mark.parametrize("status", [Status.REJECTED, Status.RESOLVED, Status.FAILED, Status.UNCERTAIN])
def test_results_need_reason_and_correct_sequence(status) -> None:
    index = 1 if status is Status.REJECTED else 2
    with pytest.raises(ValueError, match="reason"):
        CommandAcknowledgement(1, "c1", index, status)
    with pytest.raises(ValueError, match="acknowledgement_sequence"):
        CommandAcknowledgement(1, "c1", 3, status, "reason")


@pytest.mark.parametrize("changes,match", [
    ({"schema_version": 2}, "schema_version"),
    ({"status": "success"}, "status"),
    ({"observation_sequence": True}, "observation_sequence"),
    ({"acknowledgement_sequence": True}, "acknowledgement_sequence"),
    ({"unexpected": None}, "unknown field"),
])
def test_acknowledgement_strict_parsing(changes, match) -> None:
    wire = CommandAcknowledgement(1, "c1", 1, Status.ACCEPTED).to_dict()
    with pytest.raises(ValueError, match=match):
        parse_acknowledgement(wire | changes)
