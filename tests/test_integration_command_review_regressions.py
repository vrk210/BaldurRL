"""Deterministic reproductions of PR #7 review comments."""

from dataclasses import replace
import json

import pytest

from integration.command_transport import CommandSession
from integration.commands import (
    ActionCommand, ActionType, AcknowledgementStatus as Status,
    CommandAcknowledgement, PostExecutionObservation, parse_post_execution_observation,
)
from integration.mock_executor import MockExecutor, synthetic_observation
from integration.reader import RecordDirectoryReader


def write_snapshot(session: CommandSession, sequence: int) -> None:
    observed = replace(synthetic_observation(), sequence=sequence)
    (session.records / f"{sequence}.json").write_text(json.dumps(observed.to_dict()))


def intent(command_id: str, sequence: int = 1) -> ActionCommand:
    observed = synthetic_observation()
    candidate = observed.candidate_decisions[0]
    return ActionCommand(1, command_id, sequence, observed.controlled_entity_id,
                         ActionType.USE_ABILITY, candidate.target_kind,
                         observed.combat.combat_id, candidate.ability_id,
                         candidate.target_entity_id)


@pytest.fixture
def completed(tmp_path):
    session = CommandSession(tmp_path)
    write_snapshot(session, 1)
    session.publish_command(intent("first"))
    executor = MockExecutor(session)
    executor.receive()
    return session, executor


@pytest.mark.parametrize("outcome", [Status.RESOLVED, Status.FAILED])
def test_completed_command_blocks_reuse_of_pre_action_snapshot(completed, outcome) -> None:
    session, executor = completed
    executor.finish("first", outcome)
    with pytest.raises(ValueError, match="post-execution observation"):
        session.publish_command(intent("second"))


@pytest.mark.parametrize("outcome", [Status.RESOLVED, Status.FAILED])
def test_external_commands_require_post_execution_observation(completed, outcome) -> None:
    session, executor = completed
    executor.finish("first", outcome)
    write_snapshot(session, 2)
    # A larger sequence alone does not establish post-action capture timing.
    (session.commands / "second.json").write_text(json.dumps(intent("second", 2).to_dict()))
    rejected, = executor.receive()
    assert rejected.status is Status.REJECTED
    assert "post-execution observation" in rejected.reason


def test_latest_snapshot_validates_the_same_collection_it_selects(tmp_path, monkeypatch) -> None:
    session = CommandSession(tmp_path)
    write_snapshot(session, 1)
    original_read = RecordDirectoryReader.read_all
    calls = []

    def publish_after_collection(reader):
        records = original_read(reader)
        calls.append(records)
        if len(calls) == 1:
            # Deterministic publication between old validation and selection.
            write_snapshot(session, 3)
        return records

    monkeypatch.setattr(RecordDirectoryReader, "read_all", publish_after_collection)
    assert session.latest_snapshot().sequence == 1
    assert len(calls) == 1
    with pytest.raises(ValueError, match="gaps.*2"):
        session.latest_snapshot()


@pytest.mark.parametrize("outcome", [Status.RESOLVED, Status.FAILED])
def test_fresh_attested_capture_unblocks_publication_and_receipt(completed, outcome) -> None:
    session, executor = completed
    # A newer snapshot already published before completion is part of the fence.
    write_snapshot(session, 2)
    result = executor.finish("first", outcome)
    assert result.observation_boundary_sequence == 2
    with pytest.raises(ValueError, match="completion observation boundary"):
        session.publish_post_execution_observation(PostExecutionObservation(1, "first", 2, "Not fresh"))
    write_snapshot(session, 3)
    with pytest.raises(ValueError, match="post-execution observation"):
        session.publish_command(intent("second", 3))
    # The fixture initiates this capture after observing result; the persisted
    # attestation carries that producer assertion, not just sequence ordering.
    session.publish_post_execution_observation(PostExecutionObservation(1, "first", 3, "Fixture capture initiated after terminal acknowledgement"))
    reopened = CommandSession(session.directory)
    reopened.publish_command(intent("second", 3))
    accepted, = MockExecutor(reopened).receive()
    assert accepted.status is Status.ACCEPTED


def test_external_inbox_with_attested_capture_can_be_accepted(completed) -> None:
    session, executor = completed
    executor.finish("first")
    write_snapshot(session, 2)
    session.publish_post_execution_observation(PostExecutionObservation(1, "first", 2, "Fixture capture after completion"))
    (session.commands / "second.json").write_text(json.dumps(intent("second", 2).to_dict()))
    accepted, = executor.receive()
    assert accepted.status is Status.ACCEPTED


def test_genuine_rejection_permits_retry_from_unchanged_snapshot(tmp_path) -> None:
    session = CommandSession(tmp_path)
    write_snapshot(session, 1)
    invalid = replace(intent("first"), actor_entity_id="other")
    (session.commands / "first.json").write_text(json.dumps(invalid.to_dict()))
    executor = MockExecutor(session)
    rejected, = executor.receive()
    assert rejected.status is Status.REJECTED
    session.publish_command(intent("retry"))
    accepted, = executor.receive()
    assert accepted.status is Status.ACCEPTED


@pytest.mark.parametrize("outcome", [Status.ACCEPTED, Status.REJECTED, Status.UNCERTAIN])
def test_confirmation_requires_known_completed_result(completed, outcome) -> None:
    session, executor = completed
    if outcome is Status.UNCERTAIN:
        executor.finish("first", outcome)
    elif outcome is Status.REJECTED:
        # Use a distinct external intent rejected before execution.
        invalid = replace(intent("rejected"), actor_entity_id="other")
        (session.commands / "rejected.json").write_text(json.dumps(invalid.to_dict()))
        executor.receive()
    command_id = "rejected" if outcome is Status.REJECTED else "first"
    write_snapshot(session, 2)
    with pytest.raises(ValueError, match="resolved or failed"):
        session.publish_post_execution_observation(PostExecutionObservation(1, command_id, 2, "Invalid confirmation"))


def test_confirmation_checks_target_record_and_detects_duplicates(completed) -> None:
    session, executor = completed
    executor.finish("first")
    # An event is not post-execution snapshot evidence.
    from integration.schema import EventKind, GameEvent
    (session.records / "2.json").write_text(json.dumps(GameEvent(1, 2, EventKind.MISS).to_dict()))
    for command_id, sequence, match in [("first", 2, "existing snapshot"), ("first", 3, "existing snapshot"), ("missing", 2, "resolved or failed")]:
        with pytest.raises(ValueError, match=match):
            session.publish_post_execution_observation(PostExecutionObservation(1, command_id, sequence, "Fixture"))
    write_snapshot(session, 3)
    confirmation = PostExecutionObservation(1, "first", 3, "Fixture capture after result")
    session.publish_post_execution_observation(confirmation)
    with pytest.raises(ValueError, match="Duplicate"):
        session.publish_post_execution_observation(confirmation)
    (session.acknowledgements / "duplicate-proof.json").write_text(json.dumps(confirmation.to_dict()))
    with pytest.raises(ValueError, match="Duplicate post-execution observation"):
        session.publish_command(intent("second", 3))


def test_external_confirmation_cannot_bypass_boundary(completed) -> None:
    session, executor = completed
    write_snapshot(session, 2)
    executor.finish("first")
    invalid = PostExecutionObservation(1, "first", 2, "Already published snapshot")
    (session.acknowledgements / "forged.json").write_text(json.dumps(invalid.to_dict()))
    (session.commands / "second.json").write_text(json.dumps(intent("second", 2).to_dict()))
    rejected, = executor.receive()
    assert rejected.status is Status.REJECTED
    assert "completion observation boundary" in rejected.reason


def test_result_without_boundary_cannot_unblock_old_sessions(completed) -> None:
    session, executor = completed
    legacy = CommandAcknowledgement(1, "first", 2, Status.RESOLVED, "Old fixture")
    (session.acknowledgements / "first.2.json").write_text(json.dumps(legacy.to_dict()))
    write_snapshot(session, 2)
    with pytest.raises(ValueError, match="completion observation boundary"):
        session.publish_post_execution_observation(PostExecutionObservation(1, "first", 2, "No stored boundary"))
    with pytest.raises(ValueError, match="post-execution observation"):
        session.publish_command(intent("second", 2))


def test_completion_boundary_cannot_precede_published_records(completed) -> None:
    session, executor = completed
    write_snapshot(session, 2)
    result = CommandAcknowledgement(1, "first", 2, Status.FAILED, "Partial effects", observation_boundary_sequence=1)
    with pytest.raises(ValueError, match="precedes observations"):
        session.publish_acknowledgement(result)


@pytest.mark.parametrize("changes,match", [
    ({"schema_version": 2}, "schema_version"),
    ({"command_id": "../invalid"}, "command_id"),
    ({"snapshot_sequence": True}, "snapshot_sequence"),
    ({"reason": ""}, "reason"),
    ({"extra": 1}, "unknown field"),
])
def test_post_execution_confirmation_strict_roundtrip(changes, match) -> None:
    confirmation = PostExecutionObservation(1, "first", 2, "Capture initiated after completion")
    wire = json.loads(json.dumps(confirmation.to_dict()))
    assert parse_post_execution_observation(wire) == confirmation
    with pytest.raises(ValueError, match=match):
        parse_post_execution_observation(wire | changes)


@pytest.mark.parametrize("value", [0, True, "1"])
def test_completion_boundary_requires_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="observation_boundary_sequence"):
        CommandAcknowledgement(1, "first", 2, Status.RESOLVED, "Fixture", observation_boundary_sequence=value)


@pytest.mark.parametrize("outcome", [Status.RESOLVED, Status.FAILED])
def test_supporting_observation_sequence_is_not_freshness_confirmation(completed, outcome) -> None:
    session, executor = completed
    write_snapshot(session, 2)
    result = CommandAcknowledgement(1, "first", 2, outcome, "Fixture completion", observation_sequence=2)
    session.publish_acknowledgement(result)
    write_snapshot(session, 3)
    with pytest.raises(ValueError, match="post-execution observation"):
        session.publish_command(intent("second", 3))


def test_malformed_external_confirmation_is_named_and_fails_closed(completed) -> None:
    session, executor = completed
    executor.finish("first")
    path = session.acknowledgements / "broken-proof.json"
    path.write_text('{"record_type":"post_execution_observation","command_id":"first"}')
    with pytest.raises(ValueError, match="broken-proof.json.*missing field"):
        session.read_acknowledgements()
