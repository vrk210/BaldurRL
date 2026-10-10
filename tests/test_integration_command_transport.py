from dataclasses import replace
import json

import pytest

from integration.command_transport import CommandSession
from integration.commands import (
    ActionCommand, ActionType, AcknowledgementStatus as Status, CommandAcknowledgement, PostExecutionObservation,
)
from integration.mock_executor import MockExecutor, run_demo, synthetic_observation
from integration.reader import RecordDirectoryReader
from integration.schema import DecisionCandidate, EventKind, GameEvent, Legality, TargetKind, Vec3


def write(path, record) -> None:
    path.write_text(json.dumps(record.to_dict()))


def selected(command_id="c1", sequence=1) -> ActionCommand:
    observed = synthetic_observation()
    candidate = observed.candidate_decisions[0]
    return ActionCommand(1, command_id, sequence, observed.controlled_entity_id,
                         ActionType.USE_ABILITY, candidate.target_kind,
                         observed.combat.combat_id, candidate.ability_id,
                         candidate.target_entity_id)


@pytest.fixture
def session(tmp_path) -> CommandSession:
    result = CommandSession(tmp_path / "session-1")
    write(result.records / "1.json", synthetic_observation())
    return result


def test_session_separation_and_unchanged_record_reader(session) -> None:
    session.publish_command(selected())
    MockExecutor(session).receive()
    assert {path.name for path in session.directory.iterdir()} == {"records", "commands", "acknowledgements"}
    records = RecordDirectoryReader(session.records).read_new()
    assert records == (synthetic_observation(),)
    other = CommandSession(session.directory.parent / "session-2")
    assert other.read_commands() == ()


def test_publication_is_complete_atomic_and_never_overwrites(session, monkeypatch) -> None:
    from integration import command_transport
    original_link = command_transport.os.link
    checked = []

    def link(source, destination):
        assert not destination.exists()
        assert list(session.commands.glob("*.json")) == []
        payload = json.loads(open(source).read())
        assert payload == selected().to_dict()
        checked.append(True)
        return original_link(source, destination)

    monkeypatch.setattr(command_transport.os, "link", link)
    path = session.publish_command(selected())
    assert checked == [True]
    assert json.loads(path.read_text()) == selected().to_dict()
    assert list(session.commands.glob("*.tmp")) == []
    with pytest.raises(ValueError, match="Duplicate command_id"):
        session.publish_command(selected())
    assert json.loads(path.read_text()) == selected().to_dict()


def test_failed_atomic_publication_cleans_temp_file_and_lock(session, monkeypatch) -> None:
    from integration import command_transport

    def fail(*args):
        raise OSError("fixture filesystem failure")

    monkeypatch.setattr(command_transport.os, "link", fail)
    with pytest.raises(OSError, match="fixture"):
        session.publish_command(selected())
    assert list(session.commands.iterdir()) == []
    assert not (session.directory / ".writer-lock").exists()


def test_lock_prevents_concurrent_publishers_and_explicit_stale_lock(session) -> None:
    with session.writer_lock():
        with pytest.raises(ValueError, match="busy or interrupted"):
            session.publish_command(selected())
    session.publish_command(selected())


def test_only_one_pending_or_accepted_command_and_no_reuse_after_completion(session) -> None:
    session.publish_command(selected())
    with pytest.raises(ValueError, match="one in-flight"):
        session.publish_command(selected("c2"))
    executor = MockExecutor(session)
    accepted, = executor.receive()
    assert accepted.status is Status.ACCEPTED
    assert not accepted.confirmed_resolved
    with pytest.raises(ValueError, match="one in-flight"):
        session.publish_command(selected("c2"))
    resolved = executor.finish("c1")
    assert resolved.confirmed_resolved
    assert session.in_flight() == ()
    with pytest.raises(ValueError, match="Duplicate command_id"):
        session.publish_command(selected())
    with pytest.raises(ValueError, match="post-execution observation"):
        session.publish_command(selected("c2"))
    write(session.records / "2.json", replace(synthetic_observation(), sequence=2))
    session.publish_post_execution_observation(PostExecutionObservation(1, "c1", 2, "Fixture capture after completion"))
    session.publish_command(selected("c2", 2))


@pytest.mark.parametrize("status,blocked", [(Status.FAILED, False), (Status.UNCERTAIN, True)])
def test_failure_and_uncertainty_are_not_success(session, status, blocked) -> None:
    session.publish_command(selected())
    executor = MockExecutor(session)
    executor.receive()
    result = executor.finish("c1", status)
    assert not result.confirmed_resolved
    if blocked:
        with pytest.raises(ValueError, match="one in-flight"):
            session.publish_command(selected("c2"))
    else:
        with pytest.raises(ValueError, match="post-execution observation"):
            session.publish_command(selected("c2"))
        write(session.records / "2.json", replace(synthetic_observation(), sequence=2))
        session.publish_post_execution_observation(PostExecutionObservation(1, "c1", 2, "Fixture capture after failure"))
        session.publish_command(selected("c2", 2))


def test_stale_rejected_at_publication_and_receive(session) -> None:
    session.publish_command(selected())
    write(session.records / "2.json", replace(synthetic_observation(), sequence=2))
    rejected, = MockExecutor(session).receive()
    assert rejected.status is Status.REJECTED
    assert "Stale" in rejected.reason
    assert session.in_flight() == ()
    with pytest.raises(ValueError, match="Stale"):
        session.publish_command(selected("c2"))
    session.publish_command(selected("c2", sequence=2))


@pytest.mark.parametrize("change,reason", [
    ({"actor_entity_id": "other"}, "controlled"),
    ({"expected_combat_id": "other"}, "combat_id"),
    ({"ability_id": "synthetic-unknown", "target_kind": synthetic_observation().candidate_decisions[1].target_kind,
      "target_entity_id": None}, "LEGAL"),
    ({"target_entity_id": "other"}, "LEGAL"),
])
def test_executor_revalidates_external_commands(session, change, reason) -> None:
    invalid = replace(selected(), **change)
    write(session.commands / "external.json", invalid)
    ack, = MockExecutor(session).receive()
    assert ack.status is Status.REJECTED
    assert reason in ack.reason
    assert session.read_acknowledgements() == (ack,)


def test_legality_change_after_publication_rejected(session) -> None:
    session.publish_command(selected())
    observed = synthetic_observation()
    candidate = replace(observed.candidate_decisions[0], legality=Legality.UNKNOWN)
    # A newer snapshot invalidates the command even if the target still matches.
    write(session.records / "2.json", replace(observed, sequence=2, candidate_decisions=(candidate,)))
    ack, = MockExecutor(session).receive()
    assert ack.status is Status.REJECTED
    assert "Stale" in ack.reason


def test_gaps_and_trailing_events_require_fresh_snapshot(session) -> None:
    write(session.records / "3.json", replace(synthetic_observation(), sequence=3))
    with pytest.raises(ValueError, match="gaps"):
        session.publish_command(selected())
    write(session.records / "2.json", GameEvent(1, 2, EventKind.TURN_ENDED))
    assert session.latest_snapshot().sequence == 3
    write(session.records / "4.json", GameEvent(1, 4, EventKind.TURN_ENDED))
    with pytest.raises(ValueError, match="fresh snapshot"):
        session.publish_command(selected("c2", 3))


def test_duplicate_commands_fail_closed_without_execution(session) -> None:
    session.publish_command(selected())
    write(session.commands / "duplicate.json", selected())
    with pytest.raises(ValueError, match="Duplicate command_id.*duplicate.json"):
        MockExecutor(session).receive()
    assert list(session.acknowledgements.iterdir()) == []


def test_conflicting_external_pending_commands_are_rejected(session) -> None:
    write(session.commands / "1.json", selected())
    write(session.commands / "2.json", selected("c2"))
    acks = MockExecutor(session).receive()
    assert len(acks) == 2
    assert all(ack.status is Status.REJECTED for ack in acks)
    assert all("one in-flight" in ack.reason for ack in acks)


@pytest.mark.parametrize("payload,reason", [
    ("{bad", "Invalid protocol record"),
    ('{"command_id":"c1", "command_id":"c2"}', "Duplicate JSON field"),
    ('{"target_position":NaN}', "Nonstandard JSON constant"),
    ('{"target_position":{"x":1,"x":2}}', "Duplicate JSON field"),
    ('[]', "JSON object"),
])
def test_malformed_inbox_names_file(session, payload, reason) -> None:
    (session.commands / "broken.json").write_text(payload)
    with pytest.raises(ValueError, match=reason) as error:
        MockExecutor(session).receive()
    assert "broken.json" in str(error.value)
    assert list(session.acknowledgements.iterdir()) == []


def test_acknowledgement_transitions_duplicates_and_unknown_ids(session) -> None:
    session.publish_command(selected())
    result = CommandAcknowledgement(1, "c1", 2, Status.RESOLVED, "fixture confirmation")
    with pytest.raises(ValueError, match="requires accepted"):
        session.publish_acknowledgement(result)
    with pytest.raises(ValueError, match="does not match"):
        session.publish_acknowledgement(CommandAcknowledgement(1, "unknown", 1, Status.ACCEPTED))
    accepted = CommandAcknowledgement(1, "c1", 1, Status.ACCEPTED)
    session.publish_acknowledgement(accepted)
    with pytest.raises(ValueError, match="Duplicate"):
        session.publish_acknowledgement(accepted)
    session.publish_acknowledgement(result)
    with pytest.raises(ValueError, match="duplicate or invalid transition"):
        session.publish_acknowledgement(result)
    write(session.acknowledgements / "copy.json", result)
    with pytest.raises(ValueError, match="Duplicate acknowledgement sequence"):
        session.read_acknowledgements()


def test_result_after_rejection_not_allowed(session) -> None:
    session.publish_command(selected())
    session.publish_acknowledgement(CommandAcknowledgement(1, "c1", 1, Status.REJECTED, "fixture rejects"))
    with pytest.raises(ValueError, match="requires accepted"):
        MockExecutor(session).finish("c1")


def test_external_orphan_result_and_malformed_ack_are_detected(session) -> None:
    session.publish_command(selected())
    path = session.acknowledgements / "orphan.json"
    write(path, CommandAcknowledgement(1, "c1", 2, Status.RESOLVED, "fixture"))
    with pytest.raises(ValueError, match="preceding accepted"):
        session.read_acknowledgements()
    write(path, CommandAcknowledgement(1, "other", 1, Status.ACCEPTED))
    with pytest.raises(ValueError, match="unknown command_id"):
        session.read_acknowledgements()
    path.write_text('{"record_type":"acknowledgement","unexpected":1}')
    with pytest.raises(ValueError, match="orphan.json"):
        session.read_acknowledgements()


def test_restart_does_not_replay_accepted_or_finished_command(session) -> None:
    session.publish_command(selected())
    MockExecutor(session).receive()
    reopened = CommandSession(session.directory)
    restarted = MockExecutor(reopened)
    assert restarted.receive() == ()
    assert len(reopened.in_flight()) == 1
    restarted.finish("c1")  # Explicit fixture reconciliation, never automatic.
    assert MockExecutor(CommandSession(session.directory)).receive() == ()
    assert len(session.read_acknowledgements()) == 2
    assert session.latest_snapshot() == synthetic_observation()


def test_demo_observation_to_command_to_acceptance_and_result(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("integration.mock_executor.uuid4", lambda: "fixture-command-id")
    chosen, acknowledgements = run_demo(tmp_path / "demo")
    assert [ack.status for ack in acknowledgements] == [Status.ACCEPTED, Status.RESOLVED]
    assert all(ack.command_id == chosen.command_id for ack in acknowledgements)
    output = capsys.readouterr().out
    assert "observation:" in output and "selected command:" in output
    assert "mock acknowledgement:" in output and "mock result:" in output
    assert "no BG3 execution" in output
    assert "post-execution observation confirmation:" in output
    assert CommandSession(tmp_path / "demo").latest_snapshot().sequence == 2
    with pytest.raises(ValueError, match="fresh, empty session"):
        run_demo(tmp_path / "demo")


@pytest.mark.parametrize("kind,entity,position", [
    (TargetKind.NONE, None, None), (TargetKind.SELF, None, None),
    (TargetKind.ENTITY, "synthetic-enemy", None),
    (TargetKind.POSITION, None, Vec3(2, 3, 4)),
])
def test_mock_all_target_forms(session, kind, entity, position) -> None:
    candidate = DecisionCandidate("synthetic-ability", kind, Legality.LEGAL,
                                  target_entity_id=entity, target_position=position)
    observed = replace(synthetic_observation(), candidate_decisions=(candidate,))
    write(session.records / "1.json", observed)
    intent = replace(selected(), target_kind=kind, target_entity_id=entity, target_position=position)
    session.publish_command(intent)
    executor = MockExecutor(session)
    accepted, = executor.receive()
    assert accepted.status is Status.ACCEPTED
    assert executor.finish(intent.command_id).confirmed_resolved
    assert session.latest_snapshot() == observed


def test_mock_end_turn_is_only_a_handshake(session) -> None:
    intent = replace(selected(), action_type=ActionType.END_TURN, ability_id=None,
                     target_kind=TargetKind.NONE, target_entity_id=None)
    session.publish_command(intent)
    executor = MockExecutor(session)
    accepted, = executor.receive()
    assert accepted.status is Status.ACCEPTED
    executor.finish(intent.command_id)
    assert session.latest_snapshot() == synthetic_observation()
