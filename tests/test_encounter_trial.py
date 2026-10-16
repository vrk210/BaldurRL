from dataclasses import replace
import json

import pytest

from integration.command_transport import CommandSession
from integration.commands import (
    ActionCommand, ActionType, AcknowledgementStatus, CommandAcknowledgement,
    PostExecutionObservation,
)
from integration.encounter_trial import (
    CaptureOrigin, Status, TrialKind, analyze_file, analyze_trial, init_trial, main, parse_trial,
    trial_template,
)
from integration.schema import (
    CombatSnapshot, EntitySnapshot, EventKind, GameEvent, GameSnapshot, ResourceValue, TargetKind,
)


def snapshot(sequence, amount, *, level=0, active="actor", round_number=1):
    return GameSnapshot(
        1, sequence, controlled_entity_id="actor",
        combat=CombatSnapshot("combat", round_number, active, ("actor", "target")),
        entities=(
            EntitySnapshot("actor", resources=(ResourceValue("ActionPoint", amount, level),)),
            EntitySnapshot("target", hp=10),
        ),
    )


def write_records(directory, records):
    directory.mkdir(exist_ok=True)
    for record in records:
        (directory / f"{record.sequence}.json").write_text(json.dumps(record.to_dict()))


def trial(kind="human_attack", **changes):
    value = trial_template(TrialKind(kind))
    value.update({
        "trial_id": "trial-1", "capture_origin": "synthetic",
        "actor_entity_id": "actor", "target_entity_id": "target",
        "ability_id": "Attack_MainHand", "combat_id": "combat",
        "before_sequence": 1, "after_sequence": 4, "story_action_id": 0,
        "expected_resource_deltas": [
            {"resource_id": "ActionPoint", "level": 0, "expected_delta": -1},
        ],
        "completion_evidence": {"source": "fixture", "reason": "synthetic resolved marker"},
        "notes": "Synthetic fixture; no BG3 claim.",
    })
    value.update(changes)
    return parse_trial(value)


def attack_events(*, target="target", ability="Attack_MainHand", actor="actor", story=0,
                  outcome=EventKind.DAMAGE):
    return (
        GameEvent(1, 2, EventKind.ABILITY_USED, combat_id="combat", actor_id=actor,
                  ability_id=ability, story_action_id=story),
        GameEvent(1, 3, EventKind.ABILITY_USED_ON_TARGET, combat_id="combat", actor_id=actor,
                  target_id=target, ability_id=ability, story_action_id=story),
        GameEvent(1, 4, outcome, combat_id="combat", actor_id=actor, target_id=target,
                  ability_id=ability, story_action_id=story),
    )


def fixture_records(*, after_amount=0, outcome=EventKind.DAMAGE):
    events = attack_events(outcome=outcome)
    return (snapshot(1, 1), events[0], events[1], events[2], snapshot(5, after_amount))


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_init_unknown_template_is_strict_and_inconclusive(tmp_path):
    path = tmp_path / "trial.json"
    init_trial(path, TrialKind.HUMAN_ATTACK)
    with pytest.raises(ValueError, match="replace"):
        init_trial(path, TrialKind.HUMAN_ATTACK)
    records = tmp_path / "records"
    write_records(records, (snapshot(1, 1), snapshot(2, 1)))
    report = analyze_file(path, records)
    assert report["status"] == "INCONCLUSIVE"
    assert report["live_verified"] is False
    assert "actor_entity_id" in check(report, "trial_facts")["reason"]


def test_unknown_capture_origin_prevents_pass(tmp_path):
    records = tmp_path / "records"
    write_records(records, fixture_records())
    report = analyze_trial(replace(trial(), capture_origin=CaptureOrigin.UNKNOWN, after_sequence=5), records)
    assert report["status"] == "INCONCLUSIVE"
    assert check(report, "capture_origin")["status"] == "INCONCLUSIVE"


def test_valid_human_attack_and_correlated_generic_targeted_events(tmp_path):
    records = tmp_path / "records"
    write_records(records, fixture_records())
    spec = replace(trial(), after_sequence=5)
    report = analyze_trial(spec, records)
    assert report["status"] == "PASS"
    assert check(report, "action")["supporting_record_sequences"] == [2, 3]
    assert report["observed_resource_deltas"][0]["observed_delta"] == -1
    assert report["provenance"]["capture_origin"] == "synthetic"
    assert report["live_verified"] is False


def test_miss_is_execution(tmp_path):
    records = tmp_path / "records"
    write_records(records, fixture_records(outcome=EventKind.MISS))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "PASS"
    assert check(report, "action")["status"] == "PASS"


def test_rejection_evidence_contradicts_attack_trial(tmp_path):
    records = tmp_path / "records"
    write_records(records, fixture_records())
    evidence = parse_trial({
        **trial_template(TrialKind.HUMAN_ATTACK),
        "rejection_evidence": {"source": "operator", "reason": "reported rejection"},
    }).rejection_evidence
    report = analyze_trial(replace(trial(), after_sequence=5, rejection_evidence=evidence), records)
    assert report["status"] == "FAIL"
    assert "contradicts" in check(report, "completion")["reason"]


@pytest.mark.parametrize("change, phrase", [
    ({"target": "other"}, "target"),
    ({"ability": "OtherAbility"}, "ability"),
    ({"actor": "other"}, "wrong actor"),
])
def test_wrong_action_identity_fails(tmp_path, change, phrase):
    records = tmp_path / "records"
    events = attack_events(**change)
    write_records(records, (snapshot(1, 1), *events, snapshot(5, 0)))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "FAIL"
    assert phrase in check(report, "action")["reason"].lower()


@pytest.mark.parametrize('target', [None, 'target'])
@pytest.mark.parametrize('story', [None, 99])
def test_unrelated_actor_using_shared_ability_does_not_fail_trial(tmp_path, target, story):
    records = tmp_path / 'records'
    events = attack_events()
    unrelated = GameEvent(1, 4, EventKind.ABILITY_USED if target is None else EventKind.ABILITY_USED_ON_TARGET,
                          actor_id='other', target_id=target, ability_id='Attack_MainHand', story_action_id=story)
    write_records(records, (snapshot(1, 1), *events[:2], unrelated,
                            replace(events[2], sequence=5), snapshot(6, 0)))
    assert analyze_trial(replace(trial(), after_sequence=6), records)['status'] == 'PASS'


def test_wrong_actor_without_action_correlation_is_inconclusive(tmp_path):
    records = tmp_path / 'records'
    events = attack_events(actor='other', story=None)
    write_records(records, (snapshot(1, 1), *events, snapshot(5, 0)))
    report = analyze_trial(trial(after_sequence=5, story_action_id=None), records)
    assert check(report, 'action')['status'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('after,expected,status', [
    (0.2, -0.1, 'PASS'), (0.2000001, -0.1, 'PASS'),
    (0.2001, -0.1, 'FAIL'), (0.3, -1, 'FAIL'),
])
def test_resource_delta_allows_rounding_but_rejects_actual_cost_mismatch(tmp_path, after, expected, status):
    records = tmp_path / 'records'
    write_records(records, (snapshot(1, 0.3), *attack_events(), snapshot(5, after)))
    spec = trial(after_sequence=5, expected_resource_deltas=[
        {'resource_id': 'ActionPoint', 'level': 0, 'expected_delta': expected},
    ])
    assert check(analyze_trial(spec, records), 'resources')['status'] == status


def test_public_feedback_reads_results_and_validated_attestations(tmp_path):
    session = autonomous_session(tmp_path / 'session', None, resolved=True)
    write_records(session.records, fixture_records())
    feedback = session.read_feedback()
    assert [item.status for item in feedback if isinstance(item, CommandAcknowledgement)] == [
        AcknowledgementStatus.ACCEPTED, AcknowledgementStatus.RESOLVED,
    ]
    assert [item.snapshot_sequence for item in feedback if isinstance(item, PostExecutionObservation)] == [5]
    (session.acknowledgements / 'command-1.observation.json').write_text(json.dumps(
        PostExecutionObservation(1, 'command-1', 1, 'bad boundary').to_dict()))
    with pytest.raises(ValueError, match='completion observation boundary'):
        session.read_feedback()


def test_distinct_and_missing_story_ids_make_multiple_actions_fail(tmp_path):
    records = tmp_path / "records"
    write_records(records, (
        snapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED_ON_TARGET, actor_id="actor", target_id="target",
                  ability_id="Attack_MainHand", story_action_id=None),
        GameEvent(1, 3, EventKind.ABILITY_USED_ON_TARGET, actor_id="actor", target_id="target",
                  ability_id="Attack_MainHand", story_action_id=8),
        snapshot(4, 0),
    ))
    report = analyze_trial(replace(trial(story_action_id=None), after_sequence=4), records)
    assert report["status"] == "FAIL"
    assert "Multiple distinct" in check(report, "action")["reason"]


def test_missing_resource_is_unknown_and_never_zero_filled(tmp_path):
    records = tmp_path / "records"
    after = replace(snapshot(5, 0), entities=(EntitySnapshot("actor"), EntitySnapshot("target")))
    write_records(records, (snapshot(1, 1), *attack_events(), after))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "INCONCLUSIVE"
    assert report["observed_resource_deltas"][0]["observed_delta"] is None
    assert check(report, "resources")["status"] == "INCONCLUSIVE"


def test_resource_level_is_part_of_key_and_numeric_delta_is_reported(tmp_path):
    records = tmp_path / "records"
    before = snapshot(1, 2.5, level=3)
    after = snapshot(5, 1.25, level=3)
    write_records(records, (before, *attack_events(), after))
    spec = replace(trial(), after_sequence=5, expected_resource_deltas=(
        replace(trial().expected_resource_deltas[0], level=3, expected_delta=-1.25),
    ))
    report = analyze_trial(spec, records)
    assert report["status"] == "PASS"
    assert report["observed_resource_deltas"][0]["level"] == 3
    assert report["observed_resource_deltas"][0]["observed_delta"] == -1.25


def test_turn_refresh_makes_cost_attribution_inconclusive(tmp_path):
    records = tmp_path / "records"
    write_records(records, (
        snapshot(1, 1), *attack_events()[:2],
        GameEvent(1, 4, EventKind.TURN_ENDED, combat_id="combat", actor_id="actor"),
        snapshot(5, 1, active="target", round_number=1),
    ))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "INCONCLUSIVE"
    assert check(report, "turn_boundary")["status"] == "INCONCLUSIVE"
    assert check(report, "resources")["status"] == "INCONCLUSIVE"


def autonomous_session(path, records, *, resolved):
    session = CommandSession(path)
    command = ActionCommand(1, "command-1", 1, "actor", ActionType.USE_ABILITY,
                            TargetKind.ENTITY, "combat", "Attack_MainHand", "target")
    (session.commands / "command-1.json").write_text(json.dumps(command.to_dict()))
    accepted = CommandAcknowledgement(1, "command-1", 1, AcknowledgementStatus.ACCEPTED)
    (session.acknowledgements / "command-1.1.json").write_text(json.dumps(accepted.to_dict()))
    if resolved:
        done = CommandAcknowledgement(1, "command-1", 2, AcknowledgementStatus.RESOLVED,
                                      "synthetic resolution", observation_boundary_sequence=3)
        (session.acknowledgements / "command-1.2.json").write_text(json.dumps(done.to_dict()))
        confirmation = PostExecutionObservation(1, "command-1", 5, "synthetic fresh capture")
        (session.acknowledgements / "command-1.observation.json").write_text(json.dumps(confirmation.to_dict()))
    return session


def test_autonomous_accepted_only_is_inconclusive(tmp_path):
    session = autonomous_session(tmp_path / "session", None, resolved=False)
    records = session.records
    write_records(records, fixture_records())
    spec = replace(trial("autonomous_attack", command_id="command-1", completion_evidence=None),
                   after_sequence=5)
    report = analyze_trial(spec, records, session.directory)
    assert report["status"] == "INCONCLUSIVE"
    assert "acceptance alone" in check(report, "completion")["reason"]


def test_autonomous_resolved_with_confirmation_passes(tmp_path):
    session = autonomous_session(tmp_path / "session", None, resolved=True)
    records = session.records
    write_records(records, fixture_records())
    spec = replace(trial("autonomous_attack", command_id="command-1", completion_evidence=None),
                   after_sequence=5)
    report = analyze_trial(spec, records, session.directory)
    assert report["status"] == "PASS"


def test_mismatched_autonomous_command_fails(tmp_path):
    session = autonomous_session(tmp_path / "session", None, resolved=True)
    records = session.records
    write_records(records, fixture_records())
    spec = replace(trial("autonomous_attack", command_id="command-1", completion_evidence=None,
                         ability_id="Different"), after_sequence=5)
    report = analyze_trial(spec, records, session.directory)
    assert report["status"] == "FAIL"
    assert "ability_id" in check(report, "command_intent")["reason"]


def test_resource_rejection_requires_explicit_evidence_and_measurement(tmp_path):
    records = tmp_path / "records"
    write_records(records, (snapshot(1, 0), snapshot(2, 0)))
    base = replace(trial("resource_rejection", after_sequence=2, story_action_id=None,
                         completion_evidence=None), expected_resource_deltas=(
                             replace(trial().expected_resource_deltas[0], expected_delta=0),
                         ))
    missing = analyze_trial(base, records)
    assert missing["status"] == "INCONCLUSIVE"
    verified = analyze_trial(replace(base, rejection_evidence=parse_trial({
        **trial_template(TrialKind.RESOURCE_REJECTION),
        "rejection_evidence": {"source": "operator", "reason": "ordinary UI rejected attempt"},
    }).rejection_evidence), records)
    assert verified["status"] == "PASS"


def test_resource_rejection_outcome_without_attempt_event_is_contradictory(tmp_path):
    records = tmp_path / "records"
    write_records(records, (
        snapshot(1, 0),
        GameEvent(1, 2, EventKind.MISS, actor_id="actor", target_id="target",
                  ability_id="Attack_MainHand"),
        snapshot(3, 0),
    ))
    evidence = parse_trial({
        **trial_template(TrialKind.RESOURCE_REJECTION),
        "rejection_evidence": {"source": "operator", "reason": "reported rejection"},
    }).rejection_evidence
    spec = replace(trial("resource_rejection", completion_evidence=None), after_sequence=3,
                   story_action_id=None, rejection_evidence=evidence,
                   expected_resource_deltas=(replace(trial().expected_resource_deltas[0], expected_delta=0),))
    report = analyze_trial(spec, records)
    assert report["status"] == "FAIL"
    assert "outcome evidence" in check(report, "action")["reason"]


@pytest.mark.parametrize("change, reason", [
    ({"extra": 1}, "unknown field"),
    ({"schema_version": 2}, "Unsupported"),
    ({"trial_id": "bad/id"}, "filename-safe"),
    ({"before_sequence": 4, "after_sequence": 4}, "less than"),
    ({"story_action_id": -1}, "nonnegative"),
    ({"expected_resource_deltas": "not-an-array"}, "JSON array"),
])
def test_malformed_trial_specs_are_rejected(change, reason):
    value = trial_template(TrialKind.HUMAN_ATTACK)
    value.update(change)
    with pytest.raises(ValueError, match=reason):
        parse_trial(value)


def test_sequence_gap_is_precise_failure(tmp_path):
    records = tmp_path / "records"
    write_records(records, (snapshot(1, 1), snapshot(3, 0)))
    report = analyze_trial(replace(trial(), after_sequence=3), records)
    assert report["status"] == "FAIL"
    assert "gaps" in check(report, "trace_window")["reason"]


def test_missing_initial_sequence_is_a_gap(tmp_path):
    records = tmp_path / "records"
    write_records(records, (snapshot(2, 1), snapshot(3, 0)))
    report = analyze_trial(replace(trial(), before_sequence=2, after_sequence=3), records)
    assert report["status"] == "FAIL"
    assert "1" in check(report, "trace_window")["reason"]


def test_malformed_null_evidence_is_validation_error():
    value = trial_template(TrialKind.HUMAN_ATTACK)
    value["completion_evidence"] = {"source": None, "reason": "reported"}
    with pytest.raises(ValueError, match="nonempty strings"):
        parse_trial(value)


def test_boundary_actor_mismatch_fails_and_unknown_turn_is_inconclusive(tmp_path):
    records = tmp_path / "records"
    wrong = replace(snapshot(5, 0), controlled_entity_id="other")
    write_records(records, (snapshot(1, 1), *attack_events(), wrong))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "FAIL"
    assert check(report, "controlled_actor")["status"] == "FAIL"

    unknown_records = tmp_path / "unknown"
    unknown_after = replace(snapshot(5, 0), combat=CombatSnapshot("combat", None, None, ()))
    write_records(unknown_records, (snapshot(1, 1), *attack_events(), unknown_after))
    unknown = analyze_trial(replace(trial(), after_sequence=5), unknown_records)
    assert unknown["status"] == "INCONCLUSIVE"
    assert check(unknown, "turn_boundary")["status"] == "INCONCLUSIVE"


def test_boundary_combat_mismatch_fails(tmp_path):
    records = tmp_path / "records"
    wrong = replace(snapshot(5, 0), combat=CombatSnapshot("other-combat", 1, "actor", ()))
    write_records(records, (snapshot(1, 1), *attack_events(), wrong))
    report = analyze_trial(replace(trial(), after_sequence=5), records)
    assert report["status"] == "FAIL"
    assert check(report, "combat")["status"] == "FAIL"


def test_autonomous_records_must_belong_to_session_and_analysis_does_not_create_dirs(tmp_path):
    external_records = tmp_path / "external"
    write_records(external_records, fixture_records())
    session = autonomous_session(tmp_path / "session", None, resolved=False)
    spec = replace(trial("autonomous_attack", command_id="command-1", completion_evidence=None),
                   after_sequence=5)
    with pytest.raises(ValueError, match="records subdirectory"):
        analyze_trial(spec, external_records, session.directory)

    incomplete = tmp_path / "incomplete"
    incomplete_records = incomplete / "records"
    incomplete_records.mkdir(parents=True)
    for record in fixture_records():
        (incomplete_records / f"{record.sequence}.json").write_text(json.dumps(record.to_dict()))
    with pytest.raises(ValueError, match="protocol subdirectory"):
        analyze_trial(spec, incomplete_records, incomplete)
    assert not (incomplete / "commands").exists()
    assert not (incomplete / "acknowledgements").exists()


def test_output_cannot_overwrite_trial_or_enter_evidence_directories(tmp_path, capsys):
    session = CommandSession(tmp_path / "session")
    write_records(session.records, fixture_records())
    trial_path = tmp_path / "trial.json"
    trial_value = trial_template(TrialKind.HUMAN_ATTACK)
    trial_path.write_text(json.dumps(trial_value), encoding="utf-8")
    command_path = session.commands / "evidence.json"
    acknowledgement_path = session.acknowledgements / "evidence.json"
    command_path.write_text("command evidence", encoding="utf-8")
    acknowledgement_path.write_text("ack evidence", encoding="utf-8")
    candidates = (
        trial_path,
        session.records / "1.json",
        command_path,
        acknowledgement_path,
        session.records / "new-report.json",
        session.commands / "new-report.json",
        session.acknowledgements / "new-report.json",
    )
    before = {path: path.read_bytes() for path in candidates if path.exists()}
    for destination in candidates:
        assert main(("analyze", str(trial_path), "--records", str(session.records),
                     "--session-dir", str(session.directory), "--output", str(destination))) == 2
        assert not destination.exists() or destination.read_bytes() == before[destination]
    error = capsys.readouterr().err
    assert error.count('"status": "ERROR"') == len(candidates)
