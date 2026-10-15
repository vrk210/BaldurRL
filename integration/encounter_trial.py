"""Offline, evidence-bounded analysis of one BG3 encounter trial.

This module never executes a game action.  It compares an operator-authored
trial specification with an immutable observation stream and, for autonomous
trials, the existing command protocol records.

Use `init trial.json --kind human_attack` to create a template, then fill its
observed IDs, before/after snapshot sequences, measured resource deltas and
completion evidence. `analyze trial.json --records SESSION/records` exits 0 for
PASS, 1 for FAIL and 2 for INCONCLUSIVE or a validation error. Autonomous trials
also need --session-dir SESSION and linked resolved-result/capture attestations;
acceptance alone never proves completion. Missing resources remain unknown.

Resource comparisons allow 1e-6 absolute rounding error (no relative tolerance),
so fractional movement amounts can be compared without masking whole-unit costs.
Turn/round boundaries make resource attribution inconclusive. --output must be
outside the raw evidence directories. A synthetic PASS is not live verification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from math import isclose, isfinite
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from .command_transport import CommandSession
from .commands import (
    ActionCommand, ActionType, AcknowledgementStatus, CommandAcknowledgement,
    PostExecutionObservation,
)
from .reader import RecordDirectoryReader, validate_sequence
from .schema import EntitySnapshot, EventKind, GameEvent, GameSnapshot, ResourceValue, TargetKind
from .serialization import Record


TRIAL_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ACTION_KINDS = frozenset({
    EventKind.ABILITY_USED,
    EventKind.ABILITY_USED_ON_TARGET,
    EventKind.ABILITY_USED_AT_POSITION,
})
_TURN_KINDS = frozenset({EventKind.TURN_STARTED, EventKind.TURN_ENDED, EventKind.ROUND_STARTED})


class TrialKind(Enum):
    HUMAN_ATTACK = "human_attack"
    AUTONOMOUS_ATTACK = "autonomous_attack"
    RESOURCE_REJECTION = "resource_rejection"


class CaptureOrigin(Enum):
    UNKNOWN = "unknown"
    SYNTHETIC = "synthetic"
    LIVE_OPERATOR_REPORTED = "live_operator_reported"


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Evidence:
    source: str
    reason: str


@dataclass(frozen=True)
class ExpectedResourceDelta:
    resource_id: str
    level: int | None
    expected_delta: float


@dataclass(frozen=True)
class Trial:
    schema_version: int
    trial_id: str | None
    kind: TrialKind
    capture_origin: CaptureOrigin
    actor_entity_id: str | None
    target_entity_id: str | None
    ability_id: str | None
    combat_id: str | None
    before_sequence: int | None
    after_sequence: int | None
    story_action_id: int | None
    command_id: str | None
    expected_resource_deltas: tuple[ExpectedResourceDelta, ...]
    completion_evidence: Evidence | None
    rejection_evidence: Evidence | None
    game_version: str | None
    script_extender_version: str | None
    notes: str | None


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    reason: str
    supporting_record_sequences: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "supporting_record_sequences": list(self.supporting_record_sequences),
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant {value}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _object(value: Any, path: str, fields: set[str], required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{path} must be a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - fields
    if missing:
        raise ValueError(f"{path} missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
    return value


def _nullable_string(value: Any, path: str, *, safe_id: bool = False) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be null or a nonempty string")
    if safe_id and _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be 1..128 filename-safe ASCII characters")
    return value


def _nullable_sequence(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be null or a positive integer")
    return value


def _evidence(value: Any, path: str) -> Evidence | None:
    if value is None:
        return None
    obj = _object(value, path, {"source", "reason"}, {"source", "reason"})
    source = _nullable_string(obj["source"], f"{path}.source")
    reason = _nullable_string(obj["reason"], f"{path}.reason")
    if source is None or reason is None:
        raise ValueError(f"{path}.source and {path}.reason must be nonempty strings")
    return Evidence(source, reason)


def _resource_expectation(value: Any, index: int) -> ExpectedResourceDelta:
    path = f"expected_resource_deltas[{index}]"
    obj = _object(value, path, {"resource_id", "level", "expected_delta"},
                  {"resource_id", "level", "expected_delta"})
    resource_id = _nullable_string(obj["resource_id"], f"{path}.resource_id")
    if resource_id is None:
        raise ValueError(f"{path}.resource_id must be a nonempty string")
    level = obj["level"]
    if level is not None and (type(level) is not int or level < 0):
        raise ValueError(f"{path}.level must be null or a nonnegative integer")
    delta = obj["expected_delta"]
    if type(delta) not in (int, float) or not isfinite(delta):
        raise ValueError(f"{path}.expected_delta must be a finite JSON number")
    return ExpectedResourceDelta(resource_id, level, float(delta))


def parse_trial(value: Any) -> Trial:
    """Strictly parse a v1 trial. Null facts are valid but remain inconclusive."""
    fields = {
        "schema_version", "trial_id", "kind", "capture_origin", "actor_entity_id",
        "target_entity_id", "ability_id", "combat_id", "before_sequence", "after_sequence",
        "story_action_id", "command_id", "expected_resource_deltas", "completion_evidence",
        "rejection_evidence", "game_version", "script_extender_version", "notes",
    }
    obj = _object(value, "trial", fields, fields)
    if type(obj["schema_version"]) is not int or obj["schema_version"] != TRIAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trial schema_version: {obj['schema_version']!r}")
    try:
        kind = TrialKind(obj["kind"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"kind has unknown value {obj['kind']!r}") from exc
    try:
        origin = CaptureOrigin(obj["capture_origin"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"capture_origin has unknown value {obj['capture_origin']!r}") from exc
    before = _nullable_sequence(obj["before_sequence"], "before_sequence")
    after = _nullable_sequence(obj["after_sequence"], "after_sequence")
    if before is not None and after is not None and before >= after:
        raise ValueError("before_sequence must be less than after_sequence")
    story = obj["story_action_id"]
    if story is not None and (type(story) is not int or story < 0):
        raise ValueError("story_action_id must be null or a nonnegative integer (zero is valid)")
    if not isinstance(obj["expected_resource_deltas"], list):
        raise ValueError("expected_resource_deltas must be a JSON array")
    deltas = tuple(_resource_expectation(item, index) for index, item in enumerate(
        obj["expected_resource_deltas"]
    ))
    keys = [(item.resource_id, item.level) for item in deltas]
    if len(keys) != len(set(keys)):
        raise ValueError("expected_resource_deltas contains duplicate resource_id/level keys")
    return Trial(
        TRIAL_SCHEMA_VERSION,
        _nullable_string(obj["trial_id"], "trial_id", safe_id=True), kind, origin,
        _nullable_string(obj["actor_entity_id"], "actor_entity_id"),
        _nullable_string(obj["target_entity_id"], "target_entity_id"),
        _nullable_string(obj["ability_id"], "ability_id"),
        _nullable_string(obj["combat_id"], "combat_id"), before, after, story,
        _nullable_string(obj["command_id"], "command_id", safe_id=True), deltas,
        _evidence(obj["completion_evidence"], "completion_evidence"),
        _evidence(obj["rejection_evidence"], "rejection_evidence"),
        _nullable_string(obj["game_version"], "game_version"),
        _nullable_string(obj["script_extender_version"], "script_extender_version"),
        _nullable_string(obj["notes"], "notes"),
    )


def trial_template(kind: TrialKind) -> dict[str, Any]:
    """Return a reviewable template without inventing game facts."""
    return {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "trial_id": None,
        "kind": kind.value,
        "capture_origin": CaptureOrigin.UNKNOWN.value,
        "actor_entity_id": None,
        "target_entity_id": None,
        "ability_id": None,
        "combat_id": None,
        "before_sequence": None,
        "after_sequence": None,
        "story_action_id": None,
        "command_id": None,
        "expected_resource_deltas": [],
        "completion_evidence": None,
        "rejection_evidence": None,
        "game_version": None,
        "script_extender_version": None,
        "notes": None,
    }


def init_trial(path: str | Path, kind: TrialKind) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(trial_template(kind), stream, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"Refusing to replace existing trial template: {destination}") from exc
    return destination


def _missing_trial_facts(trial: Trial) -> tuple[str, ...]:
    required = ("trial_id", "actor_entity_id", "target_entity_id", "ability_id",
                "before_sequence", "after_sequence")
    missing = [name for name in required if getattr(trial, name) is None]
    if trial.kind is TrialKind.AUTONOMOUS_ATTACK and trial.command_id is None:
        missing.append("command_id")
    return tuple(missing)


def _find_entity(snapshot: GameSnapshot, entity_id: str) -> EntitySnapshot | None:
    matches = [entity for entity in snapshot.entities if entity.entity_id == entity_id]
    if len(matches) > 1:
        raise ValueError(f"Snapshot {snapshot.sequence} has duplicate entity_id {entity_id!r}")
    return matches[0] if matches else None


def _resource_map(entity: EntitySnapshot) -> dict[tuple[str, int | None], ResourceValue]:
    result: dict[tuple[str, int | None], ResourceValue] = {}
    for resource in entity.resources:
        key = (resource.resource_id, resource.level)
        if key in result:
            raise ValueError(f"Entity {entity.entity_id!r} has duplicate resource_id/level {key!r}")
        result[key] = resource
    return result


def _action_groups(events: Sequence[GameEvent]) -> list[tuple[GameEvent, ...]]:
    """Collapse correlated generic/targeted events while preserving missing-ID ambiguity."""
    groups: list[list[GameEvent]] = []
    by_story: dict[int, list[GameEvent]] = {}
    for event in events:
        if event.kind not in _ACTION_KINDS:
            continue
        if event.story_action_id is None:
            groups.append([event])
        elif event.story_action_id in by_story:
            by_story[event.story_action_id].append(event)
        else:
            group = [event]
            by_story[event.story_action_id] = group
            groups.append(group)
    return [tuple(group) for group in groups]


def _check_actions(trial: Trial, events: Sequence[GameEvent]) -> Check:
    groups = _action_groups(events)
    actor_groups = [group for group in groups if any(event.actor_id == trial.actor_entity_id for event in group)]
    wrong_actor = [event for group in groups for event in group if (
        event.actor_id is not None and event.actor_id != trial.actor_entity_id and event.ability_id == trial.ability_id
        # Ability IDs and targets can be shared by unrelated creatures. Only a
        # declared correlation ID ties another actor's event to this trial.
        and trial.story_action_id is not None
        and event.story_action_id == trial.story_action_id
        and event.target_id == trial.target_entity_id
    )]
    if wrong_actor:
        return Check("action", Status.FAIL, "Expected ability/target was attempted by the wrong actor",
                     tuple(event.sequence for event in wrong_actor))
    if trial.kind is TrialKind.RESOURCE_REJECTION:
        if actor_groups:
            sequences = tuple(event.sequence for group in actor_groups for event in group)
            return Check("action", Status.FAIL, "Controlled actor produced attempted-action evidence during a rejection trial", sequences)
        contradictory_outcomes = [event for event in events if (
            event.kind in (EventKind.DAMAGE, EventKind.MISS, EventKind.CRITICAL_HIT)
            and event.actor_id == trial.actor_entity_id
            and (event.ability_id is None or event.ability_id == trial.ability_id)
            and (event.target_id is None or event.target_id == trial.target_entity_id)
        )]
        if contradictory_outcomes:
            return Check("action", Status.FAIL,
                         "Attack outcome evidence contradicts the reported rejection",
                         tuple(event.sequence for event in contradictory_outcomes))
        return Check("action", Status.PASS, "No contradictory attempted-action event was observed in the bounded window")
    if not actor_groups:
        return Check("action", Status.INCONCLUSIVE, "No attempted-action event for the expected actor was observed")
    if len(actor_groups) != 1:
        sequences = tuple(event.sequence for group in actor_groups for event in group)
        return Check("action", Status.FAIL, "Multiple distinct controlled actions occurred in the bounded window", sequences)
    group = actor_groups[0]
    kinds = {event.kind for event in group}
    if EventKind.ABILITY_USED_ON_TARGET in kinds and EventKind.ABILITY_USED_AT_POSITION in kinds:
        return Check("action", Status.FAIL,
                     "Correlated target and position action events conflict",
                     tuple(event.sequence for event in group))
    abilities = {event.ability_id for event in group if event.ability_id is not None}
    targets = {event.target_id for event in group if event.target_id is not None}
    combats = {event.combat_id for event in group if event.combat_id is not None}
    stories = {event.story_action_id for event in group if event.story_action_id is not None}
    sequences = tuple(event.sequence for event in group)
    if abilities and abilities != {trial.ability_id}:
        return Check("action", Status.FAIL, f"Observed ability IDs {sorted(abilities)!r} do not match {trial.ability_id!r}", sequences)
    if not abilities:
        return Check("action", Status.INCONCLUSIVE, "Attempted-action evidence lacks an ability_id", sequences)
    if targets and targets != {trial.target_entity_id}:
        return Check("action", Status.FAIL, f"Observed target IDs {sorted(targets)!r} do not match {trial.target_entity_id!r}", sequences)
    if not targets:
        return Check("action", Status.INCONCLUSIVE, "Attempted-action evidence lacks a target_id", sequences)
    if trial.combat_id is not None and combats and combats != {trial.combat_id}:
        return Check("action", Status.FAIL, f"Observed combat IDs {sorted(combats)!r} do not match {trial.combat_id!r}", sequences)
    if trial.story_action_id is not None and stories != {trial.story_action_id}:
        return Check("action", Status.FAIL, f"Observed StoryActionIDs {sorted(stories)!r} do not match {trial.story_action_id!r}", sequences)
    return Check("action", Status.PASS, "Exactly one matching attempted action was observed", sequences)


def _turn_boundary(window: Sequence[Record], before: GameSnapshot, after: GameSnapshot) -> Check:
    events = [record for record in window if isinstance(record, GameEvent) and record.kind in _TURN_KINDS]
    snapshots = [record for record in window if isinstance(record, GameSnapshot)]
    state_changed = False
    unknown_sequences: list[int] = []
    first_combat = before.combat
    for snapshot in snapshots:
        if (snapshot.combat is None or snapshot.combat.round_number is None
                or snapshot.combat.active_entity_id is None):
            unknown_sequences.append(snapshot.sequence)
        if ((first_combat is None) != (snapshot.combat is None)) or (
            first_combat is not None and snapshot.combat is not None and
            (snapshot.combat.combat_id != first_combat.combat_id or
             snapshot.combat.round_number != first_combat.round_number or
             snapshot.combat.active_entity_id != first_combat.active_entity_id)
        ):
            state_changed = True
    if events or state_changed:
        sequences = tuple(record.sequence for record in events)
        return Check("turn_boundary", Status.INCONCLUSIVE,
                     "A turn/round boundary in the window can confound resource-cost attribution", sequences)
    if unknown_sequences:
        return Check("turn_boundary", Status.INCONCLUSIVE,
                     "Snapshot turn identity or round is unknown, so resource refresh cannot be ruled out",
                     tuple(unknown_sequences))
    return Check("turn_boundary", Status.PASS, "No turn or round boundary was observed in the bounded window")


def _resource_check(trial: Trial, before: GameSnapshot, after: GameSnapshot,
                    turn_check: Check) -> tuple[Check, list[dict[str, Any]]]:
    if not trial.expected_resource_deltas:
        return Check("resources", Status.INCONCLUSIVE,
                     "No measured-control resource expectations were declared"), []
    assert trial.actor_entity_id is not None
    before_actor = _find_entity(before, trial.actor_entity_id)
    after_actor = _find_entity(after, trial.actor_entity_id)
    if before_actor is None or after_actor is None:
        return Check("resources", Status.INCONCLUSIVE,
                     "Actor is missing from the before or after snapshot",
                     tuple(snapshot.sequence for snapshot, entity in ((before, before_actor), (after, after_actor)) if entity is None)), []
    before_values, after_values = _resource_map(before_actor), _resource_map(after_actor)
    observed: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatches: list[str] = []
    for expected in trial.expected_resource_deltas:
        key = (expected.resource_id, expected.level)
        left, right = before_values.get(key), after_values.get(key)
        item: dict[str, Any] = {
            "resource_id": expected.resource_id, "level": expected.level,
            "expected_delta": expected.expected_delta,
            "before_amount": None if left is None else left.amount,
            "after_amount": None if right is None else right.amount,
            "observed_delta": None,
        }
        if left is None or right is None:
            missing.append(f"{expected.resource_id!r} level {expected.level!r}")
        else:
            item["observed_delta"] = right.amount - left.amount
            if not isclose(item["observed_delta"], expected.expected_delta, rel_tol=0, abs_tol=1e-6):
                mismatches.append(
                    f"{expected.resource_id!r} level {expected.level!r}: "
                    f"expected {expected.expected_delta}, observed {item['observed_delta']}"
                )
        observed.append(item)
    sequences = (before.sequence, after.sequence)
    if missing:
        return Check("resources", Status.INCONCLUSIVE,
                     f"Resource missing from before or after snapshot: {', '.join(missing)}", sequences), observed
    if turn_check.status is Status.INCONCLUSIVE:
        detail = "Resource delta cannot be attributed across the observed turn/round boundary"
        if mismatches:
            detail += f"; raw comparison also differs ({'; '.join(mismatches)})"
        return Check("resources", Status.INCONCLUSIVE, detail, sequences), observed
    if mismatches:
        return Check("resources", Status.FAIL, "Resource delta mismatch: " + "; ".join(mismatches), sequences), observed
    return Check("resources", Status.PASS, "All declared measured-control resource deltas match", sequences), observed


def _read_protocol(session_dir: Path) -> tuple[dict[str, ActionCommand],
                                                dict[str, tuple[CommandAcknowledgement, ...]],
                                                dict[str, PostExecutionObservation]]:
    if not session_dir.is_dir():
        raise ValueError(f"Session directory does not exist: {session_dir}")
    missing = [name for name in ("records", "commands", "acknowledgements")
               if not (session_dir / name).is_dir()]
    if missing:
        raise ValueError(f"Session directory is missing protocol subdirectory/directories: {', '.join(missing)}")
    session = CommandSession(session_dir)
    commands = session.read_commands()
    feedback = session.read_feedback()
    command_map = {command.command_id: command for command in commands}
    histories: dict[str, list[CommandAcknowledgement]] = {}
    confirmations: dict[str, PostExecutionObservation] = {}
    for item in feedback:
        if isinstance(item, CommandAcknowledgement):
            histories.setdefault(item.command_id, []).append(item)
        if isinstance(item, PostExecutionObservation):
            if item.command_id in confirmations:
                raise ValueError(f"Duplicate post-execution observation for {item.command_id!r}")
            confirmations[item.command_id] = item
    return command_map, {key: tuple(value) for key, value in histories.items()}, confirmations


def _command_checks(trial: Trial, records: Sequence[Record], records_dir: Path,
                    session_dir: Path | None) -> list[Check]:
    if trial.kind is not TrialKind.AUTONOMOUS_ATTACK:
        return []
    if session_dir is None:
        return [Check("command_intent", Status.INCONCLUSIVE, "No session directory was supplied"),
                Check("completion", Status.INCONCLUSIVE, "Autonomous completion requires command protocol evidence")]
    if records_dir.resolve() != (session_dir / "records").resolve():
        raise ValueError("--records must be the records subdirectory of --session-dir for autonomous analysis")
    assert trial.command_id is not None and trial.actor_entity_id is not None
    commands, histories, confirmations = _read_protocol(session_dir)
    command = commands.get(trial.command_id)
    if command is None:
        return [Check("command_intent", Status.INCONCLUSIVE, "The explicit command_id was not found"),
                Check("completion", Status.INCONCLUSIVE, "No linked command can prove completion")]
    mismatches: list[str] = []
    if command.action_type is not ActionType.USE_ABILITY:
        mismatches.append("action_type")
    if command.actor_entity_id != trial.actor_entity_id:
        mismatches.append("actor_entity_id")
    if command.ability_id != trial.ability_id:
        mismatches.append("ability_id")
    if command.target_kind is not TargetKind.ENTITY or command.target_entity_id != trial.target_entity_id:
        mismatches.append("target")
    if trial.combat_id is not None and command.expected_combat_id != trial.combat_id:
        mismatches.append("combat_id")
    if command.snapshot_sequence != trial.before_sequence:
        mismatches.append("snapshot_sequence")
    if mismatches:
        intent = Check("command_intent", Status.FAIL,
                       "Linked command does not match trial intent: " + ", ".join(mismatches))
    else:
        intent = Check("command_intent", Status.PASS, "Explicitly linked command matches trial intent")
    history = histories.get(trial.command_id, ())
    terminal = history[-1] if history else None
    if terminal is None or terminal.status is AcknowledgementStatus.ACCEPTED:
        completion = Check("completion", Status.INCONCLUSIVE,
                           "Command acceptance alone does not prove execution completed")
    elif terminal.status is not AcknowledgementStatus.RESOLVED:
        completion = Check("completion", Status.FAIL,
                           f"Linked command ended with {terminal.status.value!r}, not resolved")
    else:
        confirmation = confirmations.get(trial.command_id)
        snapshots = {record.sequence for record in records if isinstance(record, GameSnapshot)}
        boundary = terminal.observation_boundary_sequence
        if confirmation is None:
            completion = Check("completion", Status.INCONCLUSIVE,
                               "Resolved command lacks a post-execution observation attestation")
        elif boundary is None or confirmation.snapshot_sequence <= boundary:
            completion = Check("completion", Status.FAIL,
                               "Post-execution observation does not follow the resolved boundary",
                               (confirmation.snapshot_sequence,))
        elif confirmation.snapshot_sequence not in snapshots:
            completion = Check("completion", Status.FAIL,
                               "Post-execution observation does not reference a snapshot in this record collection",
                               (confirmation.snapshot_sequence,))
        elif trial.after_sequence is not None and confirmation.snapshot_sequence > trial.after_sequence:
            completion = Check("completion", Status.FAIL,
                               "Post-execution observation falls outside the explicit trial window",
                               (confirmation.snapshot_sequence,))
        else:
            completion = Check("completion", Status.PASS,
                               "Resolved command has a valid fresh post-execution snapshot confirmation",
                               (confirmation.snapshot_sequence,))
    return [intent, completion]


def analyze_trial(trial: Trial, records_dir: str | Path,
                  session_dir: str | Path | None = None,
                  *, trial_path: str | Path | None = None) -> dict[str, Any]:
    """Analyze an exact record collection without mutating or normalizing it."""
    records_path = Path(records_dir)
    records = RecordDirectoryReader(records_path).read_all()
    sequence_report = validate_sequence(records, start_sequence=1)
    checks: list[Check] = []
    if trial.capture_origin is CaptureOrigin.UNKNOWN:
        checks.append(Check("capture_origin", Status.INCONCLUSIVE,
                            "capture_origin is unknown; label the evidence synthetic or live_operator_reported"))
    else:
        checks.append(Check("capture_origin", Status.PASS,
                            f"Capture provenance is explicitly labeled {trial.capture_origin.value!r}"))
    missing = _missing_trial_facts(trial)
    if missing:
        checks.append(Check("trial_facts", Status.INCONCLUSIVE,
                            "Template facts still missing: " + ", ".join(missing)))
        return _report(trial, checks, [], records_path, session_dir, trial_path)
    checks.append(Check("trial_facts", Status.PASS, "Required trial facts are present"))
    assert trial.before_sequence is not None and trial.after_sequence is not None
    by_sequence = {record.sequence: record for record in records}
    before, after = by_sequence.get(trial.before_sequence), by_sequence.get(trial.after_sequence)
    if sequence_report.gaps:
        checks.append(Check("trace_window", Status.FAIL,
                            f"Observation collection has sequence gaps: {sequence_report.gaps}"))
        return _report(trial, checks, [], records_path, session_dir, trial_path)
    if not isinstance(before, GameSnapshot) or not isinstance(after, GameSnapshot):
        checks.append(Check("trace_window", Status.FAIL,
                            "before_sequence and after_sequence must reference snapshots in this collection",
                            (trial.before_sequence, trial.after_sequence)))
        return _report(trial, checks, [], records_path, session_dir, trial_path)
    window = tuple(record for record in records if trial.before_sequence <= record.sequence <= trial.after_sequence)
    checks.append(Check("trace_window", Status.PASS,
                        "Explicit ordered snapshot bounds exist in the contiguous collection",
                        (before.sequence, after.sequence)))
    controlled_unknown = [snapshot.sequence for snapshot in (before, after)
                          if snapshot.controlled_entity_id is None]
    controlled_mismatch = [snapshot.sequence for snapshot in (before, after)
                           if snapshot.controlled_entity_id is not None
                           and snapshot.controlled_entity_id != trial.actor_entity_id]
    if controlled_mismatch:
        checks.append(Check("controlled_actor", Status.FAIL,
                            "Boundary snapshot controlled_entity_id does not match the trial actor",
                            tuple(controlled_mismatch)))
    elif controlled_unknown:
        checks.append(Check("controlled_actor", Status.INCONCLUSIVE,
                            "Boundary snapshot controlled_entity_id is unknown",
                            tuple(controlled_unknown)))
    else:
        checks.append(Check("controlled_actor", Status.PASS,
                            "Boundary snapshots identify the trial actor as controlled",
                            (before.sequence, after.sequence)))
    if trial.combat_id is not None:
        mismatched = [snapshot.sequence for snapshot in (before, after) if (
            snapshot.combat is not None and snapshot.combat.combat_id is not None
            and snapshot.combat.combat_id != trial.combat_id
        )]
        unknown = [snapshot.sequence for snapshot in (before, after) if (
            snapshot.combat is None or snapshot.combat.combat_id is None
        )]
        if mismatched:
            checks.append(Check("combat", Status.FAIL, "Boundary snapshot combat_id does not match the trial",
                                tuple(mismatched)))
        elif unknown:
            checks.append(Check("combat", Status.INCONCLUSIVE,
                                "Boundary snapshot combat_id is unknown",
                                tuple(unknown)))
        else:
            checks.append(Check("combat", Status.PASS, "No boundary snapshot contradicts the declared combat_id",
                                (before.sequence, after.sequence)))
    events = tuple(record for record in window if isinstance(record, GameEvent))
    checks.append(_check_actions(trial, events))
    turn = _turn_boundary(window, before, after)
    checks.append(turn)
    resource_check, observed = _resource_check(trial, before, after, turn)
    checks.append(resource_check)
    if trial.kind is TrialKind.HUMAN_ATTACK:
        if trial.rejection_evidence is not None:
            checks.append(Check("completion", Status.FAIL,
                                "Rejection evidence contradicts an attack-completion trial"))
        elif trial.completion_evidence is None:
            checks.append(Check("completion", Status.INCONCLUSIVE,
                                "Human attack requires explicit operator completion evidence"))
        else:
            checks.append(Check("completion", Status.PASS,
                                f"Explicit completion evidence: {trial.completion_evidence.source}: "
                                f"{trial.completion_evidence.reason}"))
    elif trial.kind is TrialKind.AUTONOMOUS_ATTACK:
        checks.extend(_command_checks(trial, records, records_path,
                                      None if session_dir is None else Path(session_dir)))
        if trial.rejection_evidence is not None:
            checks.append(Check("declared_outcome", Status.FAIL,
                                "Rejection evidence contradicts an attack-completion trial"))
    else:
        contradictory = trial.completion_evidence is not None
        if contradictory:
            checks.append(Check("rejection", Status.FAIL,
                                "Completion evidence contradicts a resource-rejection trial"))
        elif trial.rejection_evidence is None:
            checks.append(Check("rejection", Status.INCONCLUSIVE,
                                "Absence of events is not rejection proof; explicit rejection evidence is required"))
        else:
            checks.append(Check("rejection", Status.PASS,
                                f"Explicit rejection evidence: {trial.rejection_evidence.source}: "
                                f"{trial.rejection_evidence.reason}"))
    return _report(trial, checks, observed, records_path, session_dir, trial_path)


def _report(trial: Trial, checks: Sequence[Check], observed: Sequence[dict[str, Any]],
            records_dir: Path, session_dir: str | Path | None,
            trial_path: str | Path | None) -> dict[str, Any]:
    statuses = {check.status for check in checks}
    status = Status.FAIL if Status.FAIL in statuses else (
        Status.INCONCLUSIVE if Status.INCONCLUSIVE in statuses else Status.PASS
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "trial_id": trial.trial_id,
        "kind": trial.kind.value,
        "status": status.value,
        "exit_code": {Status.PASS: 0, Status.FAIL: 1, Status.INCONCLUSIVE: 2}[status],
        "checks": [check.to_dict() for check in checks],
        "observed_resource_deltas": list(observed),
        "provenance": {
            "capture_origin": trial.capture_origin.value,
            "trial_path": None if trial_path is None else str(Path(trial_path)),
            "records_directory": str(records_dir),
            "session_directory": None if session_dir is None else str(Path(session_dir)),
            "before_sequence": trial.before_sequence,
            "after_sequence": trial.after_sequence,
            "game_version": trial.game_version,
            "script_extender_version": trial.script_extender_version,
        },
        "live_verified": False,
    }


def analyze_file(trial_path: str | Path, records_dir: str | Path,
                 session_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(trial_path)
    return analyze_trial(parse_trial(_load_json(path)), records_dir, session_dir, trial_path=path)


def _write_report(report: Mapping[str, Any], output: str | Path | None) -> None:
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        Path(output).write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _validate_output_destination(output: Path | None, trial_path: Path,
                                 records_dir: Path, session_dir: Path | None) -> None:
    """Prevent a report from replacing or entering any supplied evidence source."""
    if output is None:
        return
    destination = output.resolve()
    if destination == trial_path.resolve():
        raise ValueError("--output must not replace the source trial JSON")
    evidence_directories = [records_dir.resolve()]
    if session_dir is not None:
        evidence_directories.extend((
            (session_dir / "records").resolve(),
            (session_dir / "commands").resolve(),
            (session_dir / "acknowledgements").resolve(),
        ))
    if any(_is_within(destination, directory) for directory in evidence_directories):
        raise ValueError("--output must be outside raw records, commands, and acknowledgements directories")
    if output.exists() and output.is_dir():
        raise ValueError("--output must be a file path, not a directory")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(f"argument error: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="create a v1 trial template without replacing files")
    init.add_argument("path", type=Path)
    init.add_argument("--kind", required=True, choices=[kind.value for kind in TrialKind])
    analyze = subparsers.add_parser("analyze", help="analyze one bounded offline trial")
    analyze.add_argument("trial", type=Path)
    analyze.add_argument("--records", required=True, type=Path, help="one session's immutable records directory")
    analyze.add_argument("--session-dir", type=Path, help="required for autonomous trials; --records must be its records subdirectory")
    analyze.add_argument("--output", type=Path, help="report path outside the trial and raw evidence; JSON also prints to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _parser()
        args = parser.parse_args(argv)
        if args.command == "init":
            path = init_trial(args.path, TrialKind(args.kind))
            print(json.dumps({"status": "CREATED", "path": str(path), "schema_version": 1}))
            return 0
        _validate_output_destination(args.output, args.trial, args.records, args.session_dir)
        report = analyze_file(args.trial, args.records, args.session_dir)
        _write_report(report, args.output)
        return int(report["exit_code"])
    except (OSError, ValueError, TypeError) as exc:
        error = {"schema_version": REPORT_SCHEMA_VERSION, "status": "ERROR", "exit_code": 2,
                 "error_type": "usage_or_validation_error", "reason": str(exc), "live_verified": False}
        sys.stderr.write(json.dumps(error, indent=2) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
