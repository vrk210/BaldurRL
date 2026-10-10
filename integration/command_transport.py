"""Small immutable-file session bridge. No exactly-once crash guarantee."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, TypeVar

from .commands import (
    ActionCommand, AcknowledgementStatus, CommandAcknowledgement,
    PostExecutionObservation, parse_post_execution_observation,
    parse_acknowledgement, parse_command, validate_command,
)
from .reader import RecordDirectoryReader, validate_sequence
from .serialization import Record
from .schema import GameSnapshot


T = TypeVar("T")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant {value}")


def _read(path: Path, parser: Callable[[Any], T]) -> T:
    try:
        return parser(json.loads(path.read_text(encoding="utf-8"),
                                 object_pairs_hook=_unique_object, parse_constant=_invalid_constant))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid protocol record in {path}: {exc}") from exc


def _publish(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON atomically without replacing an existing record.

    The temp file is on the destination filesystem. Hard linking exposes only
    the fully written file and fails if the final name already exists.
    """
    data = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"Duplicate publication: {path}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


class CommandSession:
    """One collector session, one cooperating publisher, one executor.

    Pending and accepted commands hold the slot. UNCERTAIN also holds it until
    external reconciliation; this minimal protocol has no automatic retry.
    Keep immutable files for the lifetime of a session.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.records = self.directory / "records"
        self.commands = self.directory / "commands"
        self.acknowledgements = self.directory / "acknowledgements"
        for path in (self.records, self.commands, self.acknowledgements):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        lock = self.directory / ".writer-lock"
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"Session writer busy or interrupted; inspect stale lock before recovery: {lock}") from exc
        try:
            yield
        finally:
            lock.rmdir()

    def _observation_records(self) -> tuple[Record, ...]:
        # Use the existing reader unchanged. A gap or a trailing event prevents
        # issuing decisions based on facts that may already have changed.
        reader = RecordDirectoryReader(self.records)
        records = reader.read_all()
        report = validate_sequence(records, start_sequence=reader.start_sequence)
        if report.gaps:
            raise ValueError(f"Observation sequence gaps: {report.gaps}")
        return records

    def latest_snapshot(self) -> GameSnapshot:
        return self._latest_snapshot(self._observation_records())

    @staticmethod
    def _latest_snapshot(records: tuple[Record, ...]) -> GameSnapshot:
        if not records or not isinstance(records[-1], GameSnapshot):
            raise ValueError("A fresh snapshot is required after the latest event (or empty observation stream)")
        return records[-1]

    def read_commands(self) -> tuple[ActionCommand, ...]:
        commands: dict[str, ActionCommand] = {}
        paths: dict[str, Path] = {}
        for path in sorted(self.commands.glob("*.json")):
            command = _read(path, parse_command)
            if command.command_id in commands:
                raise ValueError(f"Duplicate command_id {command.command_id!r} in {paths[command.command_id]} and {path}")
            commands[command.command_id] = command
            paths[command.command_id] = path
        return tuple(commands.values())

    def _read_feedback(self) -> dict[Path, CommandAcknowledgement | PostExecutionObservation]:
        def parse(value: Any) -> CommandAcknowledgement | PostExecutionObservation:
            if isinstance(value, dict) and value.get("record_type") == "post_execution_observation":
                return parse_post_execution_observation(value)
            return parse_acknowledgement(value)

        return {path: _read(path, parse) for path in sorted(self.acknowledgements.glob("*.json"))}

    def read_acknowledgements(self) -> tuple[CommandAcknowledgement, ...]:
        commands = {command.command_id: command for command in self.read_commands()}
        by_id: dict[str, dict[int, CommandAcknowledgement]] = {}
        for path, acknowledgement in self._read_feedback().items():
            if isinstance(acknowledgement, PostExecutionObservation):
                continue
            if acknowledgement.command_id not in commands:
                raise ValueError(f"Acknowledgement for unknown command_id {acknowledgement.command_id!r} in {path}")
            if (acknowledgement.observation_boundary_sequence is not None
                    and acknowledgement.observation_boundary_sequence < commands[acknowledgement.command_id].snapshot_sequence):
                raise ValueError("observation_boundary_sequence precedes the command's decision snapshot")
            history = by_id.setdefault(acknowledgement.command_id, {})
            sequence = acknowledgement.acknowledgement_sequence
            if sequence in history:
                raise ValueError(f"Duplicate acknowledgement sequence {sequence} for {acknowledgement.command_id!r} in {path}")
            history[sequence] = acknowledgement
        result: list[CommandAcknowledgement] = []
        for command_id, history in sorted(by_id.items()):
            if 2 in history and (1 not in history or history[1].status is not AcknowledgementStatus.ACCEPTED):
                raise ValueError(f"Result for {command_id!r} requires a preceding accepted acknowledgement")
            result.extend(history[key] for key in sorted(history))
        return tuple(result)

    def in_flight(self) -> tuple[ActionCommand, ...]:
        latest = {ack.command_id: ack for ack in self.read_acknowledgements()}
        return tuple(command for command in self.read_commands() if (
            command.command_id not in latest or not latest[command.command_id].releases_in_flight
        ))

    def publish_command(self, command: ActionCommand) -> Path:
        # Validate even when callers construct dataclasses rather than parse JSON.
        command = parse_command(command.to_dict())
        with self.writer_lock():
            if any(existing.command_id == command.command_id for existing in self.read_commands()):
                raise ValueError(f"Duplicate command_id {command.command_id!r}; never republish or retry automatically")
            if self.in_flight():
                raise ValueError("At most one in-flight command is allowed")
            self.validate_command_for_execution(command)
            path = self.commands / f"{command.command_id}.json"
            _publish(path, command.to_dict())
            return path

    def publish_acknowledgement(self, acknowledgement: CommandAcknowledgement) -> Path:
        """Publish feedback, filling a missing resolved/failed observation boundary.

        The returned file contains the persisted boundary; callers must use that
        record rather than infer it from the acknowledgement supplied here.
        """
        acknowledgement = parse_acknowledgement(acknowledgement.to_dict())
        with self.writer_lock():
            return self._publish_acknowledgement(acknowledgement)

    def _publish_acknowledgement(self, acknowledgement: CommandAcknowledgement) -> Path:
        """Caller holds writer_lock; also used for atomic mock receive checks."""
        commands = {command.command_id: command for command in self.read_commands()}
        if acknowledgement.command_id not in commands:
            raise ValueError("Acknowledgement command_id does not match a published command")
        history = [ack for ack in self.read_acknowledgements() if ack.command_id == acknowledgement.command_id]
        if acknowledgement.acknowledgement_sequence == 1:
            if history:
                raise ValueError("Duplicate acknowledgement or invalid transition")
        elif len(history) != 1 or history[0].status is not AcknowledgementStatus.ACCEPTED:
            raise ValueError("Execution result requires accepted acknowledgement; duplicate or invalid transition")
        if acknowledgement.status in (AcknowledgementStatus.RESOLVED, AcknowledgementStatus.FAILED):
            # Persist the completion boundary, including snapshots/events already
            # visible at result publication. It cannot itself prove capture timing.
            records = RecordDirectoryReader(self.records).read_all()
            boundary = max([commands[acknowledgement.command_id].snapshot_sequence,
                            *(record.sequence for record in records)])
            if acknowledgement.observation_boundary_sequence is None:
                acknowledgement = replace(acknowledgement, observation_boundary_sequence=boundary)
            elif acknowledgement.observation_boundary_sequence < boundary:
                raise ValueError("observation_boundary_sequence precedes observations already published at completion")
        path = self.acknowledgements / f"{acknowledgement.command_id}.{acknowledgement.acknowledgement_sequence}.json"
        _publish(path, acknowledgement.to_dict())
        return path

    @staticmethod
    def _validate_post_execution_observation(
        confirmation: PostExecutionObservation,
        results: dict[str, CommandAcknowledgement],
        records: tuple[Record, ...],
    ) -> None:
        result = results.get(confirmation.command_id)
        if result is None or result.status not in (AcknowledgementStatus.RESOLVED, AcknowledgementStatus.FAILED):
            raise ValueError("post-execution observation requires a resolved or failed command result")
        boundary = result.observation_boundary_sequence
        if boundary is None or confirmation.snapshot_sequence <= boundary:
            raise ValueError("post-execution observation must follow the persisted completion observation boundary")
        if not any(isinstance(record, GameSnapshot) and record.sequence == confirmation.snapshot_sequence for record in records):
            raise ValueError("post-execution observation must reference an existing snapshot in the contiguous observation stream")

    def publish_post_execution_observation(self, confirmation: PostExecutionObservation) -> Path:
        """Persist a trusted collector's capture-after-completion attestation.

        The caller must initiate capture AFTER observing the terminal result;
        this method cannot infer capture timing from JSON publication timing.
        """
        confirmation = parse_post_execution_observation(confirmation.to_dict())
        with self.writer_lock():
            results = {ack.command_id: ack for ack in self.read_acknowledgements()}
            records = self._observation_records()
            self._validate_post_execution_observation(confirmation, results, records)
            if any(isinstance(record, PostExecutionObservation) and record.command_id == confirmation.command_id
                   for record in self._read_feedback().values()):
                raise ValueError("Duplicate post-execution observation confirmation")
            path = self.acknowledgements / f"{confirmation.command_id}.observation.json"
            _publish(path, confirmation.to_dict())
            return path

    def validate_command_for_execution(self, command: ActionCommand) -> None:
        """Shared publication/receipt gate: completion does not refresh facts."""
        records = self._observation_records()
        snapshot = self._latest_snapshot(records)
        results = {ack.command_id: ack for ack in self.read_acknowledgements()}
        confirmations: dict[str, PostExecutionObservation] = {}
        for record in self._read_feedback().values():
            if isinstance(record, PostExecutionObservation):
                if record.command_id in confirmations:
                    raise ValueError(f"Duplicate post-execution observation for {record.command_id!r}")
                self._validate_post_execution_observation(record, results, records)
                confirmations[record.command_id] = record
        for result in results.values():
            if result.status in (AcknowledgementStatus.RESOLVED, AcknowledgementStatus.FAILED):
                confirmation = confirmations.get(result.command_id)
                if confirmation is None or command.snapshot_sequence < confirmation.snapshot_sequence:
                    raise ValueError(f"Fresh confirmed post-execution observation required after {result.command_id!r}")
        validate_command(command, snapshot)
