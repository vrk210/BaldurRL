"""Model-independent semantic intent and execution acknowledgement wire types."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import re
from typing import Any, Mapping

from .schema import GameSnapshot, Legality, TargetKind, Vec3


COMMAND_SCHEMA_VERSION = 1


class ActionType(Enum):
    USE_ABILITY = "use_ability"
    END_TURN = "end_turn"


class AcknowledgementStatus(Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RESOLVED = "resolved"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


def _string(value: Any, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a nonempty string{' or null' if optional else ''}")


def _positive(value: Any, field: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _header(version: int, command_id: str) -> None:
    if type(version) is not int or version != COMMAND_SCHEMA_VERSION:
        raise ValueError(f"Unsupported command schema_version: {version!r}")
    if type(command_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", command_id):
        raise ValueError("command_id must be 1..128 filename-safe ASCII characters")


@dataclass(frozen=True)
class ActionCommand:
    schema_version: int
    command_id: str
    snapshot_sequence: int
    actor_entity_id: str
    action_type: ActionType
    target_kind: TargetKind
    expected_combat_id: str | None = None
    ability_id: str | None = None
    target_entity_id: str | None = None
    target_position: Vec3 | None = None

    def __post_init__(self) -> None:
        _header(self.schema_version, self.command_id)
        _positive(self.snapshot_sequence, "snapshot_sequence")
        _string(self.actor_entity_id, "actor_entity_id")
        for field in ("expected_combat_id", "ability_id", "target_entity_id"):
            _string(getattr(self, field), field, optional=True)
        if not isinstance(self.action_type, ActionType):
            raise ValueError("action_type must be an ActionType")
        if not isinstance(self.target_kind, TargetKind):
            raise ValueError("target_kind must be a TargetKind")
        if self.action_type is ActionType.USE_ABILITY and self.ability_id is None:
            raise ValueError("use_ability requires the original BG3 ability_id")
        if self.action_type is ActionType.END_TURN and (self.ability_id is not None or self.target_kind is not TargetKind.NONE):
            raise ValueError("end_turn requires ability_id=null and target_kind=none")
        if self.target_kind is TargetKind.ENTITY:
            if self.target_entity_id is None or self.target_position is not None:
                raise ValueError("entity target requires target_entity_id and no target_position")
        elif self.target_kind is TargetKind.POSITION:
            if not isinstance(self.target_position, Vec3) or self.target_entity_id is not None:
                raise ValueError("position target requires target_position and no target_entity_id")
        elif self.target_entity_id is not None or self.target_position is not None:
            raise ValueError("none/self targets must omit target_entity_id and target_position; self means actor")
        if self.target_position is not None:
            if any(type(v) not in (int, float) or not isfinite(v) for v in (
                self.target_position.x, self.target_position.y, self.target_position.z
            )):
                raise ValueError("target_position coordinates must be finite JSON numbers")

    def to_dict(self) -> dict[str, Any]:
        position = self.target_position
        return {
            "record_type": "command", "schema_version": self.schema_version,
            "command_id": self.command_id, "snapshot_sequence": self.snapshot_sequence,
            "expected_combat_id": self.expected_combat_id,
            "actor_entity_id": self.actor_entity_id, "action_type": self.action_type.value,
            "ability_id": self.ability_id, "target_kind": self.target_kind.value,
            "target_entity_id": self.target_entity_id,
            "target_position": None if position is None else {"x": position.x, "y": position.y, "z": position.z},
        }


@dataclass(frozen=True)
class CommandAcknowledgement:
    schema_version: int
    command_id: str
    acknowledgement_sequence: int
    status: AcknowledgementStatus
    reason: str | None = None
    observation_sequence: int | None = None

    def __post_init__(self) -> None:
        _header(self.schema_version, self.command_id)
        if not isinstance(self.status, AcknowledgementStatus):
            raise ValueError("status must be an AcknowledgementStatus")
        expected = 1 if self.status in (AcknowledgementStatus.ACCEPTED, AcknowledgementStatus.REJECTED) else 2
        if type(self.acknowledgement_sequence) is not int or self.acknowledgement_sequence != expected:
            raise ValueError(f"{self.status.value} requires acknowledgement_sequence={expected}")
        _string(self.reason, "reason", optional=self.status is AcknowledgementStatus.ACCEPTED)
        if self.observation_sequence is not None:
            _positive(self.observation_sequence, "observation_sequence")

    @property
    def confirmed_resolved(self) -> bool:
        """Acceptance is never evidence that execution completed."""
        return self.status is AcknowledgementStatus.RESOLVED

    @property
    def releases_in_flight(self) -> bool:
        # Uncertainty requires external reconciliation, not an automatic retry.
        return self.status in (AcknowledgementStatus.REJECTED, AcknowledgementStatus.RESOLVED, AcknowledgementStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "acknowledgement", "schema_version": self.schema_version,
            "command_id": self.command_id, "acknowledgement_sequence": self.acknowledgement_sequence,
            "status": self.status.value, "reason": self.reason,
            "observation_sequence": self.observation_sequence,
        }


def _object(value: Any, name: str, required: set[str], optional: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing:
        raise ValueError(f"{name} missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown field(s): {', '.join(sorted(unknown))}")
    if value.get("record_type") != name:
        raise ValueError(f"record_type must be {name!r}")
    return value


def _enum(value: Any, kind: type[Enum], field: str) -> Any:
    if type(value) is not str:
        raise ValueError(f"{field} must be a JSON string")
    try:
        return kind(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} has unknown value {value!r}") from exc


def parse_command(value: Any) -> ActionCommand:
    obj = _object(value, "command", {
        "record_type", "schema_version", "command_id", "snapshot_sequence",
        "actor_entity_id", "action_type", "target_kind",
    }, {"expected_combat_id", "ability_id", "target_entity_id", "target_position"})
    fields = dict(obj)
    fields.pop("record_type")
    fields["action_type"] = _enum(obj["action_type"], ActionType, "action_type")
    fields["target_kind"] = _enum(obj["target_kind"], TargetKind, "target_kind")
    position = obj.get("target_position")
    if position is not None:
        if not isinstance(position, Mapping) or set(position) != {"x", "y", "z"}:
            raise ValueError("target_position must contain exactly x, y, z")
        if any(type(v) not in (int, float) or not isfinite(v) for v in position.values()):
            raise ValueError("target_position coordinates must be finite JSON numbers")
        fields["target_position"] = Vec3(position["x"], position["y"], position["z"])
    return ActionCommand(**fields)


def parse_acknowledgement(value: Any) -> CommandAcknowledgement:
    obj = _object(value, "acknowledgement", {
        "record_type", "schema_version", "command_id", "acknowledgement_sequence", "status",
    }, {"reason", "observation_sequence"})
    fields = dict(obj)
    fields.pop("record_type")
    fields["status"] = _enum(obj["status"], AcknowledgementStatus, "status")
    return CommandAcknowledgement(**fields)


def validate_command(command: ActionCommand, snapshot: GameSnapshot) -> None:
    """Fail closed against current facts; never derive legality from resources."""
    if command.snapshot_sequence != snapshot.sequence:
        raise ValueError(f"Stale or future snapshot_sequence {command.snapshot_sequence}; current snapshot is {snapshot.sequence}")
    if snapshot.controlled_entity_id is None or command.actor_entity_id != snapshot.controlled_entity_id:
        raise ValueError("actor_entity_id does not match the currently controlled entity")
    combat = snapshot.combat
    current_combat_id = None if combat is None else combat.combat_id
    if command.expected_combat_id != current_combat_id:
        raise ValueError(f"expected_combat_id mismatch: expected {command.expected_combat_id!r}, observed {current_combat_id!r}")
    if combat is not None and combat.active_entity_id != command.actor_entity_id:
        raise ValueError("actor_entity_id is not the verified active combat actor")
    if command.action_type is ActionType.END_TURN:
        # The v1 observation candidate schema describes abilities only.
        if combat is None:
            raise ValueError("end_turn requires a verified active combat turn")
        return
    matches = [candidate for candidate in snapshot.candidate_decisions if (
        candidate.ability_id == command.ability_id
        and candidate.target_kind is command.target_kind
        and candidate.target_entity_id == command.target_entity_id
        and candidate.target_position == command.target_position
    )]
    if len(matches) != 1 or matches[0].legality is not Legality.LEGAL:
        raise ValueError("Command requires exactly one matching currently verified LEGAL candidate; missing, ambiguous, ILLEGAL or UNKNOWN evidence is insufficient")
