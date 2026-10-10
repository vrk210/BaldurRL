"""Synthetic command handshake only: no BG3 rules, effects or real execution."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

from .command_transport import CommandSession, _publish, _read
from .commands import (
    COMMAND_SCHEMA_VERSION, ActionCommand, ActionType, AcknowledgementStatus,
    CommandAcknowledgement, PostExecutionObservation, parse_acknowledgement,
)
from .schema import (
    CombatSnapshot, DecisionCandidate, EntitySnapshot, GameSnapshot,
    Legality, TargetKind,
)


class MockExecutor:
    def __init__(self, session: CommandSession) -> None:
        self.session = session

    def receive(self) -> tuple[CommandAcknowledgement, ...]:
        """Revalidate on receive; repeat polls never re-accept historical IDs."""
        with self.session.writer_lock():
            commands = self.session.read_commands()
            acknowledged = {ack.command_id for ack in self.session.read_acknowledgements()}
            pending = [command for command in commands if command.command_id not in acknowledged]
            in_flight = self.session.in_flight()
            results: list[CommandAcknowledgement] = []
            for command in pending:
                reason = None
                try:
                    if len(in_flight) > 1:
                        raise ValueError("At most one in-flight command is allowed; conflicting inbox commands rejected")
                    self.session.validate_command_for_execution(command)
                except ValueError as exc:
                    reason = str(exc)
                ack = CommandAcknowledgement(
                    COMMAND_SCHEMA_VERSION, command.command_id, 1,
                    AcknowledgementStatus.ACCEPTED if reason is None else AcknowledgementStatus.REJECTED,
                    reason=reason,
                )
                self.session._publish_acknowledgement(ack)
                results.append(ack)
            return tuple(results)

    def finish(self, command_id: str, status: AcknowledgementStatus = AcknowledgementStatus.RESOLVED) -> CommandAcknowledgement:
        """Explicit fixture result, separate from acceptance; no state mutation.

        On restart, accepted-only histories are left pending by receive(). A
        caller must explicitly reconcile them; they are never auto-replayed.
        """
        if status not in (AcknowledgementStatus.RESOLVED, AcknowledgementStatus.FAILED, AcknowledgementStatus.UNCERTAIN):
            raise ValueError("Mock finish requires resolved, failed or uncertain")
        ack = CommandAcknowledgement(
            COMMAND_SCHEMA_VERSION, command_id, 2, status,
            reason=f"Synthetic fixture reports {status.value}; no BG3 execution or mechanics were simulated",
        )
        path = self.session.publish_acknowledgement(ack)
        return _read(path, parse_acknowledgement)


def synthetic_observation() -> GameSnapshot:
    return GameSnapshot(
        1, 1, controlled_entity_id="synthetic-player",
        combat=CombatSnapshot("synthetic-combat", 1, "synthetic-player"),
        entities=(EntitySnapshot("synthetic-player"), EntitySnapshot("synthetic-enemy")),
        candidate_decisions=(
            DecisionCandidate("synthetic-ability", TargetKind.ENTITY, Legality.LEGAL,
                              target_entity_id="synthetic-enemy", legality_source="synthetic verified fixture"),
            DecisionCandidate("synthetic-unknown", TargetKind.NONE, Legality.UNKNOWN),
        ),
    )


def run_demo(directory: str | Path) -> tuple[ActionCommand, tuple[CommandAcknowledgement, ...]]:
    session = CommandSession(directory)
    if any(path.name != ".writer-lock" for directory in (session.records, session.commands, session.acknowledgements)
           for path in directory.iterdir()):
        raise ValueError("Demo requires a fresh, empty session tree")
    snapshot = synthetic_observation()
    _publish(session.records / "0001.json", snapshot.to_dict())
    observed = session.latest_snapshot()
    candidate = next(candidate for candidate in observed.candidate_decisions if candidate.legality is Legality.LEGAL)
    command = ActionCommand(
        COMMAND_SCHEMA_VERSION, str(uuid4()), observed.sequence, observed.controlled_entity_id,
        ActionType.USE_ABILITY, candidate.target_kind,
        expected_combat_id=observed.combat.combat_id, ability_id=candidate.ability_id,
        target_entity_id=candidate.target_entity_id, target_position=candidate.target_position,
    )
    print("observation:", json.dumps(observed.to_dict()))
    session.publish_command(command)
    print("selected command:", json.dumps(command.to_dict()))
    executor = MockExecutor(session)
    accepted = executor.receive()
    for acknowledgement in accepted:
        print("mock acknowledgement:", json.dumps(acknowledgement.to_dict()))
    resolved = executor.finish(command.command_id)
    print("mock result:", json.dumps(resolved.to_dict()))
    # Initiate a separate synthetic capture only after observing the result.
    # State stays unchanged because this mock does not implement mechanics.
    post_observation = replace(synthetic_observation(), sequence=2)
    _publish(session.records / "0002.json", post_observation.to_dict())
    confirmation = PostExecutionObservation(
        COMMAND_SCHEMA_VERSION, command.command_id, post_observation.sequence,
        "Synthetic fixture capture initiated after observing the mock terminal result",
    )
    session.publish_post_execution_observation(confirmation)
    print("post-execution observation confirmation:", json.dumps(confirmation.to_dict()))
    return command, (*accepted, resolved)


def main() -> None:
    parser = ArgumentParser(description="Synthetic observation -> command -> mock acceptance -> fixture result (no BG3)")
    parser.add_argument("session_directory", type=Path)
    args = parser.parse_args()
    try:
        run_demo(args.session_directory)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
