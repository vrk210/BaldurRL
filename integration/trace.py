"""Conservative snapshot interval reconstruction from raw ordered records."""

from dataclasses import dataclass
from typing import Iterable

from .schema import EventKind, GameEvent, GameSnapshot
from .serialization import Record


@dataclass(frozen=True)
class SnapshotInterval:
    before: GameSnapshot
    events: tuple[GameEvent, ...]
    after: GameSnapshot


class AmbiguousActionError(ValueError):
    """More than one plausible controlled-entity action occurred in an interval."""


_ACTION_KINDS = frozenset({EventKind.ABILITY_USED, EventKind.ABILITY_USED_ON_TARGET, EventKind.ABILITY_USED_AT_POSITION})


def reconstruct_intervals(records: Iterable[Record]) -> tuple[SnapshotInterval, ...]:
    """Use consecutive snapshots as boundaries; trailing events remain unclosed."""
    intervals: list[SnapshotInterval] = []
    before: GameSnapshot | None = None
    events: list[GameEvent] = []
    previous: int | None = None
    for record in records:
        if previous is not None and record.sequence != previous + 1:
            raise ValueError(f"Trace sequence gap or duplicate after {previous}: {record.sequence}")
        previous = record.sequence
        if isinstance(record, GameSnapshot):
            if before is not None:
                intervals.append(SnapshotInterval(before, tuple(events), record))
            before = record
            events = []
        elif before is None:
            raise ValueError(f"Event {record.sequence} precedes the first snapshot")
        else:
            events.append(record)
    return tuple(intervals)


def controlled_action(interval: SnapshotInterval, controlled_entity_id: str) -> GameEvent | None:
    """Select one correlated action event without changing the raw interval."""
    matches = tuple(event for event in interval.events if event.actor_id == controlled_entity_id and event.kind in _ACTION_KINDS)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    action_ids = {event.story_action_id for event in matches}
    if None in action_ids or len(action_ids) != 1:
        raise AmbiguousActionError(f"Multiple action events for {controlled_entity_id}: sequences {tuple(event.sequence for event in matches)}")
    by_kind = {kind: tuple(event for event in matches if event.kind is kind) for kind in _ACTION_KINDS}
    if by_kind[EventKind.ABILITY_USED_ON_TARGET] and by_kind[EventKind.ABILITY_USED_AT_POSITION]:
        raise AmbiguousActionError(f"Target and position events conflict for {controlled_entity_id}: sequences {tuple(event.sequence for event in matches)}")
    for kind in (EventKind.ABILITY_USED_ON_TARGET, EventKind.ABILITY_USED_AT_POSITION, EventKind.ABILITY_USED):
        if by_kind[kind]:
            return by_kind[kind][0]
    raise AssertionError("Action event set unexpectedly empty")
