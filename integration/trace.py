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
    """Return one plausible human action; flag multiple without guessing."""
    matches = tuple(event for event in interval.events if event.actor_id == controlled_entity_id and event.kind in _ACTION_KINDS)
    if len(matches) > 1:
        raise AmbiguousActionError(f"Multiple action events for {controlled_entity_id}: sequences {tuple(event.sequence for event in matches)}")
    return matches[0] if matches else None
