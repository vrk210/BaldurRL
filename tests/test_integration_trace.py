import pytest

from integration.inspect import inspect_directory
from integration.mock_h0 import write_mock_trace
from integration.reader import RecordDirectoryReader, validate_sequence
from integration.schema import EventKind, GameEvent, GameSnapshot
from integration.trace import AmbiguousActionError, controlled_action, reconstruct_intervals


def test_intervals_and_controlled_action() -> None:
    records = (
        GameSnapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED_ON_TARGET, actor_id="player-1"),
        GameEvent(1, 3, EventKind.DAMAGE, actor_id="player-1"),
        GameSnapshot(1, 4),
        GameEvent(1, 5, EventKind.ABILITY_USED, actor_id="enemy-1"),
        GameSnapshot(1, 6),
    )
    intervals = reconstruct_intervals(records)
    assert len(intervals) == 2
    assert [event.sequence for event in intervals[0].events] == [2, 3]
    assert controlled_action(intervals[0], "player-1") == records[1]
    assert controlled_action(intervals[1], "player-1") is None
    assert controlled_action(intervals[1], "enemy-1") == records[4]


def test_ambiguous_actions_are_reported() -> None:
    interval = reconstruct_intervals((GameSnapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED, actor_id="player-1"),
        GameEvent(1, 3, EventKind.ABILITY_USED_AT_POSITION, actor_id="player-1"),
        GameSnapshot(1, 4)))[0]
    with pytest.raises(AmbiguousActionError, match="2, 3"):
        controlled_action(interval, "player-1")


def test_unclosed_events_and_gaps() -> None:
    assert reconstruct_intervals((GameSnapshot(1, 1), GameEvent(1, 2, EventKind.MISS))) == ()
    with pytest.raises(ValueError, match="precedes"):
        reconstruct_intervals((GameEvent(1, 1, EventKind.MISS), GameSnapshot(1, 2)))
    with pytest.raises(ValueError, match="gap"):
        reconstruct_intervals((GameSnapshot(1, 1), GameSnapshot(1, 3)))


def test_mock_h0_end_to_end(tmp_path) -> None:
    write_mock_trace(tmp_path)
    records = RecordDirectoryReader(tmp_path).read_new()
    assert len(records) == 9
    assert validate_sequence(records).gaps == ()
    intervals = reconstruct_intervals(records)
    assert len(intervals) == 2
    assert controlled_action(intervals[0], "player-1").ability_id == "ability-basic-attack"
    assert controlled_action(intervals[1], "player-1") is None
    output = inspect_directory(tmp_path)
    assert "records: 9" in output
    assert "gaps: none" in output
    assert "single controlled actions: 1" in output
