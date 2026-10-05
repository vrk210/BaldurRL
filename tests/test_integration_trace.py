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


@pytest.mark.parametrize("specific_kind", [EventKind.ABILITY_USED_ON_TARGET, EventKind.ABILITY_USED_AT_POSITION])
def test_same_story_action_id_selects_specific_event(specific_kind: EventKind) -> None:
    generic = GameEvent(1, 2, EventKind.ABILITY_USED, actor_id="player-1", story_action_id=100)
    specific = GameEvent(1, 3, specific_kind, actor_id="player-1", story_action_id=100)
    enemy = GameEvent(1, 4, EventKind.ABILITY_USED, actor_id="enemy-1", story_action_id=101)
    interval = reconstruct_intervals((GameSnapshot(1, 1), generic, specific, enemy, GameSnapshot(1, 5)))[0]
    assert interval.events == (generic, specific, enemy)
    assert controlled_action(interval, "player-1") is specific


def test_distinct_story_action_ids_remain_ambiguous() -> None:
    interval = reconstruct_intervals((GameSnapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED, actor_id="player-1", story_action_id=100),
        GameEvent(1, 3, EventKind.ABILITY_USED_ON_TARGET, actor_id="player-1", story_action_id=101),
        GameSnapshot(1, 4)))[0]
    with pytest.raises(AmbiguousActionError):
        controlled_action(interval, "player-1")


def test_missing_story_action_id_cannot_join_known_group() -> None:
    interval = reconstruct_intervals((GameSnapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED, actor_id="player-1", story_action_id=100),
        GameEvent(1, 3, EventKind.ABILITY_USED_ON_TARGET, actor_id="player-1"),
        GameSnapshot(1, 4)))[0]
    with pytest.raises(AmbiguousActionError):
        controlled_action(interval, "player-1")


def test_target_and_position_events_with_same_id_remain_ambiguous() -> None:
    interval = reconstruct_intervals((GameSnapshot(1, 1),
        GameEvent(1, 2, EventKind.ABILITY_USED_ON_TARGET, actor_id="player-1", story_action_id=100),
        GameEvent(1, 3, EventKind.ABILITY_USED_AT_POSITION, actor_id="player-1", story_action_id=100),
        GameSnapshot(1, 4)))[0]
    with pytest.raises(AmbiguousActionError, match="Target and position"):
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
    assert len(records) == 10
    assert validate_sequence(records).gaps == ()
    intervals = reconstruct_intervals(records)
    assert len(intervals) == 2
    assert [event.kind for event in intervals[0].events] == [
        EventKind.ABILITY_USED, EventKind.ABILITY_USED_ON_TARGET, EventKind.DAMAGE,
    ]
    assert controlled_action(intervals[0], "player-1") is records[2]
    assert controlled_action(intervals[0], "player-1").ability_id == "ability-basic-attack"
    assert controlled_action(intervals[1], "player-1") is None
    output = inspect_directory(tmp_path)
    assert "records: 10" in output
    assert "gaps: none" in output
    assert "single controlled actions: 1" in output
