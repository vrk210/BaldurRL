"""Synthetic H0 records for testing the file bridge without BG3."""

import json
from pathlib import Path

from .schema import SCHEMA_VERSION, CombatSnapshot, EntitySnapshot, EventKind, GameEvent, GameSnapshot
from .serialization import Record, record_to_dict


def mock_records() -> tuple[Record, ...]:
    def snapshot(sequence: int, player_hp: int, enemy_hp: int, active: str) -> GameSnapshot:
        return GameSnapshot(
            SCHEMA_VERSION, sequence, controlled_entity_id="player-1",
            combat=CombatSnapshot("fake-combat-1", 1, active, ("player-1", "enemy-1")),
            entities=(
                EntitySnapshot("player-1", hp=player_hp, max_hp=20, alive=player_hp > 0),
                EntitySnapshot("enemy-1", hp=enemy_hp, max_hp=15, alive=enemy_hp > 0),
            ),
        )

    def event(sequence: int, kind: EventKind, **fields: object) -> GameEvent:
        return GameEvent(SCHEMA_VERSION, sequence, kind, combat_id="fake-combat-1", **fields)

    return (
        snapshot(1, 20, 15, "player-1"),
        event(2, EventKind.ABILITY_USED, actor_id="player-1", ability_id="ability-basic-attack", story_action_id=100),
        event(3, EventKind.ABILITY_USED_ON_TARGET, actor_id="player-1", target_id="enemy-1", ability_id="ability-basic-attack", story_action_id=100),
        event(4, EventKind.DAMAGE, actor_id="player-1", target_id="enemy-1", damage=5, story_action_id=100),
        snapshot(5, 20, 10, "player-1"),
        event(6, EventKind.TURN_ENDED, actor_id="player-1"),
        event(7, EventKind.ABILITY_USED_ON_TARGET, actor_id="enemy-1", target_id="player-1", ability_id="ability-enemy-attack", story_action_id=101),
        event(8, EventKind.MISS, actor_id="enemy-1", target_id="player-1", story_action_id=101),
        snapshot(9, 20, 10, "player-1"),
        event(10, EventKind.COMBAT_ENDED),
    )


def write_mock_trace(directory: str | Path) -> tuple[Path, ...]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for record in mock_records():
        path = destination / f"mock-{record.sequence:04d}.json"
        path.write_text(json.dumps(record_to_dict(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Write a synthetic H0 record directory")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = write_mock_trace(args.directory)
    print(f"wrote {len(paths)} synthetic records to {args.directory}")


if __name__ == "__main__":
    main()
