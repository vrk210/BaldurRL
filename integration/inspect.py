"""Concise offline inspection of a directory of integration records."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from .reader import RecordDirectoryReader
from .schema import GameEvent, GameSnapshot
from .trace import AmbiguousActionError, controlled_action, reconstruct_intervals


def inspect_directory(directory: str | Path) -> str:
    reader = RecordDirectoryReader(directory)
    records = reader.read_all()
    report = reader.validate_all()
    snapshots = sum(isinstance(record, GameSnapshot) for record in records)
    events = [record for record in records if isinstance(record, GameEvent)]
    lines = [
        f"records: {len(records)}",
        f"sequence: {report.first}..{report.last}" if records else "sequence: none",
        f"snapshots: {snapshots}",
        f"events: {len(events)}",
        f"gaps: {', '.join(map(str, report.gaps)) if report.gaps else 'none'}",
        "",
    ]
    for kind, count in sorted(Counter(event.kind.value for event in events).items()):
        lines.append(f"{kind}: {count}")
    if not report.gaps and records and isinstance(records[0], GameSnapshot):
        intervals = reconstruct_intervals(records)
        actions = ambiguous = 0
        for interval in intervals:
            if interval.before.controlled_entity_id is None:
                continue
            try:
                actions += controlled_action(interval, interval.before.controlled_entity_id) is not None
            except AmbiguousActionError:
                ambiguous += 1
        lines += ["", f"intervals: {len(intervals)}", f"single controlled actions: {actions}", f"ambiguous controlled actions: {ambiguous}"]
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser(description="Inspect passive BG3 JSON records")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        print(inspect_directory(args.directory))
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
