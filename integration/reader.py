"""Sequence-aware polling and offline checks for one-record JSON files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schema import SCHEMA_VERSION
from .serialization import Record, parse_record


@dataclass(frozen=True)
class SequenceReport:
    first: int | None
    last: int | None
    gaps: tuple[int, ...]


def validate_sequence(records: Iterable[Record], *, start_sequence: int | None = None) -> SequenceReport:
    """Check input order and duplicates; report missing IDs without using timestamps."""
    values = tuple(records)
    versions = {record.schema_version for record in values}
    if versions and versions != {SCHEMA_VERSION}:
        raise ValueError(f"Unsupported or inconsistent schema versions: {sorted(versions)}")
    if start_sequence is not None and start_sequence < 1:
        raise ValueError("start_sequence must be positive")
    numbers = [record.sequence for record in values]
    if any(number < 1 for number in numbers):
        raise ValueError("sequence must be positive")
    for before, after in zip(numbers, numbers[1:]):
        if after == before:
            raise ValueError(f"Duplicate sequence {after}")
        if after < before:
            raise ValueError(f"Sequence is not increasing: {before} then {after}")
    if not numbers:
        return SequenceReport(None, None, ())
    first = start_sequence if start_sequence is not None else numbers[0]
    if numbers[0] < first:
        raise ValueError(f"Sequence {numbers[0]} precedes expected start {first}")
    gaps = tuple(sorted(set(range(first, numbers[-1] + 1)) - set(numbers)))
    return SequenceReport(numbers[0], numbers[-1], gaps)


class RecordDirectoryReader:
    """Poll a directory and emit each contiguous record once.

    The caller supplies the first expected sequence if a stream begins after 1.
    Files are scanned again on every poll so a newly created duplicate is caught.
    """

    def __init__(self, directory: str | Path, *, start_sequence: int = 1) -> None:
        if start_sequence < 1:
            raise ValueError("start_sequence must be positive")
        self.directory = Path(directory)
        self.start_sequence = start_sequence
        self.next_sequence = start_sequence
        self._seen: dict[Path, Record] = {}

    def _scan(self) -> list[Record]:
        if not self.directory.is_dir():
            raise ValueError(f"Record directory does not exist: {self.directory}")
        by_sequence: dict[int, Path] = {record.sequence: path for path, record in self._seen.items()}
        records: list[Record] = []
        for path in sorted(self.directory.glob("*.json")):
            if not path.is_file():
                continue
            if path not in self._seen:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSON record in {path}: {exc}") from exc
                try:
                    self._seen[path] = parse_record(payload)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid record in {path}: {exc}") from exc
            record = self._seen[path]
            if record.sequence in by_sequence and by_sequence[record.sequence] != path:
                raise ValueError(f"Duplicate sequence {record.sequence} in {by_sequence[record.sequence]} and {path}")
            by_sequence[record.sequence] = path
            records.append(record)
        return sorted(records, key=lambda record: record.sequence)

    def read_new(self) -> tuple[Record, ...]:
        """Emit only the next contiguous run; retain later records for future polls."""
        records = {record.sequence: record for record in self._scan()}
        emitted: list[Record] = []
        while self.next_sequence in records:
            emitted.append(records[self.next_sequence])
            self.next_sequence += 1
        return tuple(emitted)

    def read_all(self) -> tuple[Record, ...]:
        """Return every record in embedded sequence order, including across gaps."""
        return tuple(self._scan())

    def validate_all(self, *, start_sequence: int | None = None) -> SequenceReport:
        return validate_sequence(self.read_all(), start_sequence=self.start_sequence if start_sequence is None else start_sequence)
