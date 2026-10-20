import json

import pytest

from integration.reader import RecordDirectoryReader, validate_sequence
from integration.schema import EventKind, GameEvent, GameSnapshot
from integration.serialization import record_to_dict


def write(path, record) -> None:
    path.write_text(json.dumps(record_to_dict(record)))


def test_records_come_back_in_embedded_order(tmp_path) -> None:
    write(tmp_path / "a.json", GameEvent(1, 3, EventKind.MISS))
    write(tmp_path / "z.json", GameSnapshot(1, 1))
    write(tmp_path / "m.json", GameEvent(1, 2, EventKind.TURN_STARTED))
    (tmp_path / "notes.txt").write_text("ignored")
    reader = RecordDirectoryReader(tmp_path)
    assert [record.sequence for record in reader.read_all()] == [1, 2, 3]
    assert reader.validate_all().gaps == ()


def test_duplicate_on_the_first_read_is_rejected(tmp_path) -> None:
    write(tmp_path / "first.json", GameSnapshot(1, 1))
    write(tmp_path / "second.json", GameEvent(1, 1, EventKind.MISS))
    with pytest.raises(ValueError, match="Duplicate sequence 1 in"):
        RecordDirectoryReader(tmp_path).read_all()


def test_duplicate_still_rejected_if_original_file_disappears(tmp_path) -> None:
    reader = RecordDirectoryReader(tmp_path)
    first = tmp_path / "first.json"
    write(first, GameSnapshot(1, 1))
    reader.read_all()
    first.unlink()
    write(tmp_path / "second.json", GameEvent(1, 1, EventKind.MISS))
    with pytest.raises(ValueError, match="Duplicate sequence 1"):
        reader.read_all()


def test_offline_validation_reports_missing_start(tmp_path) -> None:
    write(tmp_path / "second.json", GameSnapshot(1, 2))
    reader = RecordDirectoryReader(tmp_path)
    assert reader.validate_all().gaps == (1,)


def test_malformed_file_names_its_path(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{bad")
    with pytest.raises(ValueError, match="broken.json"):
        RecordDirectoryReader(tmp_path).read_all()
    path.write_text('{"record_type": "event", "schema_version": 1, "sequence": 1}')
    with pytest.raises(ValueError, match="broken.json"):
        RecordDirectoryReader(tmp_path).read_all()


def test_sequence_validation() -> None:
    records = (GameSnapshot(1, 2), GameEvent(1, 4, EventKind.MISS))
    assert validate_sequence(records).gaps == (1, 3)
    with pytest.raises(ValueError, match="not increasing"):
        validate_sequence(records[::-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sequence((records[0], records[0]))
    with pytest.raises(ValueError, match="schema versions"):
        validate_sequence((GameSnapshot(1, 1), GameSnapshot(2, 2)))
