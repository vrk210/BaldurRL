"""Python-only publication tests; these do NOT execute Lua or BG3."""

import json

import pytest

from integration.publish import publish_ready
from integration.reader import RecordDirectoryReader


def staged(source, payload=None):
    payload = payload or {"record_type": "event", "schema_version": 1, "sequence": 1, "kind": "died"}
    body = (json.dumps(payload) + "\n").encode()
    source.mkdir(exist_ok=True)
    (source / "record-00000001.pending").write_bytes(body)
    (source / "record-00000001.ready").write_bytes(f"{len(body)}\n".encode())
    return body


def test_ignores_uncommitted_data_and_partial_marker(tmp_path):
    source, destination = tmp_path / "staging", tmp_path / "records"
    source.mkdir()
    (source / "record-00000001.pending").write_text('{"record_type":')
    assert publish_ready(source, destination) == ()
    (source / "record-00000001.ready").write_text("123")
    assert publish_ready(source, destination) == ()
    assert RecordDirectoryReader(destination).read_all() == ()


def test_atomic_publication_and_idempotence(tmp_path, monkeypatch):
    source, destination = tmp_path / "staging", tmp_path / "records"
    body = staged(source)
    import integration.publish as publisher
    replace = publisher.os.replace

    def checked_replace(temporary, target):
        assert not target.exists()
        assert RecordDirectoryReader(destination).read_all() == ()
        assert temporary.read_bytes() == body
        replace(temporary, target)

    monkeypatch.setattr(publisher.os, "replace", checked_replace)
    assert len(publish_ready(source, destination)) == 1
    assert publish_ready(source, destination) == ()
    assert [r.sequence for r in RecordDirectoryReader(destination).read_new()] == [1]
    assert len(list(destination.iterdir())) == 1


@pytest.mark.parametrize("failure", ["length", "schema", "sequence", "conflict", "json"])
def test_rejects_marked_corruption_and_conflicting_records(tmp_path, failure):
    source, destination = tmp_path / "staging", tmp_path / "records"
    body = staged(source)
    if failure == "length":
        (source / "record-00000001.ready").write_text("1\n")
    elif failure == "schema":
        staged(source, {"record_type": "event", "schema_version": 2, "sequence": 1, "kind": "died"})
    elif failure == "sequence":
        staged(source, {"record_type": "event", "schema_version": 1, "sequence": 2, "kind": "died"})
    elif failure == "conflict":
        destination.mkdir()
        (destination / "record-00000001.json").write_text("preserve me")
    else:
        (source / "record-00000001.pending").write_bytes(b"{" + b" " * (len(body) - 1))
    with pytest.raises(ValueError):
        publish_ready(source, destination)
    if failure == "conflict":
        assert (destination / "record-00000001.json").read_text() == "preserve me"
    else:
        assert not list(destination.glob("*.json"))
