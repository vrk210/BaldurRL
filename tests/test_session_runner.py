"""Offline session-runner tests; these do not execute BG3 or Script Extender."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from integration.session_runner import (
    LOCK_NAME, SessionRunner, _archive_content_addressed, discover_sessions, main,
)


def stage(session: Path, sequence: int, payload: dict | None = None, *, complete: bool = True) -> bytes:
    staging = session / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    value = payload or {
        "record_type": "event", "schema_version": 1, "sequence": sequence, "kind": "died",
    }
    body = (json.dumps(value, sort_keys=True) + "\n").encode()
    base = staging / f"record-{sequence:08d}"
    base.with_suffix(".pending").write_bytes(body)
    if complete:
        base.with_suffix(".ready").write_bytes(f"{len(body)}\n".encode())
    return body


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_multiple_sessions_and_new_discovery_keep_independent_sequences(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    stage(capture / "session-a", 1)
    stage(capture / "session-b", 1)
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        assert [item["session_id"] for item in runner.poll()] == ["session-a", "session-b"]
        stage(capture / "session-c", 1)
        assert [item["session_id"] for item in runner.poll()] == ["session-a", "session-b", "session-c"]
    for name in ("session-a", "session-b", "session-c"):
        record = archive / name / "records" / "record-00000001.json"
        assert read_json(record)["sequence"] == 1


def test_partial_bytes_are_content_addressed_then_completed_and_published(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    session = capture / "session"
    body = stage(session, 1, complete=False)
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        first = runner.poll()[0]
        assert first["stats"]["record_file_count"] == 0
        incomplete = list((archive / "session" / "diagnostics" / "incomplete").glob("*.bin"))
        assert [path.read_bytes() for path in incomplete] == [body]
        pending = session / "staging" / "record-00000001.pending"
        pending.write_bytes(body + b" ")
        (session / "staging" / "record-00000001.ready").write_bytes(f"{len(body) + 1}\n".encode())
        second = runner.poll()[0]
        assert second["stats"]["published_this_poll"] == 1
    assert (archive / "session" / "staging" / "record-00000001.pending").read_bytes() == body + b" "
    assert len(list((archive / "session" / "diagnostics" / "incomplete").glob("*.bin"))) == 1


def test_content_address_uses_the_same_single_byte_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "growing.pending"
    source.write_bytes(b"on disk")
    destination = tmp_path / "diagnostics"
    first, later = b"first observed bytes", b"later growing bytes"
    original_read_bytes = Path.read_bytes
    reads = 0

    def changing_read(path):
        nonlocal reads
        if path == source:
            reads += 1
            return first if reads == 1 else later
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    archived = _archive_content_addressed(source, destination)
    assert reads == 1
    assert archived.name == f"{source.name}.{hashlib.sha256(first).hexdigest()}.bin"
    assert original_read_bytes(archived) == first


def test_completed_corruption_is_error_and_raw_bytes_are_archived(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    session = capture / "bad"
    body = stage(session, 1)
    (session / "staging" / "record-00000001.ready").write_bytes(b"1\n")
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        with pytest.raises(ValueError, match=r"Session bad.*byte count mismatch"):
            runner.poll()
    assert (archive / "bad" / "staging" / "record-00000001.pending").read_bytes() == body
    assert not (session / LOCK_NAME).exists()


def test_idempotence_and_differing_immutable_archive_conflict(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    session = capture / "stable"
    stage(session, 1)
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        runner.poll()
        assert runner.poll()[0]["stats"]["published_this_poll"] == 0
        (archive / "stable" / "records" / "record-00000001.json").write_bytes(b"different")
        with pytest.raises(ValueError, match="differing archived file"):
            runner.poll()


@pytest.mark.parametrize("archive_relative", [".", "inside"])
def test_rejects_overlapping_roots(tmp_path, archive_relative):
    capture = tmp_path / "capture"
    capture.mkdir()
    archive = capture if archive_relative == "." else capture / archive_relative
    with pytest.raises(ValueError, match="must not overlap"):
        SessionRunner(capture, archive, repository=tmp_path)


def test_rejects_archive_containing_capture(tmp_path):
    archive = tmp_path / "archive"
    capture = archive / "capture"
    capture.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not overlap"):
        SessionRunner(capture, archive, repository=tmp_path)


def test_provenance_unknowns_health_gaps_and_reported_selection(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    session = capture / "health"
    stage(session, 2, {
        "record_type": "snapshot", "schema_version": 1, "sequence": 2,
        "controlled_entity_id": "selected-actor",
        "combat": {"active_entity_id": None},
        "entities": [{
            "entity_id": "selected-actor",
            "resources": [{"resource_id": "observed-resource", "amount": 1}],
            "abilities": [{"ability_id": "observed-ability", "known": True}],
        }],
        "candidate_decisions": [],
    })
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        report = runner.poll()[0]
    health = report["health"]
    assert health["sequence_gaps"] == [1]
    assert health["selected_entity"] == {
        "status": "known", "entity_id": "selected-actor", "source": "latest_snapshot.controlled_entity_id",
    }
    assert health["active_entity"]["status"] == "unknown"
    assert health["known_resource_count"] == 1
    assert health["known_ability_count"] == 1
    assert health["known_candidate_count"] == 0
    assert health["candidate_enumeration"]["status"] == "unknown"
    manifest = read_json(archive / "health" / "manifest" / "session-v1.json")
    assert manifest["operator_reported_versions"] == {"game": None, "script_extender": None}
    assert manifest["code_revision"]["value"] is None
    assert manifest["source"]["session_path"] == str(session.resolve())


def test_archive_session_id_cannot_be_reused_by_different_source(tmp_path):
    archive = tmp_path / "archive"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    stage(first / "same-id", 1)
    stage(second / "same-id", 1)
    with SessionRunner(first, archive, repository=tmp_path) as runner:
        runner.poll()
    with SessionRunner(second, archive, repository=tmp_path) as runner:
        with pytest.raises(ValueError, match="different source"):
            runner.poll()


def test_lock_conflict_is_visible_and_error_releases_owned_locks(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    first, second = capture / "a", capture / "b"
    stage(first, 1)
    stage(second, 1)
    (second / LOCK_NAME).write_text("pid=someone-else\n", encoding="utf-8")
    runner = SessionRunner(capture, archive, repository=tmp_path)
    with pytest.raises(ValueError, match=r"Session b.*lock already exists"):
        runner.poll()
    assert not (first / LOCK_NAME).exists()
    assert (second / LOCK_NAME).exists()


def test_later_discovery_error_releases_owned_locks(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    first = capture / "first"
    stage(first, 1)
    runner = SessionRunner(capture, archive, repository=tmp_path)
    runner.poll()
    assert (first / LOCK_NAME).exists()
    outside = tmp_path / "outside"
    (outside / "staging").mkdir(parents=True)
    (capture / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        runner.poll()
    assert not (first / LOCK_NAME).exists()


@pytest.mark.parametrize("interval", ["nan", "inf", "-inf", "0"])
def test_cli_rejects_non_finite_or_non_positive_poll_interval(tmp_path, monkeypatch, interval):
    capture = tmp_path / "capture"
    capture.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "session_runner", "--capture-root", str(capture),
        "--archive-root", str(tmp_path / "archive"), "--poll-interval", interval, "--once",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_symlinked_session_records_and_archive_directories_are_rejected(tmp_path):
    capture, archive = tmp_path / "capture", tmp_path / "archive"
    capture.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (capture / "linked").symlink_to(outside, target_is_directory=True)
    (outside / "staging").mkdir()
    with pytest.raises(ValueError, match="must not be a symlink"):
        discover_sessions(capture)

    (capture / "linked").unlink()
    session = capture / "plain"
    stage(session, 1)
    (session / "records").symlink_to(outside, target_is_directory=True)
    with SessionRunner(capture, archive, repository=tmp_path) as runner:
        with pytest.raises(ValueError, match="Records directory must not be a symlink"):
            runner.poll()

    capture_two, archive_two = tmp_path / "capture-two", tmp_path / "archive-two"
    capture_two.mkdir()
    stage(capture_two / "plain", 1)
    (archive_two / "plain").mkdir(parents=True)
    (archive_two / "plain" / "staging").symlink_to(outside, target_is_directory=True)
    with SessionRunner(capture_two, archive_two, repository=tmp_path) as runner:
        with pytest.raises(ValueError, match="Directory must not be a symlink"):
            runner.poll()


def test_probe_pure_logic_with_optional_lua_runtime():
    """Execute Lua helpers against stubs; this is not BG3 engine validation."""
    lupa = pytest.importorskip("lupa.lua54")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("""
        saved_path = nil
        saved_payload = nil
        printed_payload = nil
        local actor = {
            SpellBook = {observed = true},
            ActionResources = {observed = true},
        }
        function actor:GetAllComponentNames()
            return {"SpellBook", "ActionResources"}
        end
        Ext = {
            Utils = {
                Version = function() return 30 end,
                GameVersion = function() return "test-build" end,
                Print = function(value) printed_payload = value end,
            },
            Entity = {Get = function(id) if id == "selected" then return actor end end},
            DumpExport = function(value) return value.observed and "observed-dump" or "unexpected" end,
            IO = {SaveFile = function(path, payload)
                saved_path = path
                saved_payload = payload
                return true
            end},
        }
    """)
    lua.execute((Path(__file__).parents[1] / "bg3" / "probes" / "server_inspect.lua").read_text())
    saved, payload = lua.globals().BaldurRL_ServerInspect("selected", "session/diagnostics/probe.json")
    report = json.loads(payload)
    assert saved is True
    assert report["selected_actor_id"] == "selected"
    assert report["components"]["SpellBook"] == {"status": "ok", "value": "observed-dump"}
    assert report["components"]["TurnBased"]["status"] == "absent"
    assert lua.globals().saved_path == "session/diagnostics/probe.json"

