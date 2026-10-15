"""Publish and archive passive BG3 collector sessions.

The capture root is supplied explicitly.  This module never searches a game
installation and only reads the immediate collector session directories below
that root.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .publish import publish_ready
from .reader import RecordDirectoryReader, validate_sequence
from .schema import GameEvent, GameSnapshot


REPORT_VERSION = 1
LOCK_NAME = ".baldurrl-session-runner.lock"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_roots(capture_root: Path, archive_root: Path) -> tuple[Path, Path]:
    capture = capture_root.resolve()
    archive = archive_root.resolve()
    if capture == archive or _is_within(capture, archive) or _is_within(archive, capture):
        raise ValueError(f"Capture and archive roots must not overlap: {capture} and {archive}")
    if not capture.is_dir():
        raise ValueError(f"Capture root does not exist: {capture}")
    return capture, archive


def _safe_session(session: Path, capture_root: Path) -> tuple[str, Path]:
    if session.parent != capture_root or session.name in ("", ".", ".."):
        raise ValueError(f"Session path escapes capture root: {session}")
    if session.is_symlink():
        raise ValueError(f"Session directory must not be a symlink: {session}")
    resolved = session.resolve()
    if resolved.parent != capture_root:
        raise ValueError(f"Session path traversal is not allowed: {session}")
    return session.name, resolved


def discover_sessions(capture_root: str | Path) -> tuple[Path, ...]:
    """Return immediate, non-symlink session directories containing staging/."""
    root = Path(capture_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Capture root does not exist: {root}")
    sessions: list[Path] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if candidate.is_symlink():
            if (candidate / "staging").is_dir():
                raise ValueError(f"Session directory must not be a symlink: {candidate}")
            continue
        if not candidate.is_dir():
            continue
        _, resolved = _safe_session(candidate, root)
        staging = resolved / "staging"
        if staging.is_symlink():
            raise ValueError(f"Staging directory must not be a symlink: {staging}")
        if staging.is_dir():
            sessions.append(resolved)
    return tuple(sessions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_payload_immutable(payload: bytes, destination: Path) -> bool:
    """Copy an already captured byte snapshot, accepting an identical copy."""
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite differing archived file: {destination}")
        return False
    _ensure_directory(destination.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pending", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def _copy_immutable(source: Path, destination: Path) -> bool:
    """Copy exact bytes, accepting an identical prior copy only."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Archive source must be a regular file: {source}")
    return _copy_payload_immutable(source.read_bytes(), destination)


def _archive_content_addressed(
    source: Path,
    directory: Path,
    *,
    prefix: str | None = None,
    suffix: str = ".bin",
) -> Path:
    """Read mutable source bytes once and archive that exact snapshot by hash."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Diagnostic source must be a regular file: {source}")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = directory / f"{prefix or source.name}.{digest}{suffix}"
    _copy_payload_immutable(payload, destination)
    return destination


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _ensure_directory(path.parent)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Expected directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_revision(repository: Path) -> tuple[str | None, bool | None]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True,
            text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    revision = result.stdout.strip()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
            capture_output=True, text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    else:
        dirty = bool(status.stdout.strip())
    return revision or None, dirty


def _collector_hashes(repository: Path) -> dict[str, str]:
    root = repository / "bg3" / "h0"
    if not root.is_dir():
        return {}
    return {
        path.relative_to(repository).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _health(records: tuple[GameSnapshot | GameEvent, ...]) -> dict[str, Any]:
    sequence = validate_sequence(records, start_sequence=1)
    snapshots = tuple(record for record in records if isinstance(record, GameSnapshot))
    events = tuple(record for record in records if isinstance(record, GameEvent))
    latest = records[-1] if records else None
    latest_snapshot = snapshots[-1] if snapshots else None
    active = None if latest_snapshot is None or latest_snapshot.combat is None else latest_snapshot.combat.active_entity_id
    resources = 0 if latest_snapshot is None else sum(len(entity.resources) for entity in latest_snapshot.entities)
    abilities = 0 if latest_snapshot is None else sum(
        1 for entity in latest_snapshot.entities for ability in entity.abilities if ability.known
    )
    candidates = 0 if latest_snapshot is None else len(latest_snapshot.candidate_decisions)
    return {
        "report_version": REPORT_VERSION,
        "offline_schema_valid": True,
        "live_completeness_verified": False,
        "decision_ready": False,
        "caveat": "Offline schema validity does not establish live completeness or decision readiness.",
        "record_count": len(records),
        "snapshot_count": len(snapshots),
        "event_count": len(events),
        "first_sequence": sequence.first,
        "last_sequence": sequence.last,
        "sequence_gaps": list(sequence.gaps),
        "latest_record_kind": (
            None if latest is None else "snapshot" if isinstance(latest, GameSnapshot) else latest.kind.value
        ),
        "latest_snapshot_sequence": None if latest_snapshot is None else latest_snapshot.sequence,
        "selected_entity": {
            "status": "known" if latest_snapshot is not None and latest_snapshot.controlled_entity_id is not None else "unknown",
            "entity_id": None if latest_snapshot is None else latest_snapshot.controlled_entity_id,
            "source": "latest_snapshot.controlled_entity_id",
        },
        "active_entity": {"status": "known" if active is not None else "unknown", "entity_id": active},
        "known_resource_count": resources,
        "known_ability_count": abilities,
        "known_candidate_count": candidates,
        "candidate_enumeration": {
            "status": "unknown",
            "reason": "Candidate arrays are not known complete; zero does not mean no legal actions.",
        },
        "inferences_not_made": [
            "legality", "alive_from_hp", "complete_participants", "completion", "selected_actor_from_host",
        ],
    }


@dataclass
class _SessionLock:
    path: Path
    held: bool = False

    def acquire(self) -> None:
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ValueError(
                f"Session runner lock already exists: {self.path}. Confirm no runner is active, "
                "then remove this lock manually; stale locks are never deleted automatically."
            ) from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except OSError:
            os.close(descriptor)
            self.path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        self.held = True

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


class SessionRunner:
    """Cooperatively lock, publish, and archive all discovered sessions."""

    def __init__(
        self,
        capture_root: str | Path,
        archive_root: str | Path,
        *,
        game_version: str | None = None,
        se_version: str | None = None,
        code_revision: str | None = None,
        repository: str | Path | None = None,
    ) -> None:
        self.capture_root, self.archive_root = _validate_roots(Path(capture_root), Path(archive_root))
        self.repository = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
        self.game_version = game_version
        self.se_version = se_version
        if code_revision is not None:
            self.code_revision, self.working_tree_dirty = code_revision, None
            self.revision_source = "operator_reported"
        else:
            self.code_revision, self.working_tree_dirty = _git_revision(self.repository)
            self.revision_source = "repository_lookup"
        self.collector_hashes = _collector_hashes(self.repository)
        self._locks: dict[Path, _SessionLock] = {}

    def close(self) -> None:
        for lock in self._locks.values():
            lock.release()
        self._locks.clear()

    def __enter__(self) -> SessionRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _lock(self, session: Path) -> None:
        if session not in self._locks:
            lock = _SessionLock(session / LOCK_NAME)
            lock.acquire()
            self._locks[session] = lock

    def poll(self) -> tuple[dict[str, Any], ...]:
        reports: list[dict[str, Any]] = []
        try:
            sessions = discover_sessions(self.capture_root)
        except (OSError, ValueError):
            self.close()
            raise
        for session in sessions:
            session_id, source = _safe_session(session, self.capture_root)
            try:
                self._lock(source)
                reports.append(self._process(session_id, source))
            except (OSError, ValueError) as exc:
                self.close()
                raise type(exc)(f"Session {session_id} at {source}: {exc}") from exc
        return tuple(reports)

    def _process(self, session_id: str, source: Path) -> dict[str, Any]:
        archive = self.archive_root / session_id
        manifest_path = archive / "manifest" / "session-v1.json"
        _ensure_directory(self.archive_root)
        _ensure_directory(archive)
        _ensure_directory(manifest_path.parent)
        if manifest_path.is_symlink():
            raise ValueError(f"Manifest must not be a symlink: {manifest_path}")
        if manifest_path.exists():
            try:
                prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid existing manifest {manifest_path}: {exc}") from exc
            expected = str(source)
            if prior.get("source", {}).get("session_path") != expected:
                raise ValueError(
                    f"Archive session ID {session_id!r} is already bound to a different source: {manifest_path}"
                )
        else:
            # Bind the archive name before copying any bytes.  A failed/corrupt
            # first pass must not leave an unbound directory that another source
            # with the same session ID can later reuse.
            _atomic_json(manifest_path, {
                "manifest_version": REPORT_VERSION,
                "session_id": session_id,
                "source": {"session_path": str(source)},
                "state": "processing",
            })

        staging_archive = archive / "staging"
        records_archive = archive / "records"
        diagnostics_archive = archive / "diagnostics"
        for directory in (staging_archive, records_archive, diagnostics_archive):
            _ensure_directory(directory)

        source_records = source / "records"
        if source_records.is_symlink():
            raise ValueError(f"Records directory must not be a symlink: {source_records}")
        if source_records.exists() and not source_records.is_dir():
            raise ValueError(f"Records path must be a directory: {source_records}")

        staging_files = tuple(sorted((source / "staging").glob("record-*.*")))
        for path in staging_files:
            if path.suffix not in (".pending", ".ready"):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Staging source must be a regular file: {path}")
            marker = path if path.suffix == ".ready" else path.with_suffix(".ready")
            committed = marker.is_file() and not marker.is_symlink() and marker.read_bytes().endswith(b"\n")
            if committed:
                _copy_immutable(path, staging_archive / path.name)
            else:
                # These bytes may still be growing.  A content-addressed name
                # preserves each observed diagnostic without later overwrite.
                _archive_content_addressed(path, diagnostics_archive / "incomplete")

        published = publish_ready(source / "staging", source_records)
        record_files = tuple(sorted(source_records.glob("*.json")))
        for path in record_files:
            _copy_immutable(path, records_archive / path.name)

        source_diagnostics = source / "diagnostics"
        if source_diagnostics.exists():
            if source_diagnostics.is_symlink() or not source_diagnostics.is_dir():
                raise ValueError(f"Diagnostics path must be a directory: {source_diagnostics}")
            for path in sorted(source_diagnostics.glob("*.json")):
                _archive_content_addressed(
                    path, diagnostics_archive / "probes", prefix=path.stem, suffix=path.suffix,
                )

        reader = RecordDirectoryReader(source_records)
        records = reader.read_all()
        health = _health(records)
        _atomic_json(diagnostics_archive / "health-v1.json", health)

        incomplete = []
        for pending in sorted((source / "staging").glob("record-*.pending")):
            marker = pending.with_suffix(".ready")
            if not marker.exists() or not marker.read_bytes().endswith(b"\n"):
                incomplete.append(pending.name)
        stats = {
            "staging_file_count": sum(path.is_file() for path in staging_files),
            "record_file_count": len(record_files),
            "published_this_poll": len(published),
            "incomplete_staging_payloads": incomplete,
        }
        manifest = {
            "manifest_version": REPORT_VERSION,
            "session_id": session_id,
            "source": {
                "capture_root": str(self.capture_root),
                "session_path": str(source),
                "staging_path": str(source / "staging"),
                "records_path": str(source_records),
            },
            "archive": {"session_path": str(archive)},
            "operator_reported_versions": {"game": self.game_version, "script_extender": self.se_version},
            "code_revision": {
                "value": self.code_revision,
                "source": self.revision_source,
                "working_tree_dirty": self.working_tree_dirty,
            },
            "collector_source_sha256": self.collector_hashes,
            "archived_files_sha256": {
                "staging/" + path.name: _sha256(path) for path in sorted(staging_archive.glob("*")) if path.is_file()
            } | {
                "records/" + path.name: _sha256(path) for path in sorted(records_archive.glob("*")) if path.is_file()
            } | {
                "diagnostics/" + path.relative_to(diagnostics_archive).as_posix(): _sha256(path)
                for path in sorted(diagnostics_archive.rglob("*"))
                if path.is_file()
            },
            "stats": stats,
        }
        _atomic_json(manifest_path, manifest)
        return {"session_id": session_id, "manifest": str(manifest_path), "health": health, "stats": stats}


def main() -> None:
    parser = ArgumentParser(description="Publish and archive passive BG3 collector sessions")
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Process the currently discovered sessions once")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--game-version")
    parser.add_argument("--se-version")
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be a positive finite number")
    try:
        with SessionRunner(
            args.capture_root, args.archive_root, game_version=args.game_version,
            se_version=args.se_version, code_revision=args.code_revision,
        ) as runner:
            while True:
                for report in runner.poll():
                    print(json.dumps(report, sort_keys=True), flush=True)
                if args.once:
                    break
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
