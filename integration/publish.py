"""Publish completed BG3 Lua staging files as atomic, immutable v1 records.

Script Extender has no documented rename API. Lua writes .pending JSON followed
by a .ready byte count. Run one publisher per session on a local filesystem.
"""

import json
import os
import re
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

from .serialization import parse_record


def publish_ready(staging: str | Path, directory: str | Path) -> tuple[Path, ...]:
    """Validate committed staging records and atomically publish new files.

    Incomplete markers are retried on the next call. Marked corrupt records and
    destination conflicts fail visibly. Staging files are retained for audit.
    """
    source, destination = Path(staging), Path(directory)
    if not source.is_dir():
        raise ValueError(f"Staging directory does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Staging and record directories must differ")
    destination.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    for marker in sorted(source.glob("record-*.ready")):
        match = re.fullmatch(r"record-([0-9]+)\.ready", marker.name)
        if match is None:
            raise ValueError(f"Invalid completion filename: {marker}")
        completion = marker.read_bytes()
        # SaveFile may currently be writing the marker; newline terminates it.
        if not completion.endswith(b"\n"):
            continue
        if re.fullmatch(rb"[1-9][0-9]*\n", completion) is None:
            raise ValueError(f"Invalid completion marker: {marker}")
        pending = marker.with_suffix(".pending")
        payload = pending.read_bytes()
        if len(payload) != int(completion):
            raise ValueError(f"Completed record byte count mismatch: {pending}")
        try:
            record = parse_record(json.loads(payload))
        except (ValueError, UnicodeError, TypeError) as exc:
            raise ValueError(f"Invalid completed record in {pending}: {exc}") from exc
        if record.sequence != int(match[1]):
            raise ValueError(f"Filename/record sequence mismatch: {pending}")
        target = destination / marker.with_suffix(".json").name
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite immutable record: {target}")
            continue
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".pending", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        published.append(target)
    return tuple(published)


def main() -> None:
    parser = ArgumentParser(description="Publish completed passive BG3 Lua records")
    parser.add_argument("staging", type=Path, help="One collector session's staging directory")
    parser.add_argument("directory", type=Path, help="Empty/new record directory for that session")
    parser.add_argument("--watch", action="store_true", help="Publish every 0.5 seconds until Ctrl-C")
    args = parser.parse_args()
    try:
        while True:
            paths = publish_ready(args.staging, args.directory)
            if paths:
                print(f"published {len(paths)} records to {args.directory}", flush=True)
            if not args.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
