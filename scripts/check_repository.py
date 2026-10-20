"""Check the maintained inventory, documentation and artifacts.

Run from any directory: python scripts/check_repository.py
Only tracked files are inspected; ignored experiment output is not maintenance
surface. ARCHITECTURE.md is the inventory, so no parallel registry is needed.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess
from typing import NamedTuple
from urllib.parse import unquote


DOCS = {
    "README.md", "AGENTS.md", "docs/ARCHITECTURE.md", "docs/BG3_INTEGRATION.md",
    "docs/GAME_KNOWLEDGE.md", "docs/RESULTS.md", "bg3/h0/README.md",
}
ARTIFACT_SUFFIXES = {".json", ".jsonl", ".npz", ".zip", ".log"}
ROW = re.compile(r"^\| `([^`]+)` \| (supported|verification-only|remove) \| (.+) \|$", re.M)
LINK = re.compile(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")


class InventoryRow(NamedTuple):
    status: str
    purpose: str


UNLISTED = InventoryRow("", "")


def inventory(text: str) -> tuple[dict[str, InventoryRow], list[str]]:
    rows: dict[str, InventoryRow] = {}
    errors = []
    for path, status, purpose in ROW.findall(text):
        if path in rows:
            errors.append(f"Duplicate inventory row: {path}")
        rows[path] = InventoryRow(status, purpose)
    if not rows:
        errors.append("ARCHITECTURE.md has no maintained-system inventory")
    return rows, errors


def check(root: Path, tracked: set[str]) -> list[str]:
    rows, errors = inventory((root / "docs/ARCHITECTURE.md").read_text(encoding="utf-8"))
    for path in sorted(rows.keys() - tracked):
        errors.append(f"Inventory path is not tracked: {path}")
    for name in sorted(tracked):
        path = root / name
        row = rows.get(name, UNLISTED)
        if path.suffix != ".md" and name not in rows:
            errors.append(f"Tracked file needs an inventory row: {name}")
        if path.suffix == ".md":
            if name not in DOCS and row.status != "remove":
                errors.append(f"Document {name} is not one of the agreed documents; adding one is the owner's decision")
            prose = re.sub(r"^[ \t]*(```|~~~).*?^[ \t]*\1", "", path.read_text(encoding="utf-8"), flags=re.M | re.S)
            for target in LINK.findall(prose):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                destination = unquote(target.split("#", 1)[0])
                resolved = (path.parent / destination).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    errors.append(f"Broken/outside-repository link in {name}: {target}")
        if path.suffix in ARTIFACT_SUFFIXES:
            if row.status == "remove":
                continue
            if row.purpose.startswith("fixture:"):
                match = re.match(r"fixture: ((?:tests|scripts)/[^; ]+\.py)", row.purpose)
                if not match or match[1] not in tracked:
                    errors.append(f"Fixture needs a reader in tests/ or scripts/ named in inventory: {name}")
            elif not row.purpose.startswith("config:"):
                errors.append(
                    f"Committed {path.suffix} file needs an inventory row starting 'fixture: <reader>.py' "
                    f"or 'config:'; generated output belongs in ignored runs/: {name}"
                )
        if path.suffix == ".py" and not name.startswith("tests/"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name.startswith("_") and row.status != "remove" and f"private:{alias.name}" not in row.purpose:
                            errors.append(f"Private-name import {alias.name} in {name}:{node.lineno}")
            cli = any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in tree.body)
            if cli and row.status != "remove" and "CLI:" not in row.purpose:
                errors.append(f"Inventory must identify the CLI: {name}")
    return errors


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[1]
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")) - {""}
    errors = check(root, tracked)
    for error in errors:
        print(error)
    if not errors:
        print("Repository policy checks passed")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
