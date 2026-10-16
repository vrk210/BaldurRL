"""Check the maintained inventory, documentation, artifacts, and PR size.

Run from any directory: python scripts/check_repository.py [--base BASE].
Only tracked files are inspected; ignored experiment output is not maintenance
surface. ARCHITECTURE.md is the inventory, so no parallel registry is needed.
CI reads the agreed size budget from the PR body, never from shell interpolation.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


DOCS = {
    "README.md", "AGENTS.md", "docs/ARCHITECTURE.md", "docs/BG3_INTEGRATION.md",
    "docs/GAME_KNOWLEDGE.md", "docs/RESULTS.md", "bg3/h0/README.md",
}
SOURCE_SUFFIXES = {".py", ".lua", ".ps1"}
ARTIFACT_SUFFIXES = {".json", ".jsonl", ".npz", ".zip", ".log"}
ROW = re.compile(r"^\| `([^`]+)` \| (supported|verification-only|remove) \| (.+) \|$", re.M)
LINK = re.compile(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def inventory(text: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    rows: dict[str, tuple[str, str]] = {}
    errors = []
    for path, status, purpose in ROW.findall(text):
        if path in rows:
            errors.append(f"Duplicate inventory row: {path}")
        rows[path] = status, purpose
    if not rows:
        errors.append("ARCHITECTURE.md has no maintained-system inventory")
    return rows, errors


def check(root: Path, tracked: set[str]) -> list[str]:
    rows, errors = inventory((root / "docs/ARCHITECTURE.md").read_text(encoding="utf-8"))
    for path in sorted(rows.keys() - tracked):
        errors.append(f"Inventory path is not tracked: {path}")
    for name in sorted(tracked):
        path = root / name
        if path.suffix in SOURCE_SUFFIXES and name not in rows:
            errors.append(f"Module/entry point needs an inventory row: {name}")
        if path.suffix == ".md":
            if name not in DOCS and rows.get(name, (None,))[0] != "remove":
                errors.append(f"Document needs an agreed ongoing purpose: {name}")
            for target in LINK.findall(path.read_text(encoding="utf-8")):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                destination = unquote(target.split("#", 1)[0])
                resolved = (path.parent / destination).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    errors.append(f"Broken/outside-repository link in {name}: {target}")
        if path.suffix in ARTIFACT_SUFFIXES:
            status, purpose = rows.get(name, ("", ""))
            if status == "remove":
                continue
            if purpose.startswith("fixture:"):
                match = re.match(r"fixture: (tests/[^; ]+\.py)", purpose)
                if not match or match[1] not in tracked:
                    errors.append(f"Fixture needs a consuming test in inventory: {name}")
            elif not purpose.startswith("config:"):
                errors.append(f"Generated artifact must stay in ignored runs/: {name}")
        if path.suffix == ".py" and not name.startswith("tests/"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    status, purpose = rows.get(name, ("", ""))
                    for alias in node.names:
                        if alias.name.startswith("_") and status != "remove" and f"private:{alias.name}" not in purpose:
                            errors.append(f"Private-name import in {name}:{node.lineno}")
            cli = any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in tree.body)
            if cli and rows.get(name, (None,))[0] != "remove" and "CLI:" not in rows.get(name, ("", ""))[1]:
                errors.append(f"Inventory must identify the CLI: {name}")
    return errors


def check_budget(numstat: str, body: str) -> tuple[list[str], str]:
    match = re.search(r"^Size budget: (\d+)( additions)?\s*$", body, re.M)
    limit = int(match[1]) if match else 1000
    retirement = bool(match and match[2])
    errors = []
    if limit <= 0:
        errors.append("Size budget must be positive")
    if (limit != 1000 or retirement) and not re.search(r"^Budget reason: \S.+$", body, re.M):
        errors.append("Non-default budget needs the agreed scope in 'Budget reason: ...'")
    added = deleted = tests = 0
    for line in numstat.splitlines():
        insertions, deletions, name = line.split("\t", 2)
        if insertions == "-" or deletions == "-":
            errors.append(f"Binary change requires separate artifact review: {name}")
            continue
        a, d = int(insertions), int(deletions)
        if name.startswith("tests/"):
            tests += a + d
        else:
            added += a
            deleted += d
        if retirement and a and name not in {"docs/RESULTS.md", "docs/ARCHITECTURE.md"}:
            errors.append(f"Retirement permits only deletions outside summaries/inventory: {name}")
    counted = added if retirement else added + deleted
    if counted > limit:
        errors.append(f"PR exceeds agreed size budget: {counted} > {limit}")
    return errors, f"Non-test: +{added}/-{deleted}; tests: {tests} changed lines; budget: {limit}{' additions' if retirement else ' changed lines'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Compare PR size against BASE's merge base")
    parser.add_argument("--event", type=Path, help="GitHub event JSON containing pull_request.body")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")) - {""}
    errors = check(root, tracked)
    if args.base:
        body = ""
        if args.event:
            body = json.loads(args.event.read_text(encoding="utf-8")).get("pull_request", {}).get("body") or ""
        numstat = subprocess.check_output(["git", "diff", "--numstat", f"{args.base}...HEAD"], cwd=root, text=True)
        budget_errors, summary = check_budget(numstat, body)
        errors.extend(budget_errors)
        print(summary)
    for error in errors:
        print(error)
    if not errors:
        print("Repository policy checks passed")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
