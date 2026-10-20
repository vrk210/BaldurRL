"""Behavior checks for the repository contract, independent of the live checkout."""

from pathlib import Path

import pytest

from scripts.check_repository import check, inventory


def repository(tmp_path: Path, rows: str, files: dict[str, str]) -> set[str]:
    files = {"docs/ARCHITECTURE.md": rows, **files}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return set(files)


def row(path: str, status: str = "supported", purpose: str = "Behavior contract.") -> str:
    return f"| `{path}` | {status} | {purpose} |\n"


def test_inventory_rejects_duplicate_paths_and_unknown_statuses():
    _, errors = inventory(row("a.py") + row("a.py") + row("b.py", "verification only"))
    assert errors == ["Duplicate inventory row: a.py",
                      "Unknown inventory status 'verification only' for b.py; use supported or verification-only"]


def test_unclassified_file_of_any_type_is_rejected(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "", "b.py": "", "tool.sh": ""})
    assert check(tmp_path, tracked) == ["Tracked file needs an inventory row: b.py", "Tracked file needs an inventory row: tool.sh"]


def test_removed_inventory_path_is_rejected(tmp_path):
    tracked = repository(tmp_path, row("gone.py"), {})
    assert "Inventory path is not tracked: gone.py" in check(tmp_path, tracked)


def test_unagreed_document_is_rejected(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "", "plan.md": ""})
    assert check(tmp_path, tracked) == ["Document plan.md is not one of the agreed documents; adding one is the owner's decision"]


def test_relative_links_must_exist_and_stay_inside_repository(tmp_path):
    (tmp_path / "outside.md").write_text("", encoding="utf-8")
    root = tmp_path / "repo"
    tracked = repository(root, row("a.py"), {"a.py": "", "README.md": "[bad](missing.py) [outside](../outside.md) [web](https://example.com)\n```\nhandlers[kind](event)\n```\n1. Step:\n   ```\n   [int]($count)\n   ```\n"})
    errors = check(root, tracked)
    assert len(errors) == 2
    assert all("Broken/outside-repository link" in error for error in errors)


@pytest.mark.parametrize("purpose", ["Generated output.", "fixture: tests/missing.py; synthetic"])
def test_generated_output_needs_valid_fixture_consumer(tmp_path, purpose):
    tracked = repository(tmp_path, row("result.json", purpose=purpose), {"result.json": "{}"})
    assert len(check(tmp_path, tracked)) == 1


def test_explicit_fixture_and_configuration_are_allowed(tmp_path):
    rows = row("data/capture/1.json", "verification-only", "fixture: tests/test_data.py; synthetic")
    rows += row("tests/test_data.py", "verification-only") + row("config.json", purpose="config: module settings")
    tracked = repository(tmp_path, rows, {"data/capture/1.json": "{}", "tests/test_data.py": "", "config.json": "{}"})
    assert check(tmp_path, tracked) == []


def test_private_import_and_unlisted_cli_are_rejected(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "from other import _hidden\nif __name__ == '__main__':\n    pass\n"})
    assert len(check(tmp_path, tracked)) == 2

