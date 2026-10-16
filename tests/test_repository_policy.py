"""Behavior checks for the repository contract, independent of the live checkout."""

from pathlib import Path

import pytest

from scripts.check_repository import check, check_budget, inventory


def repository(tmp_path: Path, rows: str, files: dict[str, str]) -> set[str]:
    files = {"docs/ARCHITECTURE.md": rows, **files}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return set(files)


def row(path: str, status: str = "supported", purpose: str = "Behavior contract.") -> str:
    return f"| `{path}` | {status} | {purpose} |\n"


def test_inventory_requires_unique_paths():
    _, errors = inventory(row("a.py") + row("a.py"))
    assert errors == ["Duplicate inventory row: a.py"]


def test_unclassified_module_is_rejected(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "", "b.py": ""})
    assert "Module/entry point needs an inventory row: b.py" in check(tmp_path, tracked)


def test_removed_inventory_path_is_rejected(tmp_path):
    tracked = repository(tmp_path, row("gone.py"), {})
    assert "Inventory path is not tracked: gone.py" in check(tmp_path, tracked)


def test_extra_document_needs_explicit_retirement(tmp_path):
    tracked = repository(tmp_path, row("old.md", "remove"), {"old.md": "", "plan.md": ""})
    assert check(tmp_path, tracked) == ["Document needs an agreed ongoing purpose: plan.md"]


def test_relative_links_must_exist_and_stay_inside_repository(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "", "README.md": "[bad](missing.py) [outside](../outside.md) [web](https://example.com)"})
    errors = check(tmp_path, tracked)
    assert len(errors) == 2
    assert all("Broken/outside-repository link" in error for error in errors)


@pytest.mark.parametrize("purpose", ["Generated output.", "fixture: tests/missing.py; synthetic"])
def test_generated_output_needs_valid_fixture_consumer(tmp_path, purpose):
    tracked = repository(tmp_path, row("result.json", purpose=purpose), {"result.json": "{}"})
    assert len(check(tmp_path, tracked)) == 1


def test_explicit_fixture_and_configuration_are_allowed(tmp_path):
    rows = row("test.json", "verification-only", "fixture: tests/test_data.py; synthetic")
    rows += row("tests/test_data.py", "verification-only") + row("config.json", purpose="config: module settings")
    tracked = repository(tmp_path, rows, {"test.json": "{}", "tests/test_data.py": "", "config.json": "{}"})
    assert check(tmp_path, tracked) == []


def test_private_import_and_unlisted_cli_are_rejected(tmp_path):
    tracked = repository(tmp_path, row("a.py"), {"a.py": "from other import _hidden\nif __name__ == '__main__':\n    pass\n"})
    assert len(check(tmp_path, tracked)) == 2


def test_retiring_experiment_may_exist_until_removal(tmp_path):
    tracked = repository(tmp_path, row("old.py", "remove") + row("result.json", "remove"), {"old.py": "from other import _hidden", "result.json": "{}"})
    assert check(tmp_path, tracked) == []


def test_default_budget_counts_additions_and_deletions_separately_from_tests():
    errors, summary = check_budget("600\t500\ta.py\n1000\t0\ttests/test_a.py", "")
    assert errors == ["PR exceeds agreed size budget: 1100 > 1000"]
    assert "tests: 1000" in summary


def test_nondefault_budget_requires_recorded_scope_agreement():
    errors, _ = check_budget("100\t100\ta.py", "Size budget: 2500")
    assert len(errors) == 1
    errors, _ = check_budget("100\t100\ta.py", "Size budget: 2500\nBudget reason: agreed documentation relocation")
    assert errors == []


def test_retirement_counts_deletions_but_permits_only_summary_additions():
    body = "Size budget: 300 additions\nBudget reason: agreed experiment retirement"
    errors, summary = check_budget("0\t9000\told.py\n25\t10\tdocs/RESULTS.md", body)
    assert errors == []
    assert "-9010" in summary
    errors, _ = check_budget("1\t2\told.py", body)
    assert errors == ["Retirement permits only deletions outside summaries/inventory: old.py"]


def test_binary_changes_need_explicit_review():
    errors, _ = check_budget("-\t-\tmodel.zip", "")
    assert errors == ["Binary change requires separate artifact review: model.zip"]
