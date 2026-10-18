"""How the refactor comparison maps a diff to statements and decides pass or fail."""

import pytest

from scripts.compare_refactor import (
    Statement, WorkerReport, changed_lines, changed_statements, declares_behavior_preserved, verdict,
)

SOURCE = """\
\"\"\"Module docstring.\"\"\"
import os
# A comment.

@decorator
def top(x=1):
    if x:
        return 1
    return 2


class Thing:
    value = 2

    def method(self):
        def inner():
            return 3
        return inner()


class Documented:
    \"\"\"First line.

    More text.
    \"\"\"
    flag = True
"""


@pytest.mark.parametrize("line, changed", [
    (1, set()),  # docstring
    (2, {("<module>", True)}),
    (3, set()),  # comment
    (5, {("top", True)}),  # a decorator belongs to the def it decorates
    (6, {("top", True)}),  # defaults run with the def line, at import
    (7, {("top", False)}),  # an if header resolves to the if
    (8, {("top", False)}),  # a line in its body to the body statement
    (13, {("Thing", True)}),  # class body
    (15, {("Thing.method", True)}),
    (16, {("Thing.method.<locals>.inner", False)}),  # a nested def runs when its function does
    (17, {("Thing.method.<locals>.inner", False)}),
    (18, {("Thing.method", False)}),
    (24, set()),  # inside a multi-line docstring
    (26, {("Documented", True)}),
])
def test_changed_line_maps_to_innermost_statement(line, changed):
    statements = changed_statements(SOURCE, {line})
    assert {(statement.owner, statement.runs_at_import) for statement in statements} == changed


def test_hunks_map_to_base_and_working_tree_lines_of_python_files():
    diff = """\
--- a/combat/a.py
+++ b/combat/a.py
@@ -3 +3 @@ def f():
@@ -10,2 +9,0 @@ def g():
@@ -20,0 +19,3 @@
--- a/combat/removed.py
+++ /dev/null
@@ -1,2 +0,0 @@
--- /dev/null
+++ b/combat/added.py
@@ -0,0 +1 @@
--- a/combat/notes.txt
+++ b/combat/notes.txt
@@ -1 +1 @@
"""
    base, head = changed_lines(diff)
    assert base == {"combat/a.py": {3, 10, 11}, "combat/removed.py": {1, 2}}
    assert head == {"combat/a.py": {3, 19, 20, 21}, "combat/added.py": {1}}


@pytest.mark.parametrize("body, declared", [
    ("Outcome: tidy\n\nBehavior: preserve\n", True),
    ("Outcome: tidy\r\nBehavior: preserve\r\n", True),
    ("There is no `Behavior: preserve` line here.", False),
    ("```\nBehavior: preserve\n```\n", False),
    ("<!--\nBehavior: preserve\n-->", False),
    (None, False),
])
def test_only_a_marker_line_declares_preserved_behavior(body, declared):
    assert declares_behavior_preserved(body) is declared


@pytest.mark.parametrize("line", ["**Behavior:** preserve", "behavior: preserve", "  Behavior: preserve"])
def test_a_misspelled_marker_line_fails_instead_of_skipping(line):
    with pytest.raises(ValueError, match="not exactly"):
        declares_behavior_preserved(f"Outcome: tidy\n{line}\n")


RUNTIME = Statement(7, 8, "top", False)
AT_IMPORT = Statement(13, 13, "Thing", True)


def report(*, fingerprint="same", errors=None, executed=()):
    return WorkerReport({"scenario": fingerprint}, errors or {}, {"a.py": set(executed)})


@pytest.mark.parametrize("before, after, removed, added, passed", [
    (report(), report(fingerprint="other"), {}, {}, False),  # mismatch
    (report(), report(errors={"scenario": "ValueError: x"}), {}, {}, False),
    (report(), report(executed={8}), {}, {"a.py": {RUNTIME}}, True),  # ran on the working tree
    (report(), report(), {}, {"a.py": {RUNTIME}}, False),  # never ran
    (report(), report(executed={8}), {"a.py": {RUNTIME}}, {}, False),  # removed from a surviving def, never ran
    (report(executed={7}), report(), {"a.py": {RUNTIME}}, {}, True),
])
def test_verdict_fails_on_mismatch_error_or_unexercised_statement(before, after, removed, added, passed):
    result = verdict(before=before, after=after, removed=removed, added=added,
                     surviving={"a.py": {"<module>", "top"}}, not_compared=[])
    assert result["passed"] is passed


def test_removed_definitions_import_time_and_non_python_changes_are_listed_without_failing():
    result = verdict(before=report(), after=report(), removed={"a.py": {RUNTIME}}, added={"a.py": {AT_IMPORT}},
                     surviving={"a.py": {"<module>", "Thing"}}, not_compared=["bg3/x.lua"])
    assert result["passed"]
    assert result["removed_definitions"] == ["a.py::top"]
    assert result["import_time_changes"] == ["a.py::Thing"]
    assert result["not_compared"] == ["bg3/x.lua"]
