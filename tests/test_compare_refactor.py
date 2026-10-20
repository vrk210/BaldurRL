"""How the refactor comparison maps a diff to statements and decides pass or fail."""

import pytest

from scripts.compare_refactor import (
    Statement, WorkerReport, changed_lines, changed_statements, declares_behavior_preserved, split_removed_definitions,
    surviving_names, verdict,
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


def guarded(key, table):
    try:
        return table[key]
    except KeyError:
        return None
    match key:
        case "a":
            return 1
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
    (32, {("guarded", False)}),  # an except clause is its own statement, run only when reached
    (35, {("guarded", False)}),  # so is a match case
])
def test_changed_line_maps_to_innermost_statement(line, changed):
    statements = changed_statements(SOURCE, {line})
    assert {(statement.owner, statement.runs_at_import) for statement in statements} == changed


def test_except_clause_spans_only_its_handler():
    (handler,) = changed_statements(SOURCE, {32})
    assert (handler.first, handler.last) == (32, 33)


def test_a_definition_is_removed_only_when_all_of_it_goes_and_its_name_does_not_survive():
    method = set(range(15, 19))  # all of Thing.method, nested def included
    assert split_removed_definitions(SOURCE, method, surviving={"<module>"}) == (["Thing.method"], set())
    names, remaining = split_removed_definitions(SOURCE, method, surviving={"<module>", "Thing.method"})  # rewritten
    assert names == [] and {statement.owner for statement in remaining} >= {"Thing.method"}
    names, remaining = split_removed_definitions(SOURCE, {17}, surviving={"<module>"})  # one line of inner
    assert names == [] and {statement.owner for statement in remaining} == {"Thing.method.<locals>.inner"}
    assert split_removed_definitions(SOURCE, set(range(1, 48)), surviving=set()) == (["<module>"], set())  # deleted


@pytest.mark.parametrize("head_source, added, survives", [
    ("def top():\n    pass\n", [], {"<module>", "top"}),  # the file still exists, maybe renamed
    (None, [], set()),  # deleted
    (None, ["def top():\n    return 2\n"], {"<module>", "top"}),  # moved to a file git didn't pair with it
    (None, ["def other():\n    pass\n"], set()),
])
def test_surviving_names_follow_files_and_moves(head_source, added, survives):
    assert surviving_names("def top():\n    return 1\n", head_source, added) == survives


def test_hunks_map_to_lines_on_each_side_with_renames_deletions_and_additions():
    diff = """\
diff --git a/combat/a.py b/combat/a.py
--- a/combat/a.py
+++ b/combat/a.py
@@ -3 +3 @@ def f():
@@ -10,2 +9,0 @@ def g():
--- removed line that looks like a header
@@ -20,0 +19,3 @@
diff --git a/combat/removed.py b/combat/removed.py
--- a/combat/removed.py
+++ /dev/null
@@ -1,2 +0,0 @@
diff --git a/combat/old.py b/combat/new.py
similarity index 90%
rename from combat/old.py
rename to combat/new.py
--- a/combat/old.py
+++ b/combat/new.py
@@ -5 +5 @@
diff --git a/combat/created.py b/combat/created.py
--- /dev/null
+++ b/combat/created.py
@@ -0,0 +1,2 @@
diff --git a/combat/notes.txt b/combat/notes.txt
--- a/combat/notes.txt
+++ b/combat/notes.txt
@@ -1 +1 @@
"""
    changes = changed_lines(diff)
    assert changes.base == {"combat/a.py": {3, 10, 11}, "combat/removed.py": {1, 2}, "combat/old.py": {5}}
    assert changes.head == {"combat/a.py": {3, 19, 20, 21}, "combat/new.py": {5}, "combat/created.py": {1, 2}}
    assert (changes.moved, changes.added) == ({"combat/removed.py": None, "combat/old.py": "combat/new.py"},
                                              {"combat/created.py"})


@pytest.mark.parametrize("body, declared", [
    ("Outcome: tidy\n\nBehavior: preserve\n", True),
    ("Outcome: tidy\r\nBehavior: preserve\r\n", True),
    ("There is no `Behavior: preserve` line here.", False),
    ("```\nBehavior: preserve\n```\n", False),
    ("1. Example:\n   ```\n   - Behavior: preserve\n   ```\n", False),
    ("    Behavior: preserve.\n", False),  # indented code block
    ("<!--\nBehavior: preserve\n-->", False),
    (None, False),
])
def test_only_a_marker_line_declares_preserved_behavior(body, declared):
    assert declares_behavior_preserved(body) is declared


@pytest.mark.parametrize("line", [
    "**Behavior:** preserve", "behavior: preserve", "  Behavior: preserve", "- Behavior: preserve",
    "## Behavior: preserve", "> Behavior: preserve", "Behavior: preserve.", "Behavior preserve",
    "Behavior - preserve",
])
def test_a_misspelled_marker_line_fails_instead_of_skipping(line):
    with pytest.raises(ValueError, match="not exactly"):
        declares_behavior_preserved(f"Outcome: tidy\n{line}\n")


RUNTIME = Statement(7, 8, "top", False)
AT_IMPORT = Statement(13, 13, "Thing", True)


def report(*, fingerprint="same", errors=None, executed=()):
    return WorkerReport({"scenario": fingerprint}, errors or {}, {"a.py": set(executed)})


@pytest.mark.parametrize("base, head, base_changes, head_changes, passed", [
    (report(), report(fingerprint="other"), {}, {}, False),  # mismatch
    (report(), report(errors={"scenario": "ValueError: x"}), {}, {}, False),
    (report(), report(executed={8}), {}, {"a.py": {RUNTIME}}, True),  # ran on the working tree
    (report(), report(), {}, {"a.py": {RUNTIME}}, False),  # never ran
    (report(), report(executed={8}), {"a.py": {RUNTIME}}, {}, False),  # removed or changed code never ran on the base
    (report(executed={7}), report(), {"a.py": {RUNTIME}}, {}, True),
])
def test_verdict_fails_on_mismatch_error_or_unexecuted_statement(base, head, base_changes, head_changes, passed):
    result = verdict(base=base, head=head, base_changes=base_changes, head_changes=head_changes,
                     removed=[], not_compared=[])
    assert result.passed is passed


def test_import_time_changes_removed_definitions_and_other_files_are_listed_without_failing():
    result = verdict(base=report(), head=report(), base_changes={}, head_changes={"a.py": {AT_IMPORT}},
                     removed=["a.py::top"], not_compared=["bg3/x.lua"])
    assert result.passed
    assert (result.import_time_changes, result.removed_definitions, result.not_compared) == (
        ["a.py::Thing"], ["a.py::top"], ["bg3/x.lua"])
