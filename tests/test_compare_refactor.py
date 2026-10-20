"""How the refactor comparison maps a diff to statements and decides pass or fail."""

import textwrap

import pytest

from scripts.compare_refactor import (
    Statement, WorkerReport, base_files_to_trace, base_side_changes, changed_lines, changed_statements, declares_behavior_preserved, outcome_message,
    run_workers, split_removed_definitions, verdict, with_executed_lines,
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


@pytest.mark.parametrize("line, span", [
    (8, (8, 8)),  # a line in an if body is that statement, which runs only when the branch does
    (32, (32, 33)),  # an except clause spans only its handler
    (33, (33, 33)),
    (35, (35, 36)),  # a match case spans its pattern and body
    (36, (36, 36)),
])
def test_changed_line_spans_only_the_lines_that_must_run(line, span):
    (statement,) = changed_statements(SOURCE, {line})
    assert (statement.first, statement.last) == span


def test_comments_and_blank_lines_inside_a_function_change_nothing():
    source = SOURCE.replace("    if x:\n        return 1\n", "    if x:\n        # note\n\n        return 1\n")
    assert changed_statements(source, {8, 9}) == set()


def test_a_definition_is_removed_only_when_all_of_it_goes_and_none_of_it_ran():
    method = set(range(15, 19))  # all of Thing.method, nested def included
    import_time = {12, 13, 15}  # the class line, its body and the def line run when the module loads
    assert split_removed_definitions(SOURCE, method, executed=import_time) == (["Thing.method"], set())
    names, remaining = split_removed_definitions(SOURCE, method, executed={18})  # it ran: removing it is a change
    assert names == [] and {statement.owner for statement in remaining} >= {"Thing.method"}
    names, remaining = split_removed_definitions(SOURCE, {17}, executed=set())  # one line of inner
    assert names == [] and {statement.owner for statement in remaining} == {"Thing.method.<locals>.inner"}
    names, _ = split_removed_definitions(SOURCE, set(range(15, 18)), executed=set())  # method keeps a line
    assert names == []  # a def nested in a function is part of the function, not a definition of its own
    names, _ = split_removed_definitions(SOURCE, {15}, executed=set())  # header only
    assert names == []


def test_a_body_on_its_def_line_is_an_import_time_change_not_a_removal():
    source = "def pick(x): return x * 2\n\n\ndef go():\n    return pick(3)\n"
    assert base_files_to_trace({"m.py": source}, {"m.py": {1}}) == set()  # line 1 runs at import either way
    names, remaining = split_removed_definitions(source, {1}, executed=set())
    assert names == [] and {(statement.owner, statement.runs_at_import) for statement in remaining} == {("pick", True)}


def test_a_deleted_file_is_removed_whole_only_if_none_of_it_ran():
    every_line = set(range(1, len(SOURCE.splitlines()) + 1))
    assert split_removed_definitions(SOURCE, every_line, executed=set()) == (["<module>"], set())
    names, _ = split_removed_definitions(SOURCE, every_line, executed={7})  # top ran: only what never ran is removed
    assert names == ["Documented", "Thing", "guarded"]


def test_each_file_is_judged_by_its_own_executed_lines():
    lines = {"a.py": set(range(15, 19)), "b.py": set(range(15, 19))}  # Thing.method removed from both
    assert base_files_to_trace({"a.py": SOURCE, "b.py": SOURCE}, lines) == {"a.py", "b.py"}  # to see whether it ran
    base = WorkerReport({}, {}, {"b.py": {18}})  # it ran in b.py only
    removed, changes = base_side_changes({"a.py": SOURCE, "b.py": SOURCE}, lines, base)
    assert removed == ["a.py::Thing.method"]
    assert changes["a.py"] == set() and {statement.owner for statement in changes["b.py"]} >= {"Thing.method"}


def test_hunks_map_to_lines_on_each_side_with_renames_deletions_and_additions():
    diff = """\
diff --git a/combat/a.py b/combat/a.py
--- a/combat/a.py
+++ b/combat/a.py
@@ -3 +3 @@ def f():
@@ -10,2 +9,0 @@ def g():
--- removed line that looks like a header
+++ added line that looks like a header
@@ -20,0 +19,3 @@
@@ -30 +32 @@
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
new file mode 100644
--- /dev/null
+++ b/combat/created.py
@@ -0,0 +1,2 @@
diff --git a/combat/__init__.py b/combat/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/combat/notes.txt b/combat/notes.txt
--- a/combat/notes.txt
+++ b/combat/notes.txt
@@ -1 +1 @@
"""
    base, head = changed_lines(diff)
    assert base == {"combat/a.py": {3, 10, 11, 30}, "combat/removed.py": {1, 2}, "combat/old.py": {5},
                    "combat/__init__.py": set()}  # an empty file's deletion has no hunk
    assert head == {"combat/a.py": {3, 19, 20, 21, 32}, "combat/new.py": {5}, "combat/created.py": {1, 2}}


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
def test_a_mangled_marker_line_fails_instead_of_skipping(line):
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


def test_failure_names_the_statement_and_side_that_never_ran():
    result = verdict(base=report(errors={"scenario": "ValueError: x"}), head=report(), base_changes={},
                     head_changes={"a.py": {RUNTIME}}, removed=[], not_compared=[])
    assert result.scenario_errors == {"base/scenario": "ValueError: x"}
    assert result.unexecuted_statements == ["head a.py::top (line 7)"]
    assert outcome_message(result) == ("Comparison failed (scenario errors: 1; changed statements that never ran: 1); "
                                       "see the report above.")


def test_import_time_changes_removed_definitions_and_other_files_are_listed_without_failing():
    result = verdict(base=report(), head=report(), base_changes={}, head_changes={"a.py": {AT_IMPORT}},
                     removed=["a.py::top"], not_compared=["bg3/x.lua"])
    assert result.passed
    assert (result.import_time_changes, result.removed_definitions, result.not_compared) == (
        ["a.py::Thing"], ["a.py::top"], ["bg3/x.lua"])
    assert outcome_message(result) == ("Comparison passed. Say in the PR how each entry in import_time_changes, "
                                       "removed_definitions, not_compared was verified.")


def test_traced_lines_are_the_ones_that_ran(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(textwrap.dedent("""\
        def f(x):
            if x:
                return 1
            y = 2
            return y
        """))
    namespace: dict = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    _, executed = with_executed_lines(tmp_path, {"module.py"}, lambda: namespace["f"](False))
    assert executed == {"module.py": {2, 4, 5}}


def test_each_worker_reports_on_its_own_tree(tmp_path):
    stub = tmp_path / "stub.py"  # stands in for the comparison script's --worker mode
    stub.write_text(textwrap.dedent("""\
        import json, sys
        args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        traced = json.load(open(args["--trace"]))
        json.dump({"fingerprints": {"tree": args["--worker"]}, "errors": {}, "executed": {p: [1] for p in traced}},
                  open(args["--output"], "w"))
        """))
    base, head = tmp_path / "base", tmp_path / "head"
    base.mkdir(), head.mkdir()
    reports = run_workers(str(stub), {"base": (base, {"a.py"}), "head": (head, set())}, tmp_path)
    assert reports["base"].fingerprints == {"tree": str(base)} and reports["head"].fingerprints == {"tree": str(head)}
    assert reports["base"].executed == {"a.py": {1}} and reports["head"].executed == {}
