"""Diff-to-definition mapping that decides whether a refactor comparison ran a change."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_refactor", Path(__file__).resolve().parents[1] / "scripts" / "compare_refactor.py")
compare_refactor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_refactor)

SOURCE = """\
\"\"\"Module docstring.\"\"\"
import os
# A comment.

@decorator
def top():
    return 1


class Thing:
    value = 2

    def method(self):
        def inner():
            return 3
        return inner()
"""


@pytest.mark.parametrize("line, names", [
    (1, set()),  # module docstring runs nothing
    (2, {"<module>"}),
    (3, set()),  # comment
    (5, {"top"}),  # a decorator belongs to the function it decorates
    (7, {"top"}),
    (11, {"Thing"}),  # class body
    (13, {"Thing.method"}),
    (15, {"Thing.method.<locals>.inner"}),  # matches the code object's co_qualname
    (16, {"Thing.method"}),
])
def test_changed_line_maps_to_innermost_definition(line, names):
    assert compare_refactor.changed_definitions(SOURCE, {line}) == names


def test_hunks_map_to_working_tree_lines_of_python_files_only():
    diff = """\
--- a/combat/a.py
+++ b/combat/a.py
@@ -3 +3 @@ def f():
@@ -10,2 +10,0 @@ def g():
@@ -20,0 +19,3 @@
--- a/combat/removed.py
+++ /dev/null
@@ -1,2 +0,0 @@
--- a/combat/notes.txt
+++ b/combat/notes.txt
@@ -1 +1 @@
"""
    # A pure deletion is attributed to the line before it (10).
    assert compare_refactor.changed_lines(diff) == {"combat/a.py": {3, 10, 19, 20, 21}}
