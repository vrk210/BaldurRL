"""Compare semantic fingerprints of the maintained code against a Git base.

Usage: python scripts/compare_refactor.py --base COMMIT
       python scripts/compare_refactor.py --check-marker GITHUB_EVENT_JSON

CI compares only when the PR body has a line that is exactly
`Behavior: preserve`, outside code blocks and comments; --check-marker prints
true or false for that, using only the standard library. A body that merely
mentions the marker does not count; a line that is the marker with other
capitalization, punctuation or formatting fails, so a mangled declaration is not
silently skipped. A misspelled word (Behaviour, preserved) is not recognized.

Why: a PR that declares `Behavior: preserve` may rewrite tests that checked
implementation, so its tests alone cannot show that behavior is unchanged. This
script runs the same deterministic scenarios on the base and on the working
tree, with the current Python and dependencies, and compares hashes of what
they observe:

- simulator episodes for every stage and maintained policy;
- reconstruction, exact transitions and planner choices on states from those
  episodes;
- evaluation run cards and traces, trace and paired comparisons, behavior
  summaries, planning rows, value-training data, training arguments and the
  seed registry;
- the passive record path, the session runner, the mock command handshake and
  static preflight.

Timings, temporary paths and generated UUIDs are normalized away. Scenarios use
smoke seeds 0..99, never research or final-test encounters, and run no trained
model. This checks regression equivalence, not agreement with BG3.

A comparison covers only what its scenarios run, so each side traces the lines
it runs in the files the diff touches (renames are followed). Every statement
the diff removes must have run on the base, and every statement it adds or
changes must run on the working tree; otherwise the check fails and names the
statement. Three kinds of change are listed instead, for the PR to say how each
was verified:
- statements that run at import (module and class bodies, def and class lines
  with their decorators and defaults), which run whether or not their effect
  is observed;
- definitions and files the diff removes whole whose code never ran on the base,
  so their removal changes nothing the scenarios run; whether anything else
  called them is a question for `git grep`. Removed code that did run, for
  example a function moved elsewhere, is checked like any other change;
- changed files other than tests, Markdown and the maintained Python code
  (MAINTAINED_CODE).

Running a line is not testing it: a changed condition that no scenario input
tells apart still passes. Prefer adding a scenario to accepting a gap. A
refactor that renames a function the scenarios import fails on one revision; it
cannot use this check and must show equivalence another way.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
from enum import Enum
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zipfile


STAGES = ("m0", "m1a", "m1b", "m2", "m3", "m4", "m5", "m6")
MAINTAINED_CODE = ("agents/", "combat/", "evaluation/", "integration/", "train.py", "value_training.py")
# Wall-clock and clock-derived fields; they differ between any two runs.
VOLATILE_KEYS = frozenset({
    "seconds", "planned_seconds", "max_decision_seconds",
    "mean_seconds_per_decision", "mean_seconds_per_episode", "mean_seconds_per_planned_decision",
})
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Unsupported fingerprint value: {type(value)}")


def without_volatile_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_volatile_keys(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [without_volatile_keys(item) for item in value]
    return value


def digest(value: Any, *, scratch: Path, root: Path) -> str:
    """Hash a scenario result after removing timings, local paths and UUIDs."""
    text = json.dumps(without_volatile_keys(json.loads(json.dumps(value, default=jsonable))), sort_keys=True)
    for path, label in ((scratch, "<scratch>"), (root, "<repo>")):
        for spelling in (str(path), path.as_posix()):
            text = text.replace(json.dumps(spelling)[1:-1], label)
    uuids: dict[str, str] = {}
    text = UUID.sub(lambda match: uuids.setdefault(match[0], f"<uuid-{len(uuids)}>"), text)
    return hashlib.sha256(text.encode()).hexdigest()


def decision_states(stage: str) -> list[tuple[Any, Any]]:
    """Observations and masks at the first and fifth decisions of two heuristic games."""
    from agents.heuristic_agent import make_heuristic_agent
    from combat.stages import make_env

    env, policy, states = make_env(stage), make_heuristic_agent(stage), []
    for seed in (0, 1):
        observation, _ = env.reset(seed=seed)
        for decision in range(5):
            mask = env.action_masks()
            if decision in (0, 4):
                states.append((observation.copy(), mask.copy()))
            observation, _, terminated, truncated, _ = env.step(policy.choose_action(observation, mask))
            if terminated or truncated:
                break
    env.close()
    return states


def simulator_episodes(*, scratch: Path, root: Path) -> dict[str, Any]:
    """Hash 100 games per stage and maintained policy, step by step (too large to keep)."""
    from agents.heuristic_agent import make_heuristic_agent
    from agents.random_agent import RandomAgent
    from combat.stages import make_env

    results = {}
    for stage in STAGES:
        variants = ["random", "heuristic"]
        if stage in ("m5", "m6"):
            variants.append("tuned" if stage == "m5" else "adaptive")
        for variant in variants:
            steps, decisions = hashlib.sha256(), 0
            for seed in range(100):
                env = make_env(stage)
                policy = (RandomAgent(seed=seed) if variant == "random" else
                          make_heuristic_agent(stage, None if variant == "heuristic" else variant))
                observation, info = env.reset(seed=seed)
                steps.update(json.dumps([seed, observation, info], default=jsonable, sort_keys=True).encode())
                while True:
                    mask = env.action_masks()
                    action = policy.choose_action(observation.copy(), mask.copy())
                    observation, reward, terminated, truncated, info = env.step(action)
                    steps.update(json.dumps([mask, action, observation, reward, terminated, truncated, info],
                                            default=jsonable, sort_keys=True).encode())
                    decisions += 1
                    if terminated or truncated:
                        break
                env.close()
            results[f"{stage}/{variant}"] = {"decisions": decisions, "sha256": steps.hexdigest()}
    return results


def planner(*, scratch: Path, root: Path) -> dict[str, Any]:
    from agents.expectimax_agent import ExpectimaxAgent
    from agents.heuristic_agent import make_heuristic_agent
    from agents.rollout_agent import RolloutAgent
    from agents.value_functions import RaceValue
    from combat.simulation import simulation_from_observation
    from combat.tactical_transitions import TacticalTransitionModel
    from combat.transitions import transition_model

    def outcome_table(model: Any, state: Any, actions: tuple[int, ...]) -> dict[int, list[Any]]:
        table = {}
        for action in actions:
            outcomes = model.outcomes(state, action)
            successors = model.observations([successor for _, successor in outcomes])
            table[action] = sorted((round(p, 12), obs.tolist()) for (p, _), obs in zip(outcomes, successors))
        return table

    results: dict[str, Any] = {}
    for stage in STAGES:
        for index, (observation, mask) in enumerate(decision_states(stage)):
            model = transition_model(stage, observation)
            state = model.state_from_observation(observation)
            legal = model.legal_actions(state)
            replays = {}
            for action in legal:
                simulation = simulation_from_observation(stage, observation, seed=index)
                replays[action] = simulation.step(action)[:4]
                simulation.close()
            results[f"{stage}/{index}"] = {
                "legal": legal, "outcomes": outcome_table(model, state, legal), "replays": replays,
            }
            if stage in ("m5", "m6") and index == 0:
                oracle = TacticalTransitionModel.from_observation(stage, observation)
                in_turn = tuple(action for action in legal if not oracle.ends_turn(action))
                results[f"{stage}/oracle"] = outcome_table(oracle, oracle.state_from_observation(observation), in_turn)
            if stage in ("m2", "m4", "m5") and index < 2:
                agent = ExpectimaxAgent(stage, depth=1, leaf_value=RaceValue(stage))
                action = agent.choose_action(observation, mask)
                q_values = [(a, round(q, 12)) for a, q in agent.last_diagnostics.q_values]
                results[f"{stage}/expectimax/{index}"] = [action, q_values]
            if stage == "m4" and index < 2:
                rollout = RolloutAgent(stage, continuation_factory=lambda seed: make_heuristic_agent("m4"),
                                       rollouts_per_action=2, seed=0)
                results[f"{stage}/rollout/{index}"] = [rollout.choose_action(observation, mask),
                                                        rollout.last_diagnostics.scores]
    return results


def evaluation(*, scratch: Path, root: Path) -> dict[str, Any]:
    from agents.heuristic_agent import M4ThreatAwareAgent, make_heuristic_agent
    from evaluation.compare import compare_run_cards
    from evaluation.evaluate import evaluate, write_run_card
    from evaluation.paired import compare_cards
    from evaluation.planning_eval import run_chunk, summarize
    from evaluation.seeds import free_ranges, overlapping
    from evaluation.tactical_behavior import summarize_run
    from train import parse_args
    from value_training import collect

    cards = {}
    for name, variant in (("tuned", "tuned"), ("default", None)):
        trace = scratch / f"{name}.jsonl"
        summary = evaluate(make_heuristic_agent("m5", variant), range(20), trace, stage="m5")
        cards[name] = scratch / f"{name}.json"
        write_run_card(cards[name], policy=name, summary=summary, trace_file=trace.name, stage="m5")
    rows = run_chunk("m4", {"name": "threat"}, list(range(5)))
    # One agent over two encounters: per-turn caches and encounter changes.
    search_rows = run_chunk("m4", {"name": "expectimax", "leaf": "race", "depth": 1}, [0, 1])
    return {
        "run_cards": [json.loads(card.read_text(encoding="utf-8")) for card in cards.values()],
        "traces": [(scratch / f"{name}.jsonl").read_text(encoding="utf-8") for name in cards],
        "trace_comparison": compare_run_cards(cards["tuned"], cards["default"]),
        "paired": compare_cards(cards["tuned"], cards["default"]),
        "behavior": summarize_run("m5", scratch / "tuned.jsonl"),
        "planning_rows": rows,
        "planning_summary": summarize(rows),
        "expectimax_rows": search_rows,
        "value_data": collect("m4", M4ThreatAwareAgent(), range(3)),
        "training_arguments": [vars(parse_args(["--stage", stage])) for stage in STAGES],
        # Enforcement depends on bounds and status; the use labels are descriptions.
        "seed_overlaps": [(used.start, used.stop, used.status) for used in overlapping(range(220000, 224000))],
        "free_ranges": [(free.start, free.stop) for free in free_ranges()],
    }


def file_tree(directory: Path) -> list[tuple[str, Any]]:
    """Relative paths with parsed JSON content, or a byte hash for other files."""
    tree = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        content: Any = path.read_bytes()
        try:
            content = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            content = hashlib.sha256(content).hexdigest()
        tree.append((path.relative_to(directory).as_posix(), content))
    return tree


def integration(*, scratch: Path, root: Path) -> dict[str, Any]:
    from integration.inspect import inspect_directory
    from integration.mock_executor import run_demo
    from integration.mock_h0 import mock_records, write_mock_trace
    from integration.publish import publish_ready
    from integration.reader import RecordDirectoryReader
    from integration.serialization import record_to_dict
    from integration.session_runner import SessionRunner
    from integration.trace import reconstruct_intervals
    from integration.windows_preflight import run_preflight

    staging = scratch / "capture" / "session-a" / "staging"
    staging.mkdir(parents=True)
    for record in mock_records():
        body = (json.dumps(record_to_dict(record), sort_keys=True) + "\n").encode()
        base = staging / f"record-{record.sequence:08d}"
        base.with_suffix(".pending").write_bytes(body)
        base.with_suffix(".ready").write_bytes(f"{len(body)}\n".encode())
    published = publish_ready(staging, scratch / "records")
    reader = RecordDirectoryReader(scratch / "records")
    records = reader.read_all()
    # A fixed collector tree, so runbook or Lua edits don't change what this scenario observes.
    collector = scratch / "collector" / "bg3" / "h0" / "BootstrapServer.lua"
    collector.parent.mkdir(parents=True)
    collector.write_text("-- fixed collector source\n", encoding="utf-8")
    with SessionRunner(scratch / "capture", scratch / "archive", code_revision="fixed",
                       repository=scratch / "collector") as runner:
        sessions = runner.poll()
    with contextlib.redirect_stdout(io.StringIO()) as demo_output:
        command, acknowledgements = run_demo(scratch / "commands")
    return {
        "published": [path.name for path in published],
        "records": [record_to_dict(record) for record in records],
        "sequence": reader.validate_all(),
        "intervals": reconstruct_intervals(records),
        "inspection": inspect_directory(scratch / "records"),
        "mock_trace": (write_mock_trace(scratch / "mock"), file_tree(scratch / "mock")),
        "sessions": sessions,
        "archive": file_tree(scratch / "archive"),
        "demo_output": demo_output.getvalue(),
        "command": command.to_dict(),
        "acknowledgements": [acknowledgement.to_dict() for acknowledgement in acknowledgements],
        "command_tree": file_tree(scratch / "commands"),
        "preflight": run_preflight(scratch / "game", script_extender_root=scratch / "se",
                                   modsettings=scratch / "modsettings.lsx", repo_root=root,
                                   system="Windows"),
    }


SCENARIOS: dict[str, Callable[..., Any]] = {
    "simulator_episodes": simulator_episodes,
    "planner": planner,
    "evaluation": evaluation,
    "integration": integration,
}


def fingerprints(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Fingerprint every scenario against the tree at root; return (digests, errors).

    A scenario that raises compared nothing, so its error is returned for the
    caller to fail on; catching it lets the remaining scenarios still report.
    """
    sys.path.insert(0, str(root))
    results, errors = {}, {}
    for name, scenario in SCENARIOS.items():
        with tempfile.TemporaryDirectory(prefix=f"baldurrl-{name}-") as directory:
            scratch = Path(directory).resolve()
            try:
                # Printed output is observable behavior, so it is fingerprinted with the result.
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    value = scenario(scratch=scratch, root=root)
                results[name] = digest({"result": value, "stdout": printed.getvalue()}, scratch=scratch, root=root)
            except (Exception, SystemExit) as error:
                errors[name] = f"{type(error).__name__}: {error}"
    return results, errors


def with_executed_lines(root: Path, paths: set[str], run: Callable[[], Any]) -> tuple[Any, dict[str, set[int]]]:
    """Run and record which lines execute in the given repository-relative files.

    With no files to trace no hook is installed: Python 3.11 runs every frame
    slower while one is.
    """
    executed: dict[str, set[int]] = {path: set() for path in paths}
    if not paths:
        return run(), executed
    wanted = {os.path.normcase(str(root / path)): path for path in paths}
    files: dict[str, str | None] = {}

    def trace(frame: Any, event: str, _: Any) -> Any:
        filename = frame.f_code.co_filename
        if filename not in files:
            files[filename] = wanted.get(os.path.normcase(os.path.abspath(filename)))
        if files[filename] is None:
            return None
        lines = executed[files[filename]]

        def line(frame: Any, event: str, _: Any) -> Any:
            if event == "line":
                lines.add(frame.f_lineno)
            return line

        return line

    sys.settrace(trace)
    try:
        return run(), executed
    finally:
        sys.settrace(None)


@dataclasses.dataclass(frozen=True)
class Statement:
    """One executable statement: its lines, what it belongs to, and whether it runs at import.

    A def or class statement (defines=True) is named after what it defines and
    runs where it is defined. An except clause or match case counts as its own
    statement, since it runs only when reached. Any other statement is named
    after its innermost def or class, or "<module>".
    """

    first: int
    last: int
    owner: str
    runs_at_import: bool
    defines: bool = False


def is_string_only(node: ast.AST) -> bool:
    """A bare string statement, such as a docstring, runs nothing."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def statements(source: str) -> list[Statement]:
    """Every executable statement in source."""
    found = []

    def visit(node: ast.AST, prefix: str, owner: str, at_import: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.excepthandler):
                found.append(Statement(child.lineno, child.end_lineno, owner, at_import))
            elif isinstance(child, ast.match_case):
                found.append(Statement(child.pattern.lineno, child.body[-1].end_lineno, owner, at_import))
            if not isinstance(child, ast.stmt):
                visit(child, prefix, owner, at_import)
                continue
            first = min([child.lineno, *(item.lineno for item in getattr(child, "decorator_list", ()))])
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                found.append(Statement(first, child.end_lineno, name, at_import, defines=True))
                if isinstance(child, ast.ClassDef):
                    visit(child, name + ".", name, at_import)
                else:
                    visit(child, name + ".<locals>.", name, False)
            else:
                found.append(Statement(first, child.end_lineno, owner, at_import))
                visit(child, prefix, owner, at_import)

    visit(ast.parse(source), "", "<module>", True)
    return found


def changed_statements(source: str, lines: set[int]) -> set[Statement]:
    """The innermost executable statement around each changed line.

    Blank lines, comments and lines inside string-only statements change
    nothing. A compound statement spans its body, so a line in its header
    resolves to it and a line in its body to the smaller statement there.
    """
    text, found, changed = source.splitlines(), statements(source), set()
    quiet = {line for node in ast.walk(ast.parse(source)) if is_string_only(node)
             for line in range(node.lineno, node.end_lineno + 1)}
    for line in lines:
        content = text[line - 1].strip() if 0 < line <= len(text) else ""
        enclosing = [statement for statement in found if statement.first <= line <= statement.last]
        if content and not content.startswith("#") and line not in quiet and enclosing:
            changed.add(max(enclosing, key=lambda statement: (statement.first, -statement.last)))
    return changed


def split_removed_definitions(source: str, lines: set[int], executed: set[int]) -> tuple[list[str], set[Statement]]:
    """Base-side definitions the diff removes as dead code, and the changed statements outside them.

    A definition that runs at import (the module, a module-level def or class,
    a method) counts as removed when the diff changes every line of it and none
    of its runtime statements ran on the base: deleting code the scenarios
    never ran changes nothing they observe. If any of it ran, it is live code
    being removed or moved, and each statement the diff takes out of it must
    have run. executed is the base worker's executed lines for this file.
    """
    found = statements(source)
    ran = {line for statement in found if not statement.runs_at_import
           for line in range(statement.first, statement.last + 1) if line in executed}
    module = Statement(1, len(source.splitlines()), "<module>", True, defines=True)
    removed = [statement for statement in (module, *found)
               if statement.defines and statement.runs_at_import
               and all(line in lines for line in range(statement.first, statement.last + 1))
               and not any(statement.first <= line <= statement.last for line in ran)]

    def within(statement: Statement, outer: Statement) -> bool:
        return outer.first <= statement.first and statement.last <= outer.last

    outermost = sorted(statement.owner for statement in removed
                       if not any(other != statement and within(statement, other) for other in removed))
    remaining = {statement for statement in changed_statements(source, lines)
                 if not any(within(statement, other) for other in removed)}
    return outermost, remaining


def changed_lines(diff: str) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Base-side and working-tree-side lines each Python file's `git diff --unified=0` hunks touch.

    File names count only in a file's header. A deleted file with no lines, such
    as an empty __init__.py, has no hunk, so it is recorded with no lines.
    """
    base: dict[str, set[int]] = {}
    head: dict[str, set[int]] = {}
    old = new = deleted = None
    in_header = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_header, old, new = True, None, None
            deleted = re.fullmatch(r"diff --git a/(\S+\.py) b/\S+", line)
        elif in_header and line.startswith("deleted file mode") and deleted:
            base.setdefault(deleted[1], set())
        elif in_header and line.startswith("--- "):
            old = line[6:] if line.startswith("--- a/") and line.endswith(".py") else None
        elif in_header and line.startswith("+++ "):
            new = line[6:] if line.startswith("+++ b/") and line.endswith(".py") else None
            in_header = False
        elif line.startswith("@@") and not in_header:
            hunk = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk is None:
                raise ValueError(f"Unexpected diff hunk header: {line}")
            for path, start, length, sides in ((old, hunk[1], hunk[2], base), (new, hunk[3], hunk[4], head)):
                count = 1 if length is None else int(length)
                if path and count:
                    sides.setdefault(path, set()).update(range(int(start), int(start) + count))
    return base, head


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True, encoding="utf-8", errors="replace")


def declares_behavior_preserved(body: str | None) -> bool:
    """True if the PR body has a line that is exactly `Behavior: preserve`, outside code blocks and comments.

    Raises ValueError for a line that is the marker with other capitalization,
    punctuation or formatting, such as `**Behavior:** preserve` or
    `- Behavior: preserve.`; a misspelled word is not recognized.
    """
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    text = re.sub(r"^[ \t]*(```|~~~).*?^[ \t]*\1", "", text, flags=re.M | re.S)
    declared = False
    for line in text.splitlines():
        if line.startswith(("    ", "\t")):
            continue  # indented code block
        if re.fullmatch(r"Behavior: preserve\s*", line):
            declared = True
        elif re.sub(r"[^a-z]", "", line.lower()) == "behaviorpreserve":
            raise ValueError(f"PR body line {line!r} is not exactly 'Behavior: preserve'")
    return declared


@dataclasses.dataclass(frozen=True)
class WorkerReport:
    fingerprints: dict[str, str]
    errors: dict[str, str]
    executed: dict[str, set[int]]


def run_workers(script: str, trees: dict[str, tuple[Path, set[str]]], directory: Path) -> dict[str, WorkerReport]:
    """Fingerprint each (tree, traced files) in its own fresh interpreter, all at once.

    Results travel by file, so scenario output can't corrupt them. Every worker
    is waited for before any failure is raised, so none outlives the directory.
    """
    processes = {}
    for side, (tree, traced) in trees.items():
        (directory / f"{side}-traced.json").write_text(json.dumps(sorted(traced)), encoding="utf-8")
        processes[side] = subprocess.Popen([sys.executable, script, "--worker", str(tree),
                                            "--trace", str(directory / f"{side}-traced.json"),
                                            "--output", str(directory / f"{side}-result.json")], cwd=tree)
    failed = [side for side, process in processes.items() if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Comparison worker failed: {', '.join(failed)}")
    reports = {}
    for side in trees:
        report = json.loads((directory / f"{side}-result.json").read_text(encoding="utf-8"))
        reports[side] = WorkerReport(report["fingerprints"], report["errors"],
                                     {path: set(lines) for path, lines in report["executed"].items()})
    return reports


def files_to_trace(changes: dict[str, set[Statement]]) -> set[str]:
    """Files holding a changed statement the verdict checks; import-time statements are listed, not traced."""
    return {path for path, changed in changes.items() if any(not statement.runs_at_import for statement in changed)}


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The comparison report. The last three lists are for the PR to explain, not failures."""

    passed: bool
    scenario_errors: dict[str, str]
    mismatches: list[str]
    unexecuted_statements: list[str]
    import_time_changes: list[str]
    removed_definitions: list[str]
    not_compared: list[str]


def verdict(*, base: WorkerReport, head: WorkerReport, base_changes: dict[str, set[Statement]],
            head_changes: dict[str, set[Statement]], removed: list[str], not_compared: list[str]) -> Verdict:
    """Fail on a scenario error, a mismatch, or a changed statement that never ran on its side.

    base_changes must have run on the base and head_changes on the head, except
    statements that run at import, which are listed.
    """
    errors = {f"{side}/{name}": message for side, worker in (("base", base), ("head", head))
              for name, message in worker.errors.items()}
    mismatches = sorted(name for name in base.fingerprints.keys() | head.fingerprints.keys()
                        if base.fingerprints.get(name) != head.fingerprints.get(name))
    unexecuted: dict[str, list[int]] = {}
    at_import = set()
    for side, changes, worker in (("base", base_changes, base), ("head", head_changes, head)):
        for path, changed in sorted(changes.items()):
            ran = worker.executed.get(path, set())
            for statement in changed:
                if statement.runs_at_import:
                    at_import.add(f"{path}::{statement.owner}")
                elif not any(statement.first <= line <= statement.last for line in ran):
                    unexecuted.setdefault(f"{side} {path}::{statement.owner}", []).append(statement.first)
    return Verdict(
        passed=not (errors or mismatches or unexecuted),
        scenario_errors=errors,
        mismatches=mismatches,
        unexecuted_statements=[f"{name} (line {', '.join(map(str, sorted(lines)))})"
                               for name, lines in sorted(unexecuted.items())],
        import_time_changes=sorted(at_import),
        removed_definitions=sorted(removed),
        not_compared=not_compared,
    )


def outcome_message(result: Verdict) -> str:
    """One line saying why the comparison failed, or what the PR must still explain."""
    if not result.passed:
        reasons = [f"{label}: {len(items)}" for label, items in (
            ("scenario errors", result.scenario_errors), ("fingerprint mismatches", result.mismatches),
            ("changed statements that never ran", result.unexecuted_statements)) if items]
        return f"Comparison failed ({'; '.join(reasons)}); see the report above."
    listed = [key for key in ("import_time_changes", "removed_definitions", "not_compared") if getattr(result, key)]
    if listed:
        return f"Comparison passed. Say in the PR how each entry in {', '.join(listed)} was verified."
    return "Comparison passed."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="Git commit whose runtime behavior must be preserved")
    parser.add_argument("--check-marker", type=Path, metavar="GITHUB_EVENT_JSON",
                        help="print true if the event's PR body declares Behavior: preserve, else false; "
                             "exit 1 on a line that is the marker in other capitalization or formatting")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_marker:
        event = json.loads(args.check_marker.read_text(encoding="utf-8"))
        try:
            print(str(declares_behavior_preserved(event.get("pull_request", {}).get("body"))).lower())
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    if args.worker:
        root = args.worker.resolve()
        traced = set(json.loads(args.trace.read_text(encoding="utf-8")))
        (results, errors), executed = with_executed_lines(root, traced, lambda: fingerprints(root))
        report = {"fingerprints": results, "errors": errors,
                  "executed": {path: sorted(lines) for path, lines in executed.items()}}
        args.output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        return 0
    if not args.base:
        parser.error("--base or --check-marker is required")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}").strip()
    # Fixed options, so local git config can't change which lines the diff attributes to which definition.
    diff_options = ("--no-color", "--no-ext-diff", "--find-renames", "--indent-heuristic", "--diff-algorithm=myers",
                    "--src-prefix=a/", "--dst-prefix=b/", revision)
    base_lines, head_lines = changed_lines(git(root, "diff", "--unified=0", *diff_options, "--", *MAINTAINED_CODE))
    not_compared = [path for path in git(root, "diff", "--name-only", *diff_options).splitlines()
                    if not path.endswith(".md") and not path.startswith("tests/")
                    and not (path.endswith(".py") and path.startswith(MAINTAINED_CODE))]
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    with tempfile.TemporaryDirectory(prefix="baldurrl-refactor-") as directory:
        base_tree = Path(directory) / "base"
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(base_tree)
        base_sources = {path: (base_tree / path).read_text(encoding="utf-8") for path in base_lines}
        head_changes = {path: changed_statements((root / path).read_text(encoding="utf-8"), lines)
                        for path, lines in head_lines.items()}
        traced = {path: changed_statements(base_sources[path], lines) for path, lines in base_lines.items()}
        workers = run_workers(str(Path(__file__).resolve()),
                              {"base": (base_tree, files_to_trace(traced)),
                               "head": (root, files_to_trace(head_changes))}, Path(directory))
    removed, base_changes = [], {}
    for path, lines in sorted(base_lines.items()):
        names, base_changes[path] = split_removed_definitions(
            base_sources[path], lines, workers["base"].executed.get(path, set()))
        removed += [f"{path}::{name}" for name in names]
    result = verdict(base=workers["base"], head=workers["head"], base_changes=base_changes,
                     head_changes=head_changes, removed=removed, not_compared=not_compared)
    print(json.dumps({"base": revision, "fingerprints": workers["head"].fingerprints, **dataclasses.asdict(result)},
                     indent=2))
    print(outcome_message(result))
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
