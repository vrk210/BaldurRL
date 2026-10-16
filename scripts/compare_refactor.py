"""Compare semantic fingerprints of the maintained code against a Git base.

Usage: python scripts/compare_refactor.py --base COMMIT [--event GITHUB_EVENT_JSON]

With --event, as in CI, it runs only if the PR body has a line that is exactly
`Behavior: preserve`; a body that merely mentions the marker does not count.

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
- the passive record path, the session runner, the mock command handshake,
  encounter-trial analysis and static preflight.

Timings, temporary paths and generated UUIDs are normalized away. Scenarios use
smoke seeds 0..99, never research or final-test encounters, and run no trained
model. This checks regression equivalence, not agreement with BG3.

A comparison covers only the code its scenarios run. On the working tree, the
script records every called function and checks the functions the diff changed.
A changed module none of whose changed functions ran fails the check. Other
changed functions that never ran, and module-level changes (which run at import
whether or not their effect is observed), are listed so the PR can say how they
were verified. Prefer adding a scenario to accepting a gap. A refactor that
renames a function the scenarios import fails on one revision; it cannot use
this check and must show equivalence another way.
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


def simulator_episodes(scratch: Path, root: Path) -> dict[str, Any]:
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


def planner(scratch: Path, root: Path) -> dict[str, Any]:
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


def evaluation(scratch: Path, root: Path) -> dict[str, Any]:
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


def integration(scratch: Path, root: Path) -> dict[str, Any]:
    from integration.encounter_trial import analyze_file
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
    with SessionRunner(scratch / "capture", scratch / "archive", code_revision="fixed",
                       repository=root) as runner:
        sessions = runner.poll()
    with contextlib.redirect_stdout(io.StringIO()) as demo_output:
        command, acknowledgements = run_demo(scratch / "commands")
    trial = root / "bg3" / "trials" / "synthetic_human_attack"
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
        "trial": analyze_file(trial / "trial.json", trial / "records"),
        "preflight": run_preflight(scratch / "game", script_extender_root=scratch / "se",
                                   modsettings=scratch / "modsettings.lsx", repo_root=root,
                                   system="Windows"),
    }


SCENARIOS: dict[str, Callable[[Path, Path], Any]] = {
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
                    value = scenario(scratch, root)
                results[name] = digest({"result": value, "stdout": printed.getvalue()}, scratch=scratch, root=root)
            except (Exception, SystemExit) as error:
                errors[name] = f"{type(error).__name__}: {error}"
    return results, errors


def with_called_functions(root: Path, run: Callable[[], Any]) -> tuple[Any, set[tuple[str, str]]]:
    """Run and record (relative path, qualified name) of every function called under root."""
    prefix, called = str(root) + os.sep, set()

    def profile(frame: Any, event: str, _: Any) -> None:
        if event == "call" and frame.f_code.co_filename.startswith(prefix):
            path = frame.f_code.co_filename[len(prefix):].replace(os.sep, "/")
            called.add((path, frame.f_code.co_qualname))

    sys.setprofile(profile)
    try:
        return run(), called
    finally:
        sys.setprofile(None)


DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def changed_definitions(source: str, lines: set[int]) -> set[str]:
    """Qualified names of the innermost def or class around each line; "<module>" for module statements.

    Blank lines, comments and string-only statements outside definitions run
    nothing, so they are skipped rather than reported as module-level changes.
    Line 0 is a deletion before the first line, such as emptying a module.
    """
    tree = ast.parse(source)
    spans = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFINITIONS):
                name = prefix + child.name
                start = min([child.lineno, *(item.lineno for item in child.decorator_list)])
                spans.append((start, child.end_lineno, name))
                visit(child, name + ("." if isinstance(child, ast.ClassDef) else ".<locals>."))
            else:
                visit(child, prefix)

    visit(tree, "")
    statements = [(node.lineno, node.end_lineno) for node in tree.body
                  if not isinstance(node, DEFINITIONS)
                  and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                           and isinstance(node.value.value, str))]
    names = set()
    for line in lines:
        enclosing = [span for span in spans if span[0] <= line <= span[1]]
        if enclosing:
            names.add(max(enclosing)[2])
        elif line == 0 or any(first <= line <= last for first, last in statements):
            names.add("<module>")
    return names


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Working-tree line numbers each Python file's diff hunks touch.

    A pure deletion is attributed to the line before it, which is inside the
    function that lost the lines unless the deletion follows a function's end.
    """
    lines: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") and line.endswith(".py") else None
        elif line.startswith("@@") and path:
            hunk = re.match(r"@@ -\S+ \+(\d+)(,(\d+))? @@", line)
            if hunk is None:
                raise ValueError(f"Unexpected diff hunk header: {line}")
            first, length = int(hunk[1]), 1 if hunk[3] is None else int(hunk[3])
            lines.setdefault(path, set()).update(range(first, first + length) if length else {first})
    return lines


def changed_functions(root: Path, base: str) -> set[tuple[str, str]]:
    """(path, enclosing definition) of every line the diff from base touches."""
    diff = subprocess.check_output(["git", "diff", "--unified=0", "--no-color", "--no-ext-diff",
                                    "--src-prefix=a/", "--dst-prefix=b/", base, "--", *MAINTAINED_CODE],
                                   cwd=root, text=True, encoding="utf-8", errors="replace")
    return {(path, name) for path, numbers in changed_lines(diff).items()
            for name in changed_definitions((root / path).read_text(encoding="utf-8"), numbers)}


def declares_behavior_preserved(body: str | None) -> bool:
    """True if the PR body has a line that is exactly `Behavior: preserve`."""
    return re.search(r"^Behavior: preserve\s*$", body or "", re.M) is not None


def run_worker(script: str, tree: Path, *, record_calls: bool) -> dict[str, Any]:
    """Fingerprint tree in a fresh interpreter; results travel by file, so scenario output can't corrupt them."""
    with tempfile.TemporaryDirectory(prefix="baldurrl-worker-") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, script, "--worker", str(tree), "--output", str(output)]
        subprocess.run([*command, *(["--record-calls"] if record_calls else [])], cwd=tree, check=True)
        return json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git commit whose runtime behavior must be preserved")
    parser.add_argument("--event", type=Path, help="GitHub event JSON; compare only if its PR body declares preservation")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record-calls", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        root = args.worker.resolve()
        if args.record_calls:
            (results, errors), called = with_called_functions(root, lambda: fingerprints(root))
        else:
            (results, errors), called = fingerprints(root), set()
        report = {"fingerprints": results, "errors": errors, "called": sorted(called)}
        args.output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        return 0
    if not args.base:
        parser.error("--base is required")
    if args.event:
        event = json.loads(args.event.read_text(encoding="utf-8"))
        if not declares_behavior_preserved(event.get("pull_request", {}).get("body")):
            print("The PR body has no 'Behavior: preserve' line; comparison skipped.")
            return 0
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}"],
                                       cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="baldurrl-refactor-") as directory:
        baseline = Path(directory)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(baseline)
        before = run_worker(script, baseline, record_calls=False)
    after = run_worker(script, root, record_calls=True)
    errors = {f"{side}/{name}": message for side, worker in (("base", before), ("head", after))
              for name, message in worker["errors"].items()}
    mismatches = sorted(name for name in before["fingerprints"].keys() | after["fingerprints"].keys()
                        if before["fingerprints"].get(name) != after["fingerprints"].get(name))
    called = {tuple(item) for item in after["called"]}
    changed = changed_functions(root, revision)
    # Module-level lines run at import whether or not their effect is observed, so they never count as exercised.
    in_functions = {item for item in changed if item[1] != "<module>"}
    exercised_modules = {path for path, _ in in_functions & called}
    unexercised_modules = sorted({path for path, _ in in_functions} - exercised_modules)
    print(json.dumps({
        "base": revision,
        "fingerprints": after["fingerprints"],
        "scenario_errors": errors,
        "mismatches": mismatches,
        "unexercised_modules": unexercised_modules,
        "unexecuted_changed_functions": [f"{path}::{name}" for path, name in sorted(in_functions - called)],
        "module_level_changes": sorted(path for path, name in changed if name == "<module>"),
    }, indent=2))
    return int(bool(errors or mismatches or unexercised_modules))


if __name__ == "__main__":
    raise SystemExit(main())
