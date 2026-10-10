"""Parallel seeded evaluation with per-decision timing, for planners and baselines.

    python -m evaluation.planning_eval --name x_v0_d1 --policy expectimax --leaf mlp:runs/value/v0.npz --depth 1 \\
        --seed-start 10000 --episodes 5000 --workers 5 --out runs/m4_eval/x_v0_d1.json

Seeds are split into contiguous chunks, one policy instance per chunk, so the
results do not depend on the worker count (all policies here are deterministic
given the observation). Output holds per-seed outcomes for paired comparisons.
"""

import json
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from agents.policy import Policy
from combat.stages import make_env


def make_leaf(stage: str, spec: str):
    from agents.value_functions import EnsembleValue, MLPValue, RaceValue

    if spec == "race":
        return RaceValue(stage)
    if spec.startswith("race:"):
        scale, bias = (float(x) for x in spec.split(":")[1:3])
        return RaceValue(stage, scale=scale, bias=bias)
    if spec.startswith("mlp:"):
        return MLPValue.load(spec[4:])
    if spec.startswith("ens:"):
        return EnsembleValue([MLPValue.load(path) for path in spec[4:].split(",")])
    raise ValueError(f"Unknown leaf specification {spec!r}")


def make_policy(stage: str, name: str, *, model: str | None = None, leaf: str | None = None,
                depth: int | None = 1, always_search: bool = False, node_budget: int | None = None) -> Policy:
    from agents.heuristic_agent import M4ThreatAwareAgent, make_heuristic_agent

    if name == "heuristic":
        return make_heuristic_agent(stage)
    if name == "threat":
        return M4ThreatAwareAgent()
    if name == "ppo":
        import torch
        from sb3_contrib import MaskablePPO

        from evaluation.evaluate import SB3Policy

        torch.set_num_threads(1)
        return SB3Policy(MaskablePPO.load(model, device="cpu"), deterministic=True)
    if name == "expectimax":
        from agents.expectimax_agent import ExpectimaxAgent

        leaf_value = make_leaf(stage, leaf) if leaf is not None else None
        return ExpectimaxAgent(stage, depth=depth, leaf_value=leaf_value, always_search=always_search,
                               node_budget=node_budget)
    raise ValueError(f"Unknown policy {name!r}")


def run_chunk(stage: str, spec: dict, seeds: list[int]) -> list[dict]:
    policy = make_policy(stage, **spec)
    env = make_env(stage)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        times: list[float] = []
        planned: list[float] = []
        depth_counts: dict[int, int] = {}
        reward = 0.0
        terminated = False
        info: dict = {}
        while not done:
            mask = env.action_masks()
            tick = perf_counter()
            action = policy.choose_action(obs, mask)
            elapsed = perf_counter() - tick
            times.append(elapsed)
            if mask.sum() > 1:
                planned.append(elapsed)
                used = getattr(getattr(policy, "last_diagnostics", None), "depth_used", None)
                if used is not None:
                    depth_counts[used] = depth_counts.get(used, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append(dict(seed=seed, won=bool(terminated and reward > 0), lost=bool(terminated and reward < 0),
                         truncated=bool(truncated), rounds=int(info["round"]), decisions=len(times),
                         seconds=float(sum(times)), planned_decisions=len(planned),
                         planned_seconds=float(sum(planned)), max_decision_seconds=float(max(times)),
                         depth_counts={str(k): v for k, v in sorted(depth_counts.items())}))
    return rows


def summarize(rows: list[dict]) -> dict:
    wins = np.array([row["won"] for row in rows], dtype=float)
    n = len(rows)
    decisions = sum(row["decisions"] for row in rows)
    planned = sum(row["planned_decisions"] for row in rows)
    return dict(
        episodes=n, wins=int(wins.sum()), win_rate=float(wins.mean()),
        win_rate_se=float(sqrt(wins.mean() * (1 - wins.mean()) / n)),
        losses=sum(row["lost"] for row in rows), truncations=sum(row["truncated"] for row in rows),
        mean_rounds=float(np.mean([row["rounds"] for row in rows])),
        decisions=decisions, planned_decisions=planned,
        mean_seconds_per_decision=sum(row["seconds"] for row in rows) / decisions,
        mean_seconds_per_planned_decision=(sum(row["planned_seconds"] for row in rows) / planned) if planned else 0.0,
        max_decision_seconds=max(row["max_decision_seconds"] for row in rows),
        mean_seconds_per_episode=sum(row["seconds"] for row in rows) / n,
        depth_counts=_merge_counts(row.get("depth_counts", {}) for row in rows),
    )


def _merge_counts(counts) -> dict[str, int]:
    merged: dict[str, int] = {}
    for count in counts:
        for key, value in count.items():
            merged[key] = merged.get(key, 0) + value
    return dict(sorted(merged.items()))


def paired_difference(a: list[dict], b: list[dict]) -> dict:
    """Win-rate difference a - b over shared seeds, in percentage points, with a 95% CI."""
    wins_b = {row["seed"]: row["won"] for row in b}
    diff = np.array([float(row["won"]) - float(wins_b[row["seed"]]) for row in a if row["seed"] in wins_b])
    se = diff.std(ddof=1) / sqrt(len(diff))
    return dict(seeds=len(diff), delta_pp=100 * diff.mean(), paired_se_pp=100 * se,
                ci95_pp=[100 * (diff.mean() - 1.96 * se), 100 * (diff.mean() + 1.96 * se)])


def evaluate_parallel(stage: str, spec: dict, seeds: list[int], workers: int, chunks: int | None = None) -> list[dict]:
    chunks = chunks or max(1, workers * 4)
    parts = [list(part) for part in np.array_split(np.array(seeds), chunks) if len(part)]
    parts = [[int(seed) for seed in part] for part in parts]
    if workers <= 1:
        return [row for part in parts for row in run_chunk(stage, spec, part)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_chunk, [stage] * len(parts), [spec] * len(parts), parts))
    return [row for part in results for row in part]


def main() -> None:
    parser = ArgumentParser(description="Parallel seeded evaluation with decision timing")
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", default="m4")
    parser.add_argument("--policy", required=True, choices=["heuristic", "threat", "ppo", "expectimax"])
    parser.add_argument("--model")
    parser.add_argument("--leaf")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = dict(name=args.policy, model=args.model, leaf=args.leaf, depth=args.depth, node_budget=args.node_budget)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    tick = perf_counter()
    rows = evaluate_parallel(args.stage, spec, seeds, args.workers)
    summary = summarize(rows)
    summary["wall_seconds"] = perf_counter() - tick
    payload = dict(name=args.name, stage=args.stage, spec=spec, seed_start=args.seed_start,
                   episodes=args.episodes, summary=summary, results=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(dict(name=args.name, **summary)))


if __name__ == "__main__":
    main()
