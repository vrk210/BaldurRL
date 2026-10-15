"""Paired, parallel headroom experiments with streamed decision diagnostics.

python -m evaluation.headroom_probe run --stage m5 --phase pilot --seed-start 220010 \\
    --episodes 4 --workers 2 --out runs/headroom_probe/pilot_m5
python -m evaluation.headroom_probe recheck --run runs/headroom_probe/confirm_m5
"""
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from math import sqrt, exp, lgamma, log, log1p
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from agents.heuristic_agent import make_heuristic_agent
from agents.improved_heuristic_agent import ImprovedHeuristicAgent, paired_pvalue
from combat.stages import make_env
from evaluation.paired import paired_comparison

RANGES = {"pilot": (220000, 220999), "selection": (221000, 222999),
          "confirmation": (223000, 227999), "spare": (228000, 229999)}


def paired_summary(first: dict[int, int], second: dict[int, int]) -> dict:
    """Exact McNemar and conservative boundary-aware >=95% paired CI.

    Bonferroni combines 97.5% Clopper-Pearson intervals for the two discordant
    probabilities. This preserves coverage at zero discordances; a normal
    paired interval is also supplied for precision, explicitly approximate.
    """
    result = paired_comparison(first, second)
    n = result["episodes"]
    plus, minus = result["first_only_wins"], result["second_only_wins"]
    def cdf(k: int, probability: float) -> float:
        if probability <= 0:
            return 1.0
        if probability >= 1:
            return float(k >= n)
        return min(1.0, sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                                + i * log(probability) + (n - i) * log1p(-probability))
                            for i in range(k + 1)))
    def invert(k: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = (lo + hi) / 2
            if cdf(k, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    def bounds(k):
        return (invert(k - 1, .9875) if k else 0.0,
                invert(k, .0125) if k < n else 1.0)
    lo_p, hi_p = bounds(plus)
    lo_m, hi_m = bounds(minus)
    result["ci95_conservative"] = [lo_p - hi_m, hi_p - lo_m]
    result["ci95_normal_approx"] = [result["difference"] - 1.96 * result["paired_se"],
                                    result["difference"] + 1.96 * result["paired_se"]]
    result["mcnemar_exact_two_sided_p"] = min(1.0, 2 * paired_pvalue(max(plus, minus), min(plus, minus)))
    return result


def run_episode(job: dict) -> dict:
    stage, ordinal, encounter = job["stage"], job["ordinal"], job["encounter"]
    # Ordinal is independent of the encounter's numeric seed; worker scheduling
    # changes neither the planner stream nor the paired result.
    planner_seed = int(np.random.SeedSequence([job["agent_seed"], int(stage[1:]), ordinal])
                       .generate_state(1, dtype=np.uint64)[0])
    agent = ImprovedHeuristicAgent(stage, seed=planner_seed, alpha=job["alpha"],
                                  looks=tuple(job["looks"]), futility_margin=job["futility_margin"],
                                  prune_nonpositive=job.get("prune_nonpositive", False))
    heuristic = make_heuristic_agent(stage, "tuned" if stage == "m5" else "adaptive")
    env = make_env(stage)
    cpu = process_time()
    out = Path(job["out"])
    obs, _ = env.reset(seed=encounter)
    done, step, deviations, nonforced, planning, rollouts = False, 0, 0, 0, 0.0, 0
    stops = Counter()
    with (out / f"decisions_{encounter}.jsonl").open("x") as handle:
        while not done:
            mask = env.action_masks()
            action = agent.choose_action(obs, mask)
            diag = agent.last_diagnostics
            row = asdict(diag)
            row.update(encounter=encounter, step=step)
            if action != diag.anchor:
                row["observation"] = obs.tolist()
                row["action_mask"] = mask.tolist()
                deviations += 1
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            nonforced += len(diag.legal) > 1
            planning += diag.planning_seconds
            rollouts += diag.simulated_episodes
            stops[diag.stopped] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    won, rounds = bool(terminated and reward == 1), int(info["round"])
    obs, _ = env.reset(seed=encounter)
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(heuristic.choose_action(obs, env.action_masks()))
        done = terminated or truncated
    baseline_won = bool(terminated and reward == 1)
    env.close()
    result = dict(encounter=encounter, ordinal=ordinal, planner_seed=planner_seed, won=won,
                  heuristic_won=baseline_won, decisions=step, nonforced_decisions=nonforced,
                  deviations=deviations, rounds=rounds, planning_seconds=planning,
                  simulated_episodes=rollouts, cpu_seconds=process_time() - cpu, stops=dict(stops))
    (out / f"episode_{encounter}.json").write_text(json.dumps(result) + "\n")
    return result


def summarize(out: Path, rows: list[dict]) -> dict:
    paired = paired_summary({r["encounter"]: int(r["won"]) for r in rows},
                            {r["encounter"]: int(r["heuristic_won"]) for r in rows})
    names = make_env(json.loads((out / "manifest.json").read_text())["stage"]).action_names
    types = defaultdict(list)
    for path in sorted(out.glob("decisions_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["anchor"] != row["chosen"]:
                scores = dict(row["scores"])
                types[f'{names[row["anchor"]]} -> {names[row["chosen"]]}'].append(scores[row["chosen"]] - scores[row["anchor"]])
    decisions, nonforced = sum(r["decisions"] for r in rows), sum(r["nonforced_decisions"] for r in rows)
    deviations = sum(r["deviations"] for r in rows)
    return dict(paired=paired, decisions=decisions, nonforced_decisions=nonforced,
                deviations=deviations, deviation_rate=deviations / decisions,
                deviation_rate_nonforced=deviations / nonforced if nonforced else 0,
                fights_with_deviation=sum(r["deviations"] > 0 for r in rows),
                fight_deviation_rate=sum(r["deviations"] > 0 for r in rows) / len(rows),
                cpu_seconds=sum(r["cpu_seconds"] for r in rows),
                planning_seconds=sum(r["planning_seconds"] for r in rows),
                seconds_per_decision=sum(r["planning_seconds"] for r in rows) / decisions,
                simulated_episodes=sum(r["simulated_episodes"] for r in rows),
                deviations_by_type={k: dict(count=len(v), mean_estimated_gain=float(np.mean(v))) for k, v in sorted(types.items())})


def recheck(run: Path, samples: int, budget: int, seed: int) -> None:
    config = json.loads((run / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    # Uniform sample over all deviations; fresh stream never used for selection.
    reservoir, total = [], 0
    for path in sorted(run.glob("decisions_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["anchor"] == row["chosen"]:
                continue
            total += 1
            if len(reservoir) < samples:
                reservoir.append(row)
            else:
                index = int(rng.integers(total))
                if index < samples:
                    reservoir[index] = row
    cpu = process_time()
    agent = ImprovedHeuristicAgent(config["stage"])
    results = []
    for row in reservoir:
        obs = np.asarray(row["observation"], dtype=np.float32)
        seeds = rng.integers(0, 2**63 - 1, size=budget)
        anchor = np.array([agent._outcome(obs, row["anchor"], int(s)) for s in seeds])
        chosen = np.array([agent._outcome(obs, row["chosen"], int(s)) for s in seeds])
        diff = chosen - anchor
        results.append(dict(encounter=row["encounter"], step=row["step"], anchor=row["anchor"],
                            chosen=row["chosen"], paired_difference=float(diff.mean()),
                            paired_se=float(diff.std(ddof=1) / sqrt(budget)),
                            chosen_only_wins=int(sum(diff > 0)), anchor_only_wins=int(sum(diff < 0))))
    payload = dict(seed=seed, samples=samples, budget=budget, total_deviations=total, results=results,
                   cpu_seconds=process_time() - cpu)
    with (run / "recheck.json").open("x") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=("m5", "m6"), required=True)
    run.add_argument("--phase", choices=RANGES, required=True)
    run.add_argument("--seed-start", type=int, required=True)
    run.add_argument("--episodes", type=int, required=True)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--agent-seed", type=int, default=20261015)
    run.add_argument("--alpha", type=float, default=0.01)
    run.add_argument("--looks", type=int, nargs="+", default=[16, 64, 256])
    run.add_argument("--futility-margin", type=float, default=0.0)
    run.add_argument("--prune-nonpositive", action="store_true")
    run.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("recheck")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--samples", type=int, default=12)
    check.add_argument("--budget", type=int, default=1024)
    check.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.command == "recheck":
        recheck(args.run, args.samples, args.budget, args.seed)
        return
    lo, hi = RANGES[args.phase]
    if args.episodes < 1 or args.workers < 1 or not lo <= args.seed_start <= args.seed_start + args.episodes - 1 <= hi:
        parser.error("episodes/workers must be positive and encounter range inside declared phase")
    ImprovedHeuristicAgent(args.stage, alpha=args.alpha, looks=tuple(args.looks), futility_margin=args.futility_margin)
    args.out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["out"] = str(args.out)
    (args.out / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    jobs = [dict(config, encounter=args.seed_start + i, ordinal=i) for i in range(args.episodes)]
    tick = perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_episode, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), **row)), flush=True)
    result = summarize(args.out, rows)
    result["wall_seconds"] = perf_counter() - tick
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
