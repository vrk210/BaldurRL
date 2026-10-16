"""Random full-trace PPO disagreement samples stratified by trained model.

Samples are uniform without replacement within model/disagreement strata.
Population counts weight estimates despite equal sample quotas. Local values
use heuristic continuation; they are not a causal decomposition of PPO's gap.
"""

import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from agents.heuristic_agent import make_heuristic_agent
from combat.simulation import simulation_from_observation
from combat.stages import make_env


Kind = tuple[str, str]


@dataclass
class Reservoir:
    capacity: int
    count: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, item: dict[str, Any], rng: np.random.Generator) -> None:
        self.count += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = int(rng.integers(self.count))
            if index < self.capacity:
                self.items[index] = item


def collect_samples(stage: str, variant: str, root: Path, per_model: int,
                    seed: int, top_k: int) -> dict[str, Any]:
    heuristic = make_heuristic_agent(stage, variant)
    env = make_env(stage)
    labels = env.action_names
    env.close()

    def family(index: int) -> str:
        name = labels[index]
        return name.rsplit("_ENEMY_", 1)[0] if name.startswith(("ATTACK", "TRIP")) else name

    counts: Counter[Kind] = Counter()
    strata: dict[tuple[int, Kind], Reservoir] = {}
    episodes = decisions = 0
    for model in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(stage[1:]), model]))
        path = root / stage / f"ppo_s{model}" / "episodes.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                episode = json.loads(line)
                episodes += 1
                for step_index, step in enumerate(episode["steps"]):
                    mask = np.asarray(step["action_mask"], dtype=bool)
                    if mask.sum() == 1:
                        continue
                    decisions += 1
                    obs = np.asarray(step["observation"], dtype=np.float32)
                    h, p = heuristic.choose_action(obs, mask), step["action_index"]
                    if h == p:
                        continue
                    kind = ((family(h), family(p)) if family(h) != family(p)
                            else (labels[h], "other " + family(p)))
                    counts[kind] += 1
                    reservoir = strata.setdefault((model, kind), Reservoir(per_model))
                    reservoir.add({
                        "model_seed": model, "episode_seed": episode["result"]["seed"],
                        "step_index": step_index, "observation": step["observation"],
                        "action_mask": step["action_mask"], "heuristic_action": h,
                        "ppo_action": p, "heuristic_label": labels[h], "ppo_label": labels[p],
                    }, rng)
        print(f"{stage}: scanned all 5,000 encounters for PPO s{model}", flush=True)

    kinds = []
    for index, (kind, count) in enumerate(counts.most_common(top_k)):
        by_model = []
        for model in range(3):
            reservoir = strata.get((model, kind), Reservoir(per_model))
            by_model.append({"model_seed": model, "population_count": reservoir.count,
                             "samples": sorted(reservoir.items,
                                               key=lambda s: (s["episode_seed"], s["step_index"]))})
        kinds.append({"kind_index": index, "kind": list(kind), "count": count,
                      "frequency_per_fight": count / episodes, "strata": by_model})
    return {"stage": stage, "variant": variant, "episodes": episodes,
            "nonforced_decisions": decisions, "disagreements": sum(counts.values()),
            "sample_seed": seed, "samples_per_model_per_kind": per_model,
            "sampling": "uniform reservoir per model and disagreement kind, full trace scan",
            "taxonomy": "original classification, including target-slot-specific swap kinds",
            "kinds": kinds}


def score_state(job: tuple[str, str, int, int, int, dict[str, Any]]) -> dict[str, Any]:
    stage, variant, budget, seed, kind_index, sample = job
    heuristic = make_heuristic_agent(stage, variant)
    obs = np.asarray(sample["observation"], dtype=np.float32)
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, int(stage[1:]), kind_index, sample["model_seed"],
         sample["episode_seed"], sample["step_index"]]))
    seeds = rng.integers(0, 2**63 - 1, size=budget)

    def outcomes(action: int) -> np.ndarray:
        wins = np.zeros(budget, dtype=np.int8)
        for index, simulation_seed in enumerate(seeds):
            env = simulation_from_observation(stage, obs, seed=int(simulation_seed))
            try:
                observation, reward, terminated, truncated, _ = env.step(action)
                while not (terminated or truncated):
                    action_index = heuristic.choose_action(observation, env.action_masks())
                    observation, reward, terminated, truncated, _ = env.step(action_index)
                wins[index] = terminated and reward == 1.0
            finally:
                env.close()
        return wins

    h = outcomes(sample["heuristic_action"])
    p = outcomes(sample["ppo_action"])
    differences = h - p
    variance = float(np.var(differences, ddof=1) / budget)
    return {**sample, "kind_index": kind_index, "rollouts_per_action": budget,
            "heuristic_wins": int(h.sum()), "ppo_wins": int(p.sum()),
            "heuristic_only_wins": int(np.sum(differences == 1)),
            "ppo_only_wins": int(np.sum(differences == -1)),
            "difference": float(differences.mean()), "mc_variance": variance,
            "paired_mc_se": sqrt(variance)}


def summarize_strata(populations: dict[int, int], scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(populations.values())
    if not total:
        raise ValueError("No disagreement population")
    estimate = variance = 0.0
    by_model = []
    for model, population in sorted(populations.items()):
        selected = sorted((s for s in scores if s["model_seed"] == model),
                          key=lambda s: (s["episode_seed"], s.get("step_index", 0)))
        n = len(selected)
        if not population:
            if n:
                raise ValueError("Samples in an empty population")
            by_model.append({"model_seed": model, "population_count": 0, "sample_count": 0})
            continue
        if not n or n > population or (n == 1 and population > 1):
            raise ValueError("Insufficient or inconsistent stratum samples")
        effects = np.asarray([s["difference"] for s in selected])
        mc_variance = float(np.mean([s["mc_variance"] for s in selected]))
        # Finite-population sampling variance plus independent Monte Carlo noise.
        sample_variance = float(np.var(effects, ddof=1)) if n > 1 else 0.0
        stratum_variance = (1 - n / population) * sample_variance / n + mc_variance / population
        weight = population / total
        term = weight**2 * stratum_variance
        estimate += weight * float(effects.mean())
        variance += term
        by_model.append({"model_seed": model, "population_count": population,
                         "weight": weight, "sample_count": n,
                         "mean_difference": float(effects.mean()),
                         "raw_heuristic_better_count": int(np.sum(effects > 0)),
                         "episode_seed_min": min(s["episode_seed"] for s in selected),
                         "episode_seed_max": max(s["episode_seed"] for s in selected),
                         "distinct_episodes": len({s["episode_seed"] for s in selected})})
    se = sqrt(variance)
    critical = 1.96
    return {"weighted_mean_difference": estimate, "stratified_se": se,
            "approximate_95pct_ci": [estimate - critical * se, estimate + critical * se],
            "sample_count": len(scores),
            "raw_heuristic_better_count": sum(s["difference"] > 0 for s in scores),
            "by_model": by_model}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["m5", "m6"])
    parser.add_argument("variant")
    parser.add_argument("top_k", type=int, nargs="?", default=6)
    parser.add_argument("--samples-per-model", type=int, default=12)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=20261010)
    parser.add_argument("--rollout-seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trace-root", type=Path, default=Path("runs/v2"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/v2/analysis/stratified"))
    args = parser.parse_args()
    if args.samples_per_model < 2 or args.budget < 2 or args.top_k < 1 or args.workers < 1:
        parser.error("samples and budget must be >=2; top_k and workers must be positive")
    manifest = collect_samples(args.stage, args.variant, args.trace_root,
                               args.samples_per_model, args.sample_seed, args.top_k)
    output = args.output_root / args.stage
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(args.stage, args.variant, args.budget, args.rollout_seed, kind["kind_index"], sample)
            for kind in manifest["kinds"] for stratum in kind["strata"] for sample in stratum["samples"]]
    scored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_state, job) for job in jobs]
        with (output / "state_scores.jsonl").open("w", encoding="utf-8") as handle:
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                scored[result["kind_index"]].append(result)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                if index % 12 == 0 or index == len(jobs):
                    print(f"{args.stage}: rescored {index}/{len(jobs)} sampled states", flush=True)
    summaries = []
    lines = [f"{args.stage}: {manifest['episodes']} fights, {len(jobs)} stratified random states; "
             f"{args.budget} common-seed rollouts/action; heuristic continuation"]
    for kind in manifest["kinds"]:
        populations = {s["model_seed"]: s["population_count"] for s in kind["strata"]}
        summary = {**{k: v for k, v in kind.items() if k != "strata"},
                   **summarize_strata(populations, scored[kind["kind_index"]])}
        summaries.append(summary)
        low, high = summary["approximate_95pct_ci"]
        lines.append(f"{kind['kind'][0]} -> {kind['kind'][1]}: "
                     f"{kind['frequency_per_fight']:.2f}/fight; n={summary['sample_count']}; "
                     f"weighted mean {100*summary['weighted_mean_difference']:+.2f} pts "
                     f"(SE {100*summary['stratified_se']:.2f}; approximate 95% CI "
                     f"[{100*low:+.2f}, {100*high:+.2f}])")
    report = {**{k: v for k, v in manifest.items() if k != "kinds"},
              "budget": args.budget, "rollout_seed": args.rollout_seed,
              "continuation": args.variant, "uncertainty": "stratified sampling plus paired Monte Carlo; approximate normal interval",
              "caveat": "Local values under heuristic continuation are not additive or whole-fight causal attribution.",
              "kinds": summaries}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
