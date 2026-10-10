"""Reproduce report results without overwriting the original artifacts.

Run from the repository root with the project's Python environment:
  PYTHONPATH=. /path/to/python reports/review_m5_m6.py m5
  PYTHONPATH=. /path/to/python reports/review_m5_m6.py m6
"""
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from agents.random_agent import RandomAgent
from agents.tactical_heuristics import VARIANTS, make_tactical_heuristic
from combat.tactical_env import tactical_observation_fields
from evaluation.evaluate import SB3Policy, evaluate
from evaluation.paired import paired_comparison, wins_from_card


def main(stage: str) -> None:
    torch.set_num_threads(1)
    output = Path("runs/review") / stage
    output.mkdir(parents=True, exist_ok=True)
    best = "tuned" if stage == "m5" else "adaptive"
    results = {}
    policies = {"random": RandomAgent(0)}
    policies.update({v: make_tactical_heuristic(stage, v) for v in VARIANTS[stage]})
    for seed in range(3):
        model = MaskablePPO.load(f"runs/{stage}_ppo_1m_s{seed}/final_model.zip", device="cpu")
        policies[f"ppo_s{seed}"] = SB3Policy(model)
    for name, policy in policies.items():
        started = perf_counter()
        summary = evaluate(policy, range(3000, 8000), stage=stage)
        saved = json.loads(Path(f"runs/v2/{stage}/{name}/run_card.json").read_text())
        wins = {r.seed: int(r.won) for r in summary.results}
        old_wins = wins_from_card(saved)
        differences = {k: [saved["summary"].get(k), v] for k, v in summary.as_dict().items()
                       if saved["summary"].get(k) != v}
        result = {"summary": summary.as_dict(), "wins_by_seed": wins,
                  "different_outcome_seeds": [s for s in wins if wins[s] != old_wins[s]],
                  "summary_differences": differences, "seconds": perf_counter() - started}
        results[name] = result
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{stage}/{name}: {summary.wins}/5000; outcome mismatches="
              f"{len(result['different_outcome_seeds'])}; summary mismatches={list(differences)}; "
              f"{result['seconds']:.1f}s", flush=True)
    pooled = {s: sum(results[f"ppo_s{i}"]["wins_by_seed"][s] for i in range(3))/3
              for s in range(3000, 8000)}
    pairs = {name: paired_comparison(r["wins_by_seed"], results[best]["wins_by_seed"])
             for name, r in results.items() if name != best}
    pairs["ppo_pooled"] = paired_comparison(pooled, results[best]["wins_by_seed"])
    (output / "paired.json").write_text(json.dumps(pairs, indent=2) + "\n")

    # An untouched seed range: choose no new thresholds on these results.
    fresh_names = list(VARIANTS[stage]) + ["ppo_s0"]
    fresh = {}
    for name in fresh_names:
        summary = evaluate(policies[name], range(40000, 45000), stage=stage)
        fresh[name] = {"summary": summary.as_dict(),
                       "wins_by_seed": {r.seed: int(r.won) for r in summary.results}}
        print(f"fresh {stage}/{name}: {summary.wins}/5000", flush=True)
    fresh["paired_to_best"] = {name: paired_comparison(r["wins_by_seed"], fresh[best]["wins_by_seed"])
                               for name, r in fresh.items() if name != best}
    (output / "fresh.json").write_text(json.dumps(fresh, indent=2) + "\n")

    if stage == "m5":
        heuristic = policies[best]
        ppo = policies["ppo_s0"]

        class RepairPolicy:
            def __init__(self, kind):
                self.kind = kind
                self.count = 0

            def choose_action(self, observation, action_mask):
                h = heuristic.choose_action(observation, action_mask)
                p = ppo.choose_action(observation, action_mask)
                trip = 3 <= h <= 5 and p in (0, 1, 2, 6)
                cleave = h == 6 and p in (0, 1, 2)
                if (self.kind in ("trip", "both") and trip) or (self.kind in ("cleave", "both") and cleave):
                    self.count += 1
                    return h
                return p

        interventions = {}
        for kind in ("trip", "cleave", "both"):
            repair = RepairPolicy(kind)
            summary = evaluate(repair, range(3000, 8000), stage=stage)
            wins = {r.seed: int(r.won) for r in summary.results}
            interventions[kind] = {"summary": summary.as_dict(), "overrides": repair.count,
                                   "paired_to_ppo_s0": paired_comparison(wins, results["ppo_s0"]["wins_by_seed"]),
                                   "paired_to_tuned": paired_comparison(wins, results[best]["wins_by_seed"])}
            print(f"repair {kind}: {summary.wins}/5000; overrides={repair.count}", flush=True)
        (output / "interventions.json").write_text(json.dumps(interventions, indent=2) + "\n")
    else:
        fields = {name: i for i, name in enumerate(tactical_observation_fields(stage))}

        class ForceAdaptivePlan:
            """Enforce the initial plan; retain PPO choices for all other actions."""
            def __init__(self):
                self.forced_advances = 0
                self.blocked_advances = 0

            def choose_action(self, observation, action_mask):
                brute = next(s for s in range(3) if observation[fields[f"enemy_{s}_is_brute"]])
                hp = observation[fields[f"enemy_{brute}_max_hp"]]
                ac = observation[fields[f"enemy_{brute}_ac"]]
                tanky = hp / min(.95, max(.05, (26-ac)/20)) >= 42
                if tanky and action_mask[10]:
                    self.forced_advances += 1
                    return 10
                mask = action_mask.copy()
                if not tanky:
                    self.blocked_advances += bool(mask[10])
                    mask[10] = False
                return policies["ppo_s0"].choose_action(observation, mask)

        policy = ForceAdaptivePlan()
        summary = evaluate(policy, range(3000, 8000), stage=stage)
        wins = {r.seed: int(r.won) for r in summary.results}
        result = {"summary": summary.as_dict(), "forced_advances": policy.forced_advances,
                  "decisions_with_advance_blocked": policy.blocked_advances,
                  "paired_to_ppo_s0": paired_comparison(wins, results["ppo_s0"]["wins_by_seed"]),
                  "paired_to_adaptive": paired_comparison(wins, results[best]["wins_by_seed"])}
        (output / "macro_intervention.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"force adaptive plan: {summary.wins}/5000", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
