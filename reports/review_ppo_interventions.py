"""Whole-fight tests of the report's M5 mistake attribution on saved PPO models.

PYTHONPATH=. /path/to/python reports/review_ppo_interventions.py 1 2
"""
import json
import sys
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from agents.tactical_heuristics import make_tactical_heuristic
from evaluation.evaluate import SB3Policy, evaluate
from evaluation.paired import paired_comparison


def run(model_seed):
    torch.set_num_threads(1)
    heuristic = make_tactical_heuristic("m5", "tuned")
    ppo = SB3Policy(MaskablePPO.load(f"runs/m5_ppo_1m_s{model_seed}/final_model.zip", device="cpu"))
    root = Path("runs/review/m5")
    bases = {}
    for name in (f"ppo_s{model_seed}", "tuned"):
        saved = json.loads((root / f"{name}.json").read_text())
        bases[name] = {int(k): v for k, v in saved["wins_by_seed"].items()}

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

    results = {}
    for kind in ("trip", "cleave", "both"):
        policy = RepairPolicy(kind)
        summary = evaluate(policy, range(3000, 8000), stage="m5")
        wins = {r.seed: int(r.won) for r in summary.results}
        results[kind] = {"summary": summary.as_dict(), "overrides": policy.count,
                         "wins_by_seed": wins,
                         "paired_to_ppo": paired_comparison(wins, bases[f"ppo_s{model_seed}"]),
                         "paired_to_tuned": paired_comparison(wins, bases["tuned"])}
        print(f"s{model_seed} {kind}: {summary.wins}/5000; change="
              f"{100*results[kind]['paired_to_ppo']['difference']:+.2f} pts; overrides={policy.count}", flush=True)
        (root / f"interventions_s{model_seed}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    for model_seed in map(int, sys.argv[1:]):
        run(model_seed)
