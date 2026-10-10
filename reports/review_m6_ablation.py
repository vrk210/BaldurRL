"""Separate Trip use from the naive M6 baseline's target-count mismatch."""
import json
from pathlib import Path

from agents.tactical_heuristics import TacticalHeuristicAgent, TacticalHeuristicConfig
from evaluation.evaluate import evaluate
from evaluation.paired import paired_comparison


def main():
    root = Path("runs/review/m6")
    saved = json.loads((root / "naive.json").read_text())
    naive = {int(k): v for k, v in saved["wins_by_seed"].items()}
    configs = {
        "reachable_count_only": TacticalHeuristicConfig(count_reachable_for_cleave=True),
        "trip_only": TacticalHeuristicConfig(trip=True),
        "trip_and_reachable_count": TacticalHeuristicConfig(trip=True, count_reachable_for_cleave=True),
    }
    report = {}
    for name, config in configs.items():
        summary = evaluate(TacticalHeuristicAgent("m6", config), range(3000, 8000), stage="m6")
        wins = {r.seed: int(r.won) for r in summary.results}
        report[name] = {"summary": summary.as_dict(), "wins_by_seed": wins,
                        "paired_to_naive": paired_comparison(wins, naive)}
        (root / "baseline_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, summary.wins, summary.win_rate, flush=True)


if __name__ == "__main__":
    main()
