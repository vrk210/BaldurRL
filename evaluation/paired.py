"""Paired win-rate comparison of two run cards evaluated on the same episode seeds."""

import json
from argparse import ArgumentParser
from math import sqrt
from pathlib import Path
from typing import Any


def wins_from_card(card: dict[str, Any]) -> dict[int, int]:
    """Per-seed win indicator: a seed wins unless listed as a loss or truncation."""
    not_won = set(card["loss_seeds"]) | set(card["truncation_seeds"])
    return {int(seed): int(seed not in not_won) for seed in card["episode_seeds"]}


def paired_comparison(first: dict[int, int], second: dict[int, int]) -> dict[str, float | int]:
    """Mean per-seed win difference (first - second) with its paired standard error.

    Seeds fix each encounter's sampled enemies; later dice can still differ once
    the policies diverge, so pairing removes encounter variance, not all noise.
    """
    if set(first) != set(second) or not first:
        raise ValueError("Paired comparison needs the same non-empty seed set")
    diffs = [first[seed] - second[seed] for seed in sorted(first)]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = sqrt(variance / n)
    return {
        "episodes": n,
        "first_win_rate": sum(first.values()) / n,
        "second_win_rate": sum(second.values()) / n,
        "difference": mean,
        "paired_se": se,
        "z": mean / se if se > 0 else 0.0,
        "first_only_wins": sum(d == 1 for d in diffs),
        "second_only_wins": sum(d == -1 for d in diffs),
    }


def compare_cards(first_path: Path, second_path: Path) -> dict[str, float | int]:
    first = json.loads(Path(first_path).read_text(encoding="utf-8"))
    second = json.loads(Path(second_path).read_text(encoding="utf-8"))
    if first.get("stage") != second.get("stage"):
        raise ValueError("Run cards come from different stages")
    return paired_comparison(wins_from_card(first), wins_from_card(second))


def main() -> None:
    parser = ArgumentParser(description="Paired win-rate difference between two run cards")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_cards(args.first, args.second), indent=2))


if __name__ == "__main__":
    main()
