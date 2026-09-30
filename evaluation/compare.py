"""Compare saved evaluation traces without rerunning combat or policies."""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any


def _read_run(card_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    trace_path = card_path.parent / card["trace_file"]
    episodes: dict[int, dict[str, Any]] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            episodes[episode["result"]["seed"]] = episode
    return card, episodes


def _first_divergence(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any] | None:
    first_steps, second_steps = first["steps"], second["steps"]
    for index, (a, b) in enumerate(zip(first_steps, second_steps)):
        if a["observation"] != b["observation"] or a["action_mask"] != b["action_mask"]:
            return {"step": index, "kind": "state", "first": a["observation"], "second": b["observation"]}
        if a["action_index"] != b["action_index"]:
            return {
                "step": index,
                "kind": "action",
                "observation": a["observation"],
                "action_mask": a["action_mask"],
                "first": a["action"],
                "second": b["action"],
            }
    if len(first_steps) != len(second_steps):
        return {"step": min(len(first_steps), len(second_steps)), "kind": "episode_length"}
    return None


def compare_run_cards(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Summarize outcomes and first divergence for seeds with a loss or truncation."""
    first_card, first_episodes = _read_run(first_path)
    second_card, second_episodes = _read_run(second_path)
    seeds = first_card["episode_seeds"]
    if (
        len(seeds) != len(set(seeds))
        or seeds != second_card["episode_seeds"]
        or set(seeds) != set(first_episodes)
        or set(seeds) != set(second_episodes)
    ):
        raise ValueError("Run cards must contain the same episode seeds and complete traces")

    outcome_counts = {"both_won": 0, "first_only_won": 0, "second_only_won": 0, "neither_won": 0}
    loss_cases = []
    for seed in seeds:
        first = first_episodes[seed]
        second = second_episodes[seed]
        first_won = first["result"]["won"]
        second_won = second["result"]["won"]
        key = "both_won" if first_won and second_won else (
            "first_only_won" if first_won else "second_only_won" if second_won else "neither_won"
        )
        outcome_counts[key] += 1
        if not (first_won and second_won):
            loss_cases.append({
                "seed": seed,
                "first_result": first["result"],
                "second_result": second["result"],
                "first_divergence": _first_divergence(first, second),
            })
    return {
        "first": {"policy": first_card["policy"], "model_path": first_card["model_path"], "run_card": str(first_path)},
        "second": {"policy": second_card["policy"], "model_path": second_card["model_path"], "run_card": str(second_path)},
        "episodes": len(seeds),
        "outcomes": outcome_counts,
        "loss_cases": loss_cases,
    }


def main() -> None:
    parser = ArgumentParser(description="Compare two traced evaluation runs on the same seeds")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_run_cards(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": report["episodes"], "outcomes": report["outcomes"], "loss_cases": len(report["loss_cases"])}, indent=2))


if __name__ == "__main__":
    main()
