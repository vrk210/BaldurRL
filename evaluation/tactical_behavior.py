"""Role-aware behavior summaries of M5/M6 evaluation traces (diagnostics only).

Reads ``episodes.jsonl`` written by ``evaluation.evaluate`` and decodes roles and
positions from the recorded observations using the stage's field names.
"""

import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from combat.tactical_env import tactical_observation_fields

_ROLES = ("brute", "archer", "healer")


def _roles(fields: dict[str, int], observation: list[float]) -> list[str]:
    return [
        next(role.upper() for role in _ROLES if observation[fields[f"enemy_{slot}_is_{role}"]])
        for slot in range(3)
    ]


def _front_durability(fields: dict[str, int], observation: list[float], roles: list[str]) -> float:
    slot = roles.index("BRUTE")
    ac = observation[fields[f"enemy_{slot}_ac"]]
    return observation[fields[f"enemy_{slot}_hp"]] / min(0.95, max(0.05, (26 - ac) / 20))


def summarize_episode(stage: str, episode: dict[str, Any]) -> dict[str, Any]:
    fields = {name: index for index, name in enumerate(tactical_observation_fields(stage))}
    steps = episode["steps"]
    roles = _roles(fields, steps[0]["observation"])
    counts: Counter[str] = Counter()
    trip_followed = 0
    for index, step in enumerate(steps):
        action, target = step["semantic_action"], step["target_index"]
        counts[action] += 1
        if target is not None:
            counts[f"{action}:{roles[target]}"] += 1
        if action == "TRIP":
            counts["TRIP_SUCCESS"] += bool(step["info"]["trip"]["success"])
            actor = step["observation"][0]
            for later in steps[index + 1:]:
                if later["observation"][0] != actor or later["semantic_action"] == "END_TURN":
                    break
                if later["semantic_action"] in ("ATTACK", "CLEAVE") and (
                    later["semantic_action"] == "CLEAVE" or later["target_index"] == target
                ):
                    trip_followed += 1
                    break
    summary: dict[str, Any] = {
        "won": episode["result"]["won"],
        "roles": roles,
        "counts": dict(counts),
        "trip_then_attack": trip_followed,
        "first_kill_role": (episode["result"].get("kill_order_roles") or [None])[0],
    }
    if stage == "m6":
        advances = [step for step in steps if step["semantic_action"] == "ADVANCE"]
        summary["advances"] = len(advances)
        summary["first_advance_round"] = advances[0]["observation"][-1] if advances else None
        summary["opening_brute_durability"] = _front_durability(fields, steps[0]["observation"], roles)
    return summary


def summarize_run(stage: str, trace_path: Path) -> dict[str, Any]:
    with Path(trace_path).open(encoding="utf-8") as handle:
        episodes = [summarize_episode(stage, json.loads(line)) for line in handle]
    n = len(episodes)
    total: Counter[str] = Counter()
    for episode in episodes:
        total.update(episode["counts"])
    trips = total["TRIP"]
    attacks = sum(total[f"ATTACK:{role}"] for role in ("BRUTE", "ARCHER", "HEALER"))
    report: dict[str, Any] = {
        "episodes": n,
        "win_rate": sum(e["won"] for e in episodes) / n,
        "per_episode": {key: total[key] / n for key in ("TRIP", "ATTACK", "CLEAVE", "DODGE", "SECOND_WIND")},
        "episodes_using": {
            key: sum(e["counts"].get(key, 0) > 0 for e in episodes) / n
            for key in ("TRIP", "CLEAVE", "DODGE", "SECOND_WIND", "ACTION_SURGE")
        },
        "trip_target_share": {role: total[f"TRIP:{role}"] / trips if trips else 0.0
                              for role in ("BRUTE", "ARCHER", "HEALER")},
        "trip_success_rate": total["TRIP_SUCCESS"] / trips if trips else 0.0,
        "trip_then_attack_rate": sum(e["trip_then_attack"] for e in episodes) / trips if trips else 0.0,
        "attack_target_share": {role: total[f"ATTACK:{role}"] / attacks if attacks else 0.0
                                for role in ("BRUTE", "ARCHER", "HEALER")},
        "first_kill_role": {role: sum(e["first_kill_role"] == role for e in episodes) / n
                            for role in ("BRUTE", "ARCHER", "HEALER")},
    }
    if stage == "m6":
        tanky = [e for e in episodes if e["opening_brute_durability"] >= 42]
        soft = [e for e in episodes if e["opening_brute_durability"] < 42]
        dove = [e for e in episodes if e["advances"]]
        report["m6"] = {
            "episodes_with_advance": len(dove) / n,
            "mean_advances": mean(e["advances"] for e in episodes),
            "mean_first_advance_round": mean(e["first_advance_round"] for e in dove) if dove else None,
            "advance_rate_tanky_brute": sum(bool(e["advances"]) for e in tanky) / len(tanky) if tanky else None,
            "advance_rate_soft_brute": sum(bool(e["advances"]) for e in soft) / len(soft) if soft else None,
            "win_rate_tanky_brute": sum(e["won"] for e in tanky) / len(tanky) if tanky else None,
            "win_rate_soft_brute": sum(e["won"] for e in soft) / len(soft) if soft else None,
            "disengage_per_episode": total["DISENGAGE"] / n,
        }
    return report


def main() -> None:
    parser = ArgumentParser(description="Summarize M5/M6 trace behavior by enemy role")
    parser.add_argument("--stage", choices=["m5", "m6"], required=True)
    parser.add_argument("traces", type=Path, nargs="+", help="episodes.jsonl files")
    args = parser.parse_args()
    print(json.dumps({str(path): summarize_run(args.stage, path) for path in args.traces}, indent=2))


if __name__ == "__main__":
    main()
