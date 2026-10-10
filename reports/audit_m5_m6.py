"""Independent trace/statistics audit and broad reconstruction stress check."""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from agents.random_agent import RandomAgent
from agents.tactical_heuristics import make_tactical_heuristic
from combat.simulation import simulation_from_observation
from combat.stages import make_env


def reconstruction():
    report = {}
    for stage in ("m5", "m6"):
        env = make_env(stage)
        policy = RandomAgent(981)
        count = states = 0
        for seed in range(51000, 51100):
            obs, _ = env.reset(seed=seed)
            done = False
            index = 0
            while not done:
                mask = env.action_masks()
                if index % 4 == 0:
                    states += 1
                    for action in np.flatnonzero(mask):
                        immutable = {id(env.rules): env.rules}
                        immutable.update({id(actor.turn_refresh): actor.turn_refresh
                                          for actor in (*env.allies, *env.enemies)})
                        live = copy.deepcopy(env, immutable)
                        live.np_random = np.random.default_rng(seed + index)
                        sim = simulation_from_observation(stage, obs, seed=seed + index)
                        assert np.array_equal(sim.action_masks(), mask)
                        a, b = live.step(int(action)), sim.step(int(action))
                        assert np.array_equal(a[0], b[0]), (stage, seed, index, action)
                        assert a[1:] == b[1:], (stage, seed, index, action, a[1:], b[1:])
                        assert np.array_equal(live.action_masks(), sim.action_masks())
                        count += 1
                obs, _, t, tr, _ = env.step(policy.choose_action(obs, mask))
                done = t or tr
                index += 1
        report[stage] = {"states": states, "all_legal_candidate_transitions": count, "mismatches": 0}
        print("reconstruction", stage, report[stage], flush=True)
    return report


def audit_traces(stage):
    root = Path("runs/v2") / stage
    env = make_env(stage)
    best = make_tactical_heuristic(stage, "tuned" if stage == "m5" else "adaptive")
    labels = env.action_names
    family = lambda i: labels[i].rsplit("_ENEMY_", 1)[0] if labels[i].startswith(("ATTACK", "TRIP")) else labels[i]
    report = {}
    kinds, samples, seeds_seen = Counter(), defaultdict(list), defaultdict(set)
    decisions = episodes = 0
    for directory in sorted(root.iterdir()):
        if not (directory / "run_card.json").exists() or "best" in directory.name:
            continue
        card = json.loads((directory / "run_card.json").read_text())
        not_won = set(card["loss_seeds"]) | set(card["truncation_seeds"])
        if card["summary"]["episodes"] != 5000:
            continue
        totals = Counter()
        trip_count = Counter()
        mismatch = 0
        selected = directory.name.startswith("ppo_s") and directory.name[-1].isdigit()
        for line in (directory / "episodes.jsonl").open():
            ep = json.loads(line)
            r = ep["result"]
            mismatch += r["won"] != (r["seed"] not in not_won)
            totals["wins"] += r["won"]
            totals["episodes"] += 1
            trip_count[sum(s["semantic_action"] == "TRIP" for s in ep["steps"])] += 1
            for step in ep["steps"]:
                assert step["action_mask"][step["action_index"]]
                totals[step["semantic_action"]] += 1
                if not selected or sum(step["action_mask"]) == 1:
                    continue
                decisions += 1
                obs = np.array(step["observation"], dtype=np.float32)
                mask = np.array(step["action_mask"], dtype=bool)
                h, p = best.choose_action(obs, mask), step["action_index"]
                if h == p:
                    continue
                kind = (family(h), family(p)) if family(h) != family(p) else (labels[h], "other " + family(p))
                kinds[kind] += 1
                seeds_seen[kind].add((directory.name, r["seed"]))
                if len(samples[kind]) < 12 and kinds[kind] % 7 == 1:
                    samples[kind].append({"model": directory.name, "episode_seed": r["seed"]})
            episodes += selected
        report[directory.name] = {"totals": dict(totals), "card_outcome_mismatches": mismatch,
                                  "trip_count_distribution": dict(trip_count)}
        print("trace", stage, directory.name, "mismatches", mismatch, flush=True)
    report["ppo_diagnostic_sampling"] = {
        "episodes": episodes, "nonforced_decisions": decisions, "disagreements": sum(kinds.values()),
        "top_six": [{"kind": list(k), "count": n, "per_fight": n/episodes,
                     "distinct_model_episodes_available": len(seeds_seen[k]),
                     "original_selected_states": samples[k]} for k, n in kinds.most_common(6)]}
    return report


if __name__ == "__main__":
    result = {"reconstruction": reconstruction()}
    for stage in ("m5", "m6"):
        result[stage] = audit_traces(stage)
    Path("runs/review/trace_audit.json").write_text(json.dumps(result, indent=2) + "\n")
