import json, sys
from collections import Counter
import numpy as np
from combat.stages import make_env
from evaluation.planning_eval import make_policy
out = {}
for st in ("m5", "m6"):
    pol = make_policy(st, "guided", leaf=f"mlp:runs/planning_followup/{st}/v1/value.npz",
                      prior=f"runs/planning_followup/{st}/v1/prior.npz", root_top_k=3, horizon=1, deviation_margin=.03)
    env = make_env(st); names = env.action_names
    pairs, gaps, decisions, dev_eps = Counter(), [], 0, 0
    for seed in range(122000, 122100):
        obs, _ = env.reset(seed=seed); done = False; dev = False
        while not done:
            a = pol.choose_action(obs, env.action_masks()); d = pol.last_diagnostics; decisions += 1
            if d is not None and a != d.anchor_action:
                q = dict(d.q_values); pairs[(names[d.anchor_action], names[a])] += 1; gaps.append(q[a] - q[d.anchor_action]); dev = True
            obs, r, t, tr, _ = env.step(a); done = t or tr
        dev_eps += dev
    def short(n): return n.split("_ENEMY_")[0]
    coarse = Counter()
    for (x, y), c in pairs.items(): coarse[(short(x), short(y))] += c
    out[st] = dict(decisions=decisions, deviations=sum(pairs.values()), episodes_with_deviation=dev_eps,
                   gap_mean=float(np.mean(gaps)), gap_median=float(np.median(gaps)),
                   top=[(f"{x} -> {y}", c) for (x, y), c in coarse.most_common(8)])
print(json.dumps(out, indent=1))
