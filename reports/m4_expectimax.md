# M4 expectimax planning with exact chance nodes and a learned value

Verdict: searching the simulator as an exact model beats every M4 baseline on
the shared test seeds 10000–14999. Depth-1 expectimax with a learned value leaf
wins 65.50% (+1.84 pp over the best baseline, the threat-aware heuristic at
63.66%; 95% CI +0.19 to +3.49), and node-budgeted depth 2 wins 66.64% (+2.98 pp,
CI +1.34 to +4.62) at about 0.3 s per planned decision. The leaf value is the
crux: a hand-built heuristic leaf is exploited by the search (48.08%), and the
first learned value only matches the baselines (62.88%) until one round of
retraining on the planner's own games.

## Method

- **Exact transitions** (`combat/transitions.py`, `combat/distributions.py`).
  For a compact hashable state and a legal decision, the model returns every
  successor with its exact probability, merging identical successors. `END_TURN`
  includes the whole automatic enemy phase (slot-order lowest-living-ally
  targeting, dead-actor skipping, defeat, round-50 truncation, turn-start
  refresh). Dice, hit/miss/natural 1/natural 20, doubled critical dice, Cleave
  halving with minimum 1, and Second Wind 1d10+2 capped at max HP are
  enumerated through the same small rule functions the sampling mechanics call.
  No capping or bucketing.
- **Expectimax** (`agents/expectimax_agent.py`). Max over the active ally's
  decisions, exact expectation over chance outcomes, exact terminal values
  (victory 1; defeat and truncation 0). Depth counts controlled-turn boundaries:
  depth 1 covers the rest of the current ally's turn plus the following enemy
  phase when there is one, then scores the next controlled turn start with the
  leaf value. Leaves are evaluated in one batch per search, and a transposition
  table is shared across the decisions of one turn. With a node budget, a depth-2
  search that would expand more than 10,000 decision nodes is abandoned and
  depth 1 is used instead.
- **Leaf values** (`agents/value_functions.py`): `RaceValue`, a mean-field damage
  race with a fitted two-parameter logistic, and `MLPValue`, a 32-128-128-1 ReLU
  MLP on observations divided by the observation-space highs, with inactive
  allies' Action and Bonus Action counts zeroed (they are reset at that ally's
  next turn start, so they cannot matter).
- **Value learning** (`value_training.py`). Binary cross-entropy, AdamW
  (lr 2e-3, weight decay 1e-4, cosine schedule), batch 2048, 25 epochs, keeping
  the epoch with the best validation loss.

Seed ranges: value data 20000–39999 (round 0) and 60000–67999 (self-play
rounds); value validation loss 40000–41999; planner selection 42000–43999 (2,000
episodes); final test 10000–14999, used only for the final runs below.

## Validating the exact model

- Exhaustive: for 14 states across M0–M4 and every legal action, a scripted RNG
  enumerated every RNG path through the real `env.step` (an odometer over draws).
  The enumerated distributions matched to 1e-12, including round-50 truncation,
  ally death mid-phase, and skipping a dead first ally (`tests/test_transitions.py`).
- Statistical: 65 (state, action) cases across M0–M4, with 200,000 sampled
  transitions each (`runs/transition_validation/sampling_200k.json`). No sample
  fell outside the enumerated support, and probabilities summed to 1 within
  4.4e-16. Chi-square z-scores had mean 0.19 and maximum 3.26. The one high
  case (an M2 Cleave) was then shown to be exact by full path enumeration.
- Against the M2 exact solver: depth-unbounded search reproduced the
  supervisor's `m2_dp.py` values on 12 random small M2 states to 6e-16, and an
  independent condensed oracle in `tests/m2_oracle.py` to 1e-12. Depth-1 search
  with the exact V* as its leaf reproduced Q* at every decision.
- Seeded behavior is unchanged: on the test seeds the heuristic, threat-aware,
  and both PPO baselines reproduce `runs/m4_longer_training/comparison.json`
  exactly (3062, 3183, 3113, 3161 wins).

## M2: regret against the exact optimum

Per-decision regret is `max Q*(legal) - Q*(chosen)` from `m2_dp.py` Q tables,
summed per episode (`runs/m2_validation/`). M2 value nets were trained the same
way as on M4, on M2 heuristic and PPO games from seeds 20000–41999.

| Policy (seeds 9000–9999, n=1000) | Win rate (SE) | Regret / episode (SE) | ms / planned decision |
| --- | ---: | ---: | ---: |
| Exact optimum (mean V*) | 71.16% | 0 | — |
| Depth 1 + exact V* leaf (sanity check) | 71.13% (1.43) | 0.0000 | 8.7 |
| Depth 1 + learned M2 V0 | 70.13% (1.45) | 0.0193 (0.0012) | 10.4 |
| Depth 1 + race leaf | 60.14% (1.55) | 0.0948 (0.0028) | 12.7 |
| M2 heuristic | 63.94% (1.52) | 0.0594 (0.0020) | — |

On seeds 9000–9399, the heuristic's regret (0.0561) matches the supervisor's
`regret.json`, where PPO 1M s2 had 0.0162. On the same 400 seeds:

| Leaf (seeds 9000–9399) | Regret / episode |
| --- | ---: |
| V0, depth 1 | 0.0177 |
| V1 trained on outcome / search-root / lambda 0.5 / lambda 0.8 targets, depth 1 | 0.0130 / 0.0149 / 0.0157 / 0.0143 |
| V1 outcome, depth 2 | 0.0040 (paired change vs depth 1: -0.0090, SE 0.0011) |

Depth 2 cuts exact regret by about 3x with the same leaf. Its M2 cost is
321 ms mean per planned decision, with a maximum of 6.8 s.

## Learning the M4 value

| Round | Data (states) | Targets | Planner win rate, seeds 42000–43999 (paired vs threat-aware) |
| --- | --- | --- | --- |
| 0 | 22,000 episodes from the heuristic, threat-aware heuristic, both with epsilon 0.15, and four PPO models (498k) | outcome | V0: 60.70% (-1.75 pp, SE 1.34) |
| 1 | + 4,000 depth-1 V0 self-play episodes (602k total) | outcome | **V1: 65.35% (+2.90 pp, SE 1.32)** |
| 1 | same | search root value | 64.05% (+1.60) |
| 1 | same | lambda-return 0.5 over root values | 63.55% (+1.10) |
| 2 | + 4,001 depth-1 V1 self-play episodes (702k total) | outcome | 64.15% (-1.20 vs V1, SE 1.10) |
| 2 | same | search root value | 62.20% (-3.15 vs V1, SE 1.10) |
| — | ensemble of V1 outcome, V2 outcome, V1 search | — | 65.00% (-0.35 vs V1) |

On the same validation seeds, the heuristic won 58.85%, the threat-aware
heuristic 62.45%, and depth 1 with the race leaf 47.65%. Gains flattened after
round 1, so V1 (outcome targets, `runs/value/v1_outcome.npz`) and node-budgeted
depth 2 (validation 65.65%, +0.30 pp over depth 1) were frozen before the test
runs.

Two variants were tried and dropped. A wider 256x256x256 net overfit
(validation loss 0.416). Appending engineered features (exact expected damage,
threat, attacks-to-kill, race margin) lowered validation loss but won only
63.05% (-2.30 pp vs V1), so that code was removed (archived at
`runs/value/value_functions_with_derived_features.py`).

## M4 test results (seeds 10000–14999, 5,000 episodes each)

Paired differences are against the best baseline (threat-aware heuristic), with
encounter seeds as the pairing unit. No episode was truncated.

| Policy | Wins | Win rate (SE) | Paired diff, pp [95% CI] | ms / planned decision | Max s / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage heuristic | 3062 | 61.24% (0.69) | -2.42 [-3.40, -1.44] | <0.01 | <0.01 |
| Threat-aware heuristic | 3183 | 63.66% (0.68) | reference | <0.01 | <0.01 |
| PPO 1M seed 2 | 3113 | 62.26% (0.69) | -1.40 [-2.88, +0.08] | 0.1 | 0.01 |
| PPO 5M run, best validation (seed 0) | 3161 | 63.22% (0.68) | -0.44 [-2.11, +1.23] | 0.1 | 0.01 |
| Expectimax d1 + race leaf | 2404 | 48.08% (0.71) | -15.58 [-17.32, -13.84] | 66 | 4.4 |
| Expectimax d1 + V0 | 3144 | 62.88% (0.68) | -0.78 [-2.45, +0.89] | 39 | 3.7 |
| **Expectimax d1 + V1** | 3275 | 65.50% (0.67) | **+1.84 [+0.19, +3.49]** | 42 | 4.6 |
| **Expectimax d2 (10k-node budget) + V1** | 3332 | 66.64% (0.67) | **+2.98 [+1.34, +4.62]** | 299 | 4.1 |

Other paired differences: depth 2 vs depth 1 (both with V1) is +1.14 pp
[+0.31, +1.97]; V1 vs V0 at depth 1 is +2.62 pp [+1.20, +4.04]. Against PPO 5M
best, depth 1 is +2.28 pp [+0.70, +3.86] and depth 2 is +3.42 pp [+1.83, +5.01].
The intervals condition on these fitted value nets and PPO models.

## Cost

Times are wall-clock per decision inside the evaluation loop. They were
measured while two to five processes shared an 11-core machine with another
agent, so they are indicative only.

- Depth 1 + V1: 42 ms per planned decision (28 ms over all decisions, including
  forced ones), 0.71 s per episode, maximum 4.6 s. The slowest states are the
  second ally's first turn with all resources and three living enemies: Cleave
  has 729 successors, each followed by a full enemy phase, giving about
  136k leaves.
- Depth 2 with a 10,000-node budget: 299 ms per planned decision; 53% of
  planned decisions (44,989 of 84,425) were searched at depth 2, the rest fell
  back to depth 1. Without the budget, early-fight depth-2 searches took up to
  77 s and 3.3M leaves, so full-width depth 2 was not evaluated.
- Data and training: round-0 collection took seconds per 1,000 episodes;
  self-play collection about 0.8 s per episode; fitting about 10 s per net
  (CPU). The four planner test runs took about 10.5 CPU-hours, about 7 of
  them for depth 2.

## Behavior and failure modes

- **Leaf exploitation.** The max operator finds states where the leaf is
  optimistic. The race heuristic is reasonably calibrated (validation BCE 0.441)
  yet loses 15.6 pp. V0, trained on outcomes of fixed and epsilon-noisy
  policies, only matches the baselines. Retraining on the planner's own
  trajectories (V1) fixed most of this.
- **Validation loss is not play strength.** V0 had a lower validation loss
  than V1 on the same set (0.4135 vs 0.4159) but plays 4.65 pp worse on
  validation and 2.62 pp worse on test. The derived-feature net also had a
  lower loss and played worse. Selection must use play.
- **Bootstrapping drift.** Search-root (TD-style) targets were never better than
  outcomes, and got worse in round 2 (-3.15 pp). On M2 all target types were
  within noise.
- **Winner's curse.** V1 was the best of six validation variants. Its edge over
  the threat-aware heuristic shrank from +2.90 pp on validation to +1.84 pp on
  test.
- **Behavior shift** (validation seeds 42000–42499, `runs/m4_eval/behavior_42000_42499.json`).
  Compared with the threat-aware heuristic, the depth-1 V1 planner Cleaves less
  (1.22 vs 2.00 Cleaves per episode, 2.56 vs 2.96 living enemies when it does)
  and uses Second Wind in more episodes (96% vs 85%).
- **Worst-case latency.** The mean meets the target easily, but individual
  depth-1 decisions reach about 4.6 s. A node budget at depth 1 with a cheaper
  fallback would bound it if needed.
- **Model dependence.** All values assume the simulator is the true game. Any
  mismatch with real BG3 rules is not modeled.

## How a new stage plugs in

Planners use only the `TransitionModel` protocol: `state_from_observation`,
`same_encounter`, `status`, `legal_actions`, `outcomes`, `ends_turn`, and
`observations`. A stage with roles, conditions, or ranks registers its own
enumeration over its own state encoding:

```python
from combat.transitions import register_transition_model
register_transition_model("m5", lambda stage, observation: M5TransitionModel.from_observation(observation))
```

It should build its outcome lists from shared distribution helpers next to its
mechanics, as `combat/distributions.py` does for M0–M4, and copy the
exhaustive RNG-path test in `tests/transition_harness.py` to prove agreement
with its environment. `ExpectimaxAgent`, `MLPValue` (fields are looked up by
name) and `value_training.py` then work unchanged. The value net must be
retrained per stage.

## Reproduce

```bash
# Phase 1 statistical check (about 6.5 min)
PYTHONPATH=.:tests .venv/bin/python runs/transition_validation/validate_sampling.py
# Value data, fit, and self-play round (scripts in runs/value/)
bash runs/value/collect_round0_a.sh; bash runs/value/collect_round0_b.sh
.venv/bin/python value_training.py fit --train runs/value/data/*.npz --val runs/value/val/*.npz --target outcome --out runs/value/v0.npz
.venv/bin/python value_training.py collect --policy expectimax --leaf mlp:runs/value/v0.npz --depth 1 --seed-start 60000 --episodes 2000 --out runs/value/data_r1/x_v0_60000.npz   # and 62000
.venv/bin/python value_training.py fit --train runs/value/data/*.npz runs/value/data_r1/*.npz --val runs/value/val/*.npz --target outcome --out runs/value/v1_outcome.npz
# Final test runs
.venv/bin/python -m evaluation.planning_eval --name x_v1_outcome_d2b10k --policy expectimax --leaf mlp:runs/value/v1_outcome.npz \
    --depth 2 --node-budget 10000 --seed-start 10000 --episodes 5000 --workers 5 --out runs/m4_eval/test/x_v1_outcome_d2b10k.json
PYTHONPATH=. .venv/bin/python runs/m4_eval/summarize.py runs/m4_eval/test threat runs/m4_eval/test_summary.json heuristic threat ...
```

Artifacts (git-ignored `runs/`): `runs/m4_eval/{val,test}/*.json` (per-seed
outcomes and timing), `runs/m4_eval/test_summary.json`, `runs/value/*.npz` (value
nets with `.history.json`), `runs/m2_validation/`, `runs/transition_validation/`.
