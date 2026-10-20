# Results cache

Each entry holds the numbers that one current decision rests on, the code area
concerned, and the experiment-history entries that record how they were
measured. Read only the entries for the area you're touching. The full record
of every experiment, with its method, irregularities and open leads, is
`HISTORY.md` on the `history` branch: `git show origin/history:HISTORY.md`.
How entries enter and leave this cache is in [AGENTS.md](../AGENTS.md#evidence).

Seed ranges, including spent and frozen ones, are in `evaluation/seeds.py`.

## Simulator scope

**Area:** `combat/`, `agents/`. **History:** H11, H18, H19.

**Decision:** M0–M6 are frozen as a test bed and a stand-in game; see
[ARCHITECTURE.md](ARCHITECTURE.md#stage-progression).

| Stage | Best hand heuristic | MaskablePPO | Best search or probe |
|---|---:|---:|---|
| M4 | 63.66% (threat-aware) | 63.22% (best checkpoint) | Expectimax 66.64%, +2.98 [+1.34, +4.62] |
| M5 | 78.52% (tuned) | 74.38% (3-seed pool) | Headroom probe +1.70, conservative CI [−1.23, +4.61] |
| M6 | 80.60% (adaptive) | 75.21% (3-seed pool) | Headroom probe +2.00, conservative CI [−1.03, +5.02] |

M4 is on seeds 10000–14999, M5/M6 on 3000–7999, and the probe on
223000–223999. PPO plateaued by about 500k steps.

## Reference heuristics

**Area:** `agents/heuristic_agent.py`, `agents/tactical_heuristics.py`.
**History:** H9, H11, H12.

**Decision:** the hand heuristics are the baselines and the planners'
continuation policies: threat-aware on M4, `tuned` on M5, `adaptive` on M6.

| Stage | Policy | Win rate | Seeds, n |
|---|---|---:|---|
| M4 | stage heuristic / threat-aware | 61.24% / 63.66% | 10000–14999, 5,000 |
| M5 | naive / trip_focus / healer_first / tuned | 61.44 / 76.62 / 73.86 / 78.52% | 3000–7999, 5,000 |
| M6 | naive / hold / dive_all / adaptive | 40.92 / 78.88 / 79.36 / 80.60% | 3000–7999, 5,000 |
| M5 / M6 | tuned / adaptive, fresh seeds | 77.84 / 81.02% | 40000–44999, 5,000 |

- **adaptive − hold:** +1.72 points (SE 0.59), but only +1.14 (SE 0.60) on
  the fresh seeds.
- **Dive rule:** adaptive dives when the front Brute's HP divided by our hit
  chance is at least 42.
- **Seed caveat:** 3000–7999 informed the stage retune, so they aren't an
  untouched test set.

## M5/M6 transition model

**Area:** `combat/table_tactical_transitions.py`, `combat/tactical_transitions.py`.
**History:** H14, H15.

**Decision:** the table model is the registered default, and the replay model is
its test oracle.
- **Agreement:** on seeds 210000–210009 the two give the same successor sets,
  with a largest probability difference of 1.0e-14.
- **Speed:** 16× faster on M5 and 53× on M6.
- **Audit:** no mismatches in 118,089 cases, constructed edge states included
  (seeds 211000–211029).
- **Oracle check:** the replay model matches raw path enumeration through
  `env.step` to about 1e-16 (241 cases).

## Expectimax planner and value leaves

**Area:** `agents/expectimax_agent.py`, `agents/value_functions.py`,
`value_training.py`, `combat/transitions.py`. **History:** H6, H10.

**Decision:** exact chance-node expectimax is the maintained planner. Select
leaves on paired play, not validation loss. On M4 (test 10000–14999, n=5,000,
paired against threat-aware 63.66%):

| Planner | Win rate | Difference [95% CI] | ms per planned decision |
|---|---:|---:|---:|
| Depth 1 + race leaf | 48.08% | −15.58 [−17.32, −13.84] | 66 |
| Depth 1 + V0 | 62.88% | −0.78 [−2.45, +0.89] | 39 |
| Depth 1 + V1 | 65.50% | +1.84 [+0.19, +3.49] | 42 |
| Depth 2 (10k-node budget) + V1 | 66.64% | +2.98 [+1.34, +4.62] | 299 |

- **Validation loss misleads:** V0 had the lower validation loss (0.4135
  against V1's 0.4159) but played 4.65 points worse on selection seeds.
- **Depth 2 helps:** on M2 it cut the learned leaf's regret about 3× (0.0130 →
  0.0040 per episode).
- **Latency:** the worst-case decision is about 4.6 s.

## Rollout planner

**Area:** `agents/rollout_agent.py`. **History:** H8, H13.

**Decision:** the sampling planner is kept as an independent verification
reference for continuation and reconstruction, not as a policy; see
[ARCHITECTURE.md](ARCHITECTURE.md#dependency-direction-and-verification). With
32 rollouts per action it lost to its own continuation:
- M4: 47/100 against 64/100;
- M5: −20.8 points (SE 2.5);
- M6: −16.0 points (SE 3.3).

## Learned-leaf planning on M5/M6

**Area:** future planners in `agents/`. **History:** H14, H16, H18.

**Decision:** shelved. A planner that extends a baseline policy gates its
deviations on statistical evidence, not a fixed margin.

The guided planner lost to its anchor:
- M5: −15 [−25.9, −4.1];
- M6: −19 [−29.7, −8.3].

The earlier M5 pilots lost 27–34 points with unconstrained search. With a fixed
3-point margin they only reverted to the heuristic.

Revisit only with all of these:
- one planner configuration for both collection and evaluation;
- pooled training data;
- significance-gated deviations;
- breadth-first use of the budget before horizons beyond one decision.

For BG3, planning needs a transition model of the real game, which needs logged
fights first.
