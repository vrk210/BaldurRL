# Results cache

Each entry holds the numbers that one current decision rests on, the code area
concerned, and the experiment-history entries that record how they were
measured. Read only the entries for the area you're touching. The full record
of every experiment, with its method, irregularities and open leads, is
`HISTORY.md` on the `history` branch:
`git fetch origin history && git show origin/history:HISTORY.md`. How entries
enter and leave this cache is in [AGENTS.md](../AGENTS.md#evidence).

Seed ranges, including spent and frozen ones, are in `evaluation/seeds.py`.

## Simulator scope

**Area:** `combat/`, `agents/`. **History:** H9, H10, H11, H18, H19.

**Decision:** M0–M6 are frozen as a test bed and a stand-in game; see
[ARCHITECTURE.md](ARCHITECTURE.md#stage-progression). Only M4 expectimax beat
the reference heuristics; both are in the entries below.
- **PPO:** on M4 the best checkpoint won 63.22%, the highest of three on the
  test seeds; the 5M-step mean was 61.79%. On M5/M6 pooled PPO trails the best
  heuristic by 4.14 (SE 0.60) and 5.39 (SE 0.61) points, flat after about 500k
  steps.
- **Headroom probe**, a one-step improvement of the heuristic that deviates
  only on strong evidence (seeds 223000–223999): 80.3% against the heuristic's
  78.6% on M5 (+1.70, conservative 95% CI [−1.23, +4.61]) and 81.6% against
  79.6% on M6 (+2.00, [−1.03, +5.02]).

## Reference heuristics

**Area:** `agents/heuristic_agent.py`, `agents/tactical_heuristics.py`.
**History:** H9, H11, H12.

**Decision:** new policies are compared against the hand heuristics:
threat-aware on M4 (`threat` in `evaluation/planning_eval.py`), `tuned` on M5
and `adaptive` on M6 (`--heuristic-variant`). The named variants are defined in
`VARIANTS` in `agents/tactical_heuristics.py`.

| Stage | Policy | Win rate | Seeds, n |
|---|---|---:|---|
| M4 | stage heuristic / threat-aware | 61.24% / 63.66% | 10000–14999, 5,000 |
| M5 | naive / trip_focus / healer_first / tuned | 61.44 / 76.62 / 73.86 / 78.52% | 3000–7999, 5,000 |
| M6 | naive / hold / dive_all / adaptive | 40.92 / 78.88 / 79.36 / 80.60% | 3000–7999, 5,000 |
| M5 / M6 | tuned / adaptive, fresh seeds | 77.84 / 81.02% | 40000–44999, 5,000 |

On M6, adaptive − hold is +1.72 points (SE 0.59), but only +1.14 (SE 0.60) on
the fresh seeds.

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

**Decision:** exact chance-node expectimax over `RosterTransitionModel` is the
maintained M0–M4 planner. Leaves are selected on paired play, not validation
loss.

**The exact model:**
- It matches exhaustive RNG-path enumeration through `env.step` to 1e-12 on 14
  states.
- In 65 cases × 200k samples, no sample fell outside its support.
- Unbounded search matches the M2 dynamic-programming solver to 6e-16 and
  `tests/m2_oracle.py` to 1e-12.

**On M4** (test 10000–14999, n=5,000, paired against threat-aware 63.66%):
- the race leaf is a hand-built damage-race estimate;
- V0 is a value network trained on heuristic and PPO games;
- V1 is V0's data plus 4,000 of the planner's own games.

| Planner | Win rate | Points vs threat-aware [95% CI] | ms per planned decision |
|---|---:|---:|---:|
| Depth 1 + race leaf | 48.08% | −15.58 [−17.32, −13.84] | 66 |
| Depth 1 + V0 | 62.88% | −0.78 [−2.45, +0.89] | 39 |
| Depth 1 + V1 | 65.50% | +1.84 [+0.19, +3.49] | 42 |
| Depth 2 (10k-node budget) + V1 | 66.64% | +2.98 [+1.34, +4.62] | 299 |

- **Validation loss misleads:** V0 had the lower validation loss (0.4135
  against V1's 0.4159) but played 4.65 points worse on selection seeds.
- **Depth 2 helps:** on M2 it cut the learned leaf's regret about 3× (0.0130 →
  0.0040 per episode).
- **Latency:** the worst-case decision is about 4.6 s, at depth 1.

## Rollout planner

**Area:** `agents/rollout_agent.py`. **History:** H13.

**Decision:** the sampling planner is kept as an independent verification
reference for continuation and reconstruction, not as a policy; see
[ARCHITECTURE.md](ARCHITECTURE.md#dependency-direction-and-verification). With
32 rollouts per action it lost to its own continuation: −20.8 points (SE 2.5)
on M5 and −16.0 (SE 3.3) on M6.
