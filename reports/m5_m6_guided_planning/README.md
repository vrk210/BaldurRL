# Archived: M5/M6 guided planning and expert iteration

This branch preserves an experiment that was shelved on 2026-10-15. It is not
meant to be merged as is. It sits on top of the commit that made the table
transition model the M5/M6 default, which is the part of the work that was kept.

## What is here

| Path | Contents |
|---|---|
| `agents/guided_planner.py` | Bounded decision search with exact chance outcomes, policy top-k proposals, heuristic anchor and deviation margin |
| `agents/policy_prior.py` | Masked heuristic and learned policy priors |
| `expert_iteration.py` | Collect teacher labels with exploration; fit value and prior networks |
| `value_training.py` | Adds `td:<lambda>` successor-value targets (data must come from `expert_iteration.py`) |
| `evaluation/planning_eval.py` | Adds `--policy guided` and `--policy prior` |
| `tests/test_guided_planning.py` | Tests for the above |
| `review_and_ppo_tuning_handoff.md` | The original handoff and PPO plan, with inline review comments and the full findings list |
| `PROGRESS.md` | The experiment's seed declarations and progress log |
| `deviations.py`, `deviations.json` | Review script and output: where the planner deviates from the anchor |

Results and the review summary are in `docs/RESULTS.md` on the main line (branch `m5-m6-table-model`, PR #11).

## Fix before reusing

From the review (details in `review_and_ppo_tuning_handoff.md`):

1. **Teacher mismatch.** `expert_iteration.py collect` defaults to
   `--root-top-k 0` (score every action), while evaluation used top 3 plus the
   anchor. On M5 the collection setting won 37/100 against 70/100 for the
   evaluated one. Collection and evaluation should share one planner config.
2. **Too little data for the iterated network.** The second-round value network
   was fit from scratch on 60 fights and dropped the 600 bootstrap fights. Pool
   the data or warm-start.
3. **Depth-first budget use.** `agents/guided_planner.py` lines 100–110 score
   later siblings at a shallower depth once the node budget runs out, then take
   a maximum across depths. This doesn't matter at horizon 1. Fix it before
   using longer horizons.
4. **Value exploitation.** A fixed 3-point margin did not stop the planner from
   deviating at about 13% of decisions, which cost 15–19 points. Use a
   significance-based rule or restrict deviations to well-supported actions.

Seeds already used by this experiment are recorded in `evaluation/seeds.py` on the main line.
