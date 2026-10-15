# Simulator phase summary (M0–M6)

As of 2026-10-15 the simulator stages are no longer the main line of work.
They're hard enough to separate good combat policies from weak ones, and more
abstract difficulty wouldn't bring the project closer to playing BG3. The
simulator stays as a test bed and as a stand-in game for the harness. The next
simulator work should be calibrated to logged BG3 fights, not invented.

## Results by stage

Seed ranges and sample sizes differ between rows, so compare within a row only.
Details are in each stage's report.

| Stage | Adds | Best hand heuristic | MaskablePPO | Best search or exact result |
|---|---|---:|---:|---|
| M0 | Fixed 1v1 | 97.5% | 98.0% (200k) | — |
| M1A | Random opponent | 95.0% | 95.7–96.2% (200k) | — |
| M1B | Two enemies, target choice | 67.5% | 66.0–69.3% (200k) | — |
| M2 | One-use Cleave | 66.0% | 69.3% mean (1M) | Exact optimum 71.8% (seeds 9000–13999) |
| M3 | 2v2 | 99.2% | 98.8% (1M) | — |
| M4 | 2v3, scaled enemies | 63.7% (threat-aware) | 62.3–63.2% (1M–5M) | Expectimax depth 2 with learned leaf: 66.6%, +3.0 points [+1.3, +4.6] |
| M5 | Enemy roles, Trip/Prone, Dodge, advantage | 78.5% (tuned) | 74.4% (1M, 3 seeds) | Rollout and guided planners lost to the heuristic |
| M6 | Front/back ranks, Advance, opportunity attacks | 80.6% (adaptive) | 75.2% (1M, 3 seeds) | Same |

M0, M1A and M3 are at the ceiling. On M2, the heuristic and optimum figures are
exact expected win rates on seeds 9000–13999, and the PPO figure is the
three-seed mean of 5,000 played fights on the same seeds.

## What we learned

1. **PPO is a competent learner, not a superior one.** It gets within about 2
   points of the exact optimum on M2 and matches heuristics on M4. On M5/M6 it
   plateaus about 4–5 points below a good hand heuristic by 500k steps. Only
   default hyperparameters were tried.
2. **Search with an exact model is the only method that beat the heuristics**,
   and only on M4. There, exact enemy-phase enumeration added real information
   before each value estimate.
3. **Search exploits value errors.** Hand-built leaves (M4: 48%), noisy rollout
   estimates (M5: −21 points at 32 rollouts per action) and learned value
   networks (M5/M6 guided planner: −15 and −19 points) all lost to the policy
   they were meant to improve. Taking the best of many noisy action values
   selects the errors. Any planner on the real game needs either exact
   transitions plus accurate values, or a conservative rule backed by
   significance rather than a fixed margin.
4. **Good hand heuristics are strong baselines.** Most of the M5/M6 spread is one
   rule ("use Trip"). The remaining gains are second-order: target priority,
   resource timing, and an encounter-dependent plan on M6.
5. **Specific tactics in these rules:**
   - Focus the lowest-HP enemy rather than the Healer first (+2.8 points on M5).
   - Trip the focus target before attacking it.
   - On M6, choose to dive or hold based on how durable the Brute is.
   - Never Disengage.

   These are properties of invented rules; whether any hold in BG3 is untested.
6. **Evaluation discipline matters more than any single result.** Paired seeds,
   5,000-fight evaluations, a frozen test range and independent reviews caught
   several conclusions that a 200-fight check would have gotten wrong. Examples:
   an early Cleave diagnostic, the best-checkpoint bias, and the expert-iteration
   teacher mismatch.

## What carries over to BG3

- **Infrastructure:** exact transition models and their oracle-style
  verification, expectimax, paired evaluation, and the [seed ledger](../docs/SEEDS.md).
- **Method lessons:** points 2, 3 and 6 above.
- **Not carried over:** trained policies. Their observations and action indices
  are fixed simulator vectors that don't exist in BG3.
- **Hypotheses, not facts:** the tactics in point 5.

## Status of open lines

| Line | Status |
|---|---|
| M5/M6 table transition model | Merged as the default ([report](m5_m6_planning.md)) |
| Guided planner and expert iteration | Shelved on `archive/m5-m6-guided-planning`, with the review |
| PPO tuning sweep | Not started. Superseded by the headroom probe below |
| Headroom probe: heuristic improved by many rollouts per action | M5 +1.70 pp [-1.23, +4.61]; M6 +2.00 pp [-1.03, +5.02]; conservative 95% CIs, 1,000 paired fights/stage. [Method and result](../agents/improved_heuristic_agent.py) |
| Mining existing simulator logs into a tactics playbook | Proposed; seeds 230000–239999 reserved |
| Simulator behind the BG3 snapshot and command interface | In progress separately with the harness work |
