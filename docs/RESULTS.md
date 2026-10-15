# Simulator results, M0–M6

**Status, 2026-10-15:** the simulator stages are no longer the main line of work.
They are hard enough to tell good combat policies from weak ones, and more
abstract difficulty wouldn't bring the project closer to playing BG3. The
simulator remains a test bed and a stand-in game behind the BG3 harness
protocols. The next simulator work should be calibrated to logged BG3 fights.

This file replaces the per-experiment reports that were in `reports/` until
2026-10-15. Those reports have the full tables, reproduction commands and
artifact paths; read them with `git show b082262:reports/<name>`, using the names
listed at the end. Raw artifacts live in git-ignored `runs/` directories. Seed
ranges are recorded in `evaluation/seeds.py`.

## Overview

Seed ranges and sample sizes differ between rows, so compare within a row only.

| Stage | Adds | Best hand heuristic | MaskablePPO | Best search or exact result |
|---|---|---:|---:|---|
| M0 | Fixed 1v1 | 97.5% | 98.0% (200k) | — |
| M1A | Random opponent | 95.0% | 95.7–96.2% (200k) | — |
| M1B | Two enemies, target choice | 67.5% | 66.0–69.3% (200k) | — |
| M2 | One-use Cleave | 66.0% | 69.3% mean (1M) | Exact optimum 71.8% |
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
3. **Search exploits value errors.** Each of these lost to the policy it was
   meant to improve: hand-built leaves (M4: 48%), noisy rollout estimates (M5:
   −21 points at 32 rollouts per action) and learned value networks (M5/M6
   guided planner: −15 and −19 points). Taking the best of many noisy action
   values selects errors. Any planner on the real game needs exact transitions
   with accurate values, or a conservative rule backed by significance rather
   than a fixed margin.
4. **Good hand heuristics are strong baselines.** Most of the M5/M6 spread is one
   or two basic rules. The remaining gains are second-order: target priority,
   resource timing, and an encounter-dependent plan on M6.
5. **Validation loss is not play strength.** On M4, lower-loss value networks
   played worse. Select on play, using paired seeds.
6. **Evaluation discipline caught several wrong conclusions.** Paired seeds,
   5,000-fight evaluations, a frozen test range and independent reviews
   corrected: an early Cleave diagnosis, the best-checkpoint bias, the M6 headline
   spread, and the expert-iteration teacher mismatch.

What carries over to BG3: the transition-model and oracle-verification approach,
expectimax, paired evaluation, the seed registry, and points 2, 3, 5 and 6.
Trained policies do not, because their observations and action indices are
fixed simulator vectors. The tactics below are hypotheses for BG3, not facts.

## M0–M4: PPO against hand heuristics

MaskablePPO with terminal ±1 reward, 4 environments, `train.py` defaults, and
deterministic masked prediction. Results are on held-out seeds 3000–3999
(n=1,000) unless noted.

| Stage | Random | Heuristic | PPO seeds 0 / 1 / 2 | Fresh 5,000-seed check |
|---|---:|---:|---|---|
| M0 | 71.7% | 97.5% | 98.0% (200k); 97.1% at 50k | — |
| M1A | 73.5% | 95.0% | 95.7 / 96.2 / 95.9% (200k) | 4000–8999: heuristic 96.00%, PPO 95.80 / 96.02 / 95.84% |
| M1B | 23.7% | 67.5% | 66.0 / 67.7 / 69.3% (200k) | 4000–8999: heuristic 67.86%, PPO 67.90 / 68.42 / 69.92% |
| M2 | 28.4% | 65.0% | 66.6 / 64.8 / 66.3% (200k) | 4000–8999: heuristic 66.32%, PPO 67.76 / 67.18 / 68.28% |
| M3 | 86.3% | 99.2% | 99.0 / 98.7 / 98.8% (1M) | — |
| M4 | 15.6% | 61.4% | 58.2 / 61.1 / 62.2% (1M) | — |

- **M0–M1A:** saturated. PPO's differences from the heuristic, such as delaying
  Action Surge, change behavior but not outcomes.
- **M1B:** the target decision exposes variation between training seeds. One
  seed's 2-point lead didn't reproduce in the other two.
- **M3:** easy. PPO matches the heuristic while barely using Cleave (6–33% of
  fights).
- **M4:** a fair fight. Cleave and Second Wind matter: survivors keep about
  14/40 HP.

### M2: training budget, reward shaping and the exact optimum

- **Training budget.** 1M-step PPO beat the M2 heuristic on fresh seeds
  9000–13999: 68.68 / 69.48 / 69.82% against 65.12%. The 200k models had left
  win rate on the table. Checkpoint selection was not the cause of the earlier
  gap.
- **Cleave.** PPO's gain comes from avoiding the heuristic's automatic opening
  Cleave, not from using Cleave well. The 1M models often Cleave against a
  single enemy, which is dominated. A matched-start diagnostic (seeds 2000–2199)
  shows opening with an attack beats opening with Cleave by +0.59 to +3.01
  points. The same heuristic without Cleave wins 67.98%.
- **Reward shaping.** A bounded damage-dealt term didn't improve sample
  efficiency at 25k–200k steps or final win rate: terminal 65.90% vs shaped
  66.03% mean on seeds 3000–3999.
- **Exact optimum.** A dynamic-programming solver gives an expected optimal win
  rate of 71.79% on seeds 9000–13999 (heuristic: 66.03%). On seeds 9000–9999:

  | Policy | Win rate | Regret per episode |
  |---|---:|---:|
  | Exact optimum (mean V*) | 71.16% | 0 |
  | Depth-1 search + learned V0 | 70.13% | 0.0193 |
  | Depth-1 search + race heuristic leaf | 60.14% | 0.0948 |
  | M2 heuristic | 63.94% | 0.0594 |

  PPO 1M seed 2 had a regret of 0.0162 on seeds 9000–9399. Depth 2 cut the
  learned leaf's regret about 3× (0.0130 → 0.0040).

### M4: expectimax with exact chance nodes

Method details are in [EXPECTIMAX_PLANNING.md](EXPECTIMAX_PLANNING.md). Test
seeds 10000–14999 were used only for these final runs (n=5,000 each). Paired
differences are against the threat-aware heuristic.

| Policy | Win rate | Paired difference [95% CI] | ms per planned decision |
|---|---:|---:|---:|
| Stage heuristic | 61.24% | −2.42 [−3.40, −1.44] | <0.01 |
| Threat-aware heuristic | 63.66% | reference | <0.01 |
| PPO 1M seed 2 | 62.26% | −1.40 [−2.88, +0.08] | 0.1 |
| PPO 5M, best checkpoint | 63.22% | −0.44 [−2.11, +1.23] | 0.1 |
| Expectimax depth 1 + race leaf | 48.08% | −15.58 [−17.32, −13.84] | 66 |
| Expectimax depth 1 + V0 | 62.88% | −0.78 [−2.45, +0.89] | 39 |
| **Expectimax depth 1 + V1** | 65.50% | **+1.84 [+0.19, +3.49]** | 42 |
| **Expectimax depth 2 (10k-node budget) + V1** | 66.64% | **+2.98 [+1.34, +4.62]** | 299 |

- **Value learning.** V0 was trained on 22,000 episodes of heuristics and PPO
  and only matched the baselines. V1 added 4,000 of the planner's own games and
  produced the gain. Search-value and λ-return targets were never better than
  outcomes. A third round and an ensemble didn't help.
- **Failure modes:**
  - Leaf exploitation.
  - Lower validation loss with worse play.
  - Winner's curse: V1's edge shrank from +2.90 points on selection seeds to
    +1.84 on test.
  - Worst-case latency around 4.6 s at depth 1.

## M5/M6: decision-rich stages

The rules are specified in [STAGES_SPEC.md](STAGES_SPEC.md). They are simulator
choices, not BG3 rules.

**Tuning.** The first design (v1) made M5 too easy: naive 75.9%, and Trip every
turn was nearly always right. Version 2 changed four things, chosen on
development seeds 20000–21999:

1. Trip limited to 2 charges per ally.
2. Healer heal of 2d6+2.
3. +1 attack and +1 damage for the Brute and Archer.
4. On M6, Brute HP 20–26 and 2 Healer charges.

Held-out results, seeds 3000–7999, n=5,000 per policy:

| M5 policy | Win rate | M6 policy | Win rate |
|---|---:|---|---:|
| random | 6.70% | random | 6.20% |
| naive (M4 rules) | 61.44% | naive (M4 rules) | 40.92% |
| trip_focus | 76.62% | hold | 78.88% |
| healer_first | 73.86% | dive_all | 79.36% |
| **tuned** | **78.52%** | **adaptive** | **80.60%** |
| PPO 0 / 1 / 2 | 74.90 / 74.68 / 73.56% | PPO 0 / 1 / 2 | 75.74 / 75.36 / 74.52% |

Paired differences (points, with SE):

- **M5:**
  - tuned − naive: +17.08 (0.86)
  - Trip alone: +15.18 (0.88)
  - lowest-HP focus − Healer-first: +2.76 (0.69)
  - tuned − trip_focus: +1.90 (0.43)
  - PPO pooled − tuned: −4.14 (0.60)
- **M6:**
  - adaptive − naive: +39.68 (0.87)
  - dive_all − hold: +0.48 (0.81)
  - adaptive − hold: +1.72 (0.59)
  - PPO pooled − adaptive: −5.39 (0.61)

**Review qualifications (2026-10-10):**

- **Fresh-seed check.** On seeds 40000–44999, tuned scored 77.84% and adaptive
  81.02%. The ranking held, but adaptive − hold was only +1.14 points (SE 0.60).
- **The M6 headline is mostly two baseline fixes.** Counting only reachable
  Cleave targets and using Trip take naive from 40.9% to 77.7%. That is 36.8 of
  the 39.7 points.
- **Seeds 3000–7999 aren't an untouched test set.** They also informed the
  v1 → v2 retune.

**Tactics that won in these rules** (hypotheses for BG3):

- **Kill priority.** Focus the lowest-HP target. Healer-first loses 2.8 points
  because the Healer's self-heal absorbs focus fire.
- **Trip and attack in the same turn.** The competent policies follow about 90%
  of their Trips with an attack on the Prone target the same turn. PPO does so
  only 60–76% of the time.
- **Dive or hold depending on the Brute (M6).** Diving wins against durable
  Brutes, meaning Brute HP divided by our hit chance is at least 42. Holding wins
  against soft ones. Sending a single diver was weak.
- **Never Disengage.** Diving with Disengage won 51.4% on development seeds,
  against 70.4% without it. No final heuristic Dodges.

**PPO diagnostics.**

- Training plateaued by about 500k steps, and best-checkpoint selection didn't
  close the gap.
- A stratified counterfactual check found two supported mistakes:
  - Trip timing on M5: Trip → Attack +1.53 [+0.82, +2.23]; Trip → Cleave +2.70
    [+1.82, +3.58].
  - Trip → Attack on M6: +0.94 [+0.37, +1.51].
- Whole-fight repair of the M5 Trip and Cleave choices recovers 1.05 of the
  4.14-point gap.
- Forcing adaptive's opening plan on M6 recovers +0.44 (SE 0.68).
- Most of PPO's gap remains unexplained.

**Rollout planner pilot** (32 rollouts per action, heuristic continuation):

- M5: 58.0% against the heuristic's 78.8% (−20.8, SE 2.5).
- M6: 65.3% against 81.3% (−16.0, SE 3.3).
- Rescoring the deviations with 1,024 rollouts mostly favored the heuristic, so
  this was finite-sample selection error, not evidence of a ceiling.

## M5/M6: exact transitions and planners

**Transition models.**

- **Replay oracle** (`combat/tactical_transitions.py`):
  - matched raw path enumeration through the unmodified `env.step` to about
    1e-16 (241 cases);
  - had no unseen successor in 20,000 real samples per case (288 cases).
- **Table model** (`combat/table_tactical_transitions.py`, its docstring has the
  details): the default since 2026-10-15. It is 16× faster on M5 and 53× faster
  on M6, with no mismatches in 118,089 audited cases.

**Planner pilots (2026-10-10, selection seeds from 122000).** Value networks
were trained on outcomes.

| Stage | Planner | Fights | Planner | Best heuristic |
|---|---|---:|---:|---:|
| M5 | One-decision lookahead, first networks | 1,000 | 45.3% | 79.4% |
| M5 | Same, after one self-play round | 400 | 52.0% | 81.8% |
| M5 | Exact rest-of-turn rollout, afterstate network | 100 | 54% | 81% |
| M5 | Same, deviating only by more than 3 points | 100 | 80% | 81% |
| M6 | Exact rest-of-turn rollout | 20 | 75% | 70% |

The search chose actions the networks misjudged: Dodge instead of attacks, and
ending turns with actions unused. The gaps behind those choices were within the
networks' error.

**Guided planning and expert iteration (shelved 2026-10-15).** The planner
combined policy-prior proposals, the heuristic as an anchor, exact enemy phases
and a 3-point deviation margin. On seeds 122000–122099 it lost to the anchor:

- M5: 66 vs 81 of 100 (−15 [−25.9, −4.1]).
- M6: 64 vs 83 of 100 (−19 [−29.7, −8.3]).

The review found the pilot did not fairly test expert iteration, for three
reasons:

- **Teacher mismatch.** Data was collected with every action scored (37/100 on
  M5) but evaluated with top-3 plus the anchor (70/100).
- **Too little data.** The iterated network was trained on only 60 fights.
- **Margin didn't hold.** The fixed margin still let it deviate at about 13% of
  decisions.

The conclusion is "this configuration loses", not "expert iteration can't work".
The code and full review are on branch `archive/m5-m6-guided-planning`.

## Open lines (2026-10-15)

| Line | Status |
|---|---|
| Headroom probe: heuristic improved by many rollouts per action, with a significance gate | Proposed; seeds 220000–229999 reserved |
| Mining simulator logs into tactics for `GAME_KNOWLEDGE.md` | Completed 2026-10-15: [seven evidence-graded BG3 hypotheses](GAME_KNOWLEDGE.md#simulator-tactical-evidence-hypotheses-for-bg3), [reproducible analysis](../evaluation/sim_mining.py); 13 paired contrasts on seeds 232000–235999 |
| Simulator behind the BG3 snapshot and command interface | In progress with the harness work |
| PPO tuning sweep, cloning the heuristic then PPO | Not started; waits on the headroom probe |

## Removed reports

These were removed on 2026-10-15; read them with `git show b082262:reports/<name>`.

- **Held-out evaluations:** `m0_heldout_3000_3999.md`, `m1a_heldout_3000_3999.md`,
  `m1b_heldout_3000_3999.md`, `m2_heldout_3000_3999.md`, `m3_heldout_3000_3999.md`,
  `m4_heldout_3000_3999.md`.
- **M2 follow-ups:** `m2_ppo_1m_followup.md`, `m2_reward_shaping.md`.
- **M4 search:** `m4_expectimax.md`, `m4_rollout_pilot_18000_18099.json`.
- **M5/M6 stages and their review:** `m5_m6_decision_stages.md`,
  `m5_m6_adversarial_review.txt`.
- **M5/M6 planning:** `m5_m6_planning.md`.
- **Phase summary:** `sim_phase_summary.md`.
- **Review scripts:** `audit_m5_m6.py`, `review_m5_m6.py`, `review_m6_ablation.py`,
  `review_ppo_interventions.py`.
