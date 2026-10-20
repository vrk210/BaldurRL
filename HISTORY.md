# BaldurRL experiment history

One entry per exploration or experiment, numbered in the order recorded. The
rules for entries are in `AGENTS.md` ("Evidence") on `main`; `docs/RESULTS.md`
there caches the numbers current decisions rest on and cites entries by ID.
Nothing here is merged into `main`.

**Dates** are when the result was recorded, by report or commit; the runs may
be a day or two earlier.

**To append**, from any clone:

```bash
git fetch origin history
git worktree add ../history history   # an orphan branch: it shares no files with main
# add "## H<n>: <title> (<YYYY-MM-DD>)" at the end, n one more than the last entry,
# with the fields Motive, Run, Outcome, Learned, Irregularities, Open leads, Artifacts
git -C ../history commit -am "Add H<n>: <title>"
git -C ../history push origin history
```

If the push is rejected, rebase onto `origin/history`, renumber your entry and
push again; never force-push. A superseded entry gains one line,
`- **Superseded by:** H<n>`, and is otherwise left as written.

Where artifacts live:
- **Reports** retired on 2026-10-15: list them with
  `git ls-tree --name-only b082262 reports`, read one with
  `git show b082262:reports/<name>`.
- **Archived code** sits under `archive/*` tags: `git show <tag>:<path>`.
- **Raw runs** are git-ignored and local to the owner's Mac.
  - `BaldurRL/runs/` is in the main checkout.
  - `~/BaldurRL-archive/<worktree>/` holds the `runs/` and PROGRESS logs of the
    retired worktrees, moved there on 2026-10-20.

Seed ranges are registered in `evaluation/seeds.py` on `main`.

## H1: M0 PPO against a hand heuristic (2026-09-30)
- **Motive:** check that MaskablePPO learns the fixed 1v1 Fighter-vs-Goblin
  fight before adding difficulty.
- **Run:**
  - commit `17edb95`;
  - MaskablePPO with terminal ±1 reward, 4 environments, `train.py` defaults
    and deterministic masked prediction (the setup of every later PPO entry);
    50k and 200k steps, training seed 0, model selection on 1000–1199;
  - held-out seeds 3000–3999, n=1,000 per policy.
- **Outcome:** random 71.7%, heuristic 97.5%, PPO 97.1% (50k) and 98.0% (200k).
- **Learned:** M0 is at the ceiling. PPO delays Action Surge (87.8% of fights,
  against 98.8% for the heuristic) with no outcome difference.
- **Irregularities:** first divergences describe behavior. After two policies
  diverge they consume the dice stream differently, so a divergence isn't a
  causal estimate.
- **Open leads:** none.
- **Artifacts:** `b082262:reports/m0_heldout_3000_3999.md`; `BaldurRL/runs/heldout/`.

## H2: M1A random opponent and M1B target choice (2026-10-02)
- **Motive:** see whether a randomized opponent (M1A) and a second enemy (M1B)
  separate PPO from the heuristic.
- **Run:**
  - commit `11a88c2`;
  - PPO 200k, training seeds 0–2;
  - held-out 3000–3999, n=1,000;
  - predeclared fresh range 4000–8999, n=5,000.
- **Outcome:**
  - **M1A:** random 73.5%, heuristic 95.0%; PPO 95.7 / 96.2 / 95.9%. Fresh range: heuristic
    96.00%, PPO 95.80 / 96.02 / 95.84%.
  - **M1B:** random 23.7%, heuristic 67.5%; PPO 66.0 / 67.7 / 69.3%. Fresh range: heuristic
    67.86%, PPO 67.90 / 68.42 / 69.92%.
- **Learned:**
  - M1A is saturated.
  - M1B's target decision exposes variation between training seeds: one seed's
    2-point lead didn't reproduce in the other two.
- **Irregularities:** none recorded.
- **Open leads:** none.
- **Artifacts:** `b082262:reports/m1a_heldout_3000_3999.md`,
  `m1b_heldout_3000_3999.md`; `BaldurRL/runs/m1a_heldout/`, `m1b_heldout/`,
  and `extended_5000/` for the fresh range.

## H3: M2 one-use Cleave, PPO 200k (2026-10-02)
- **Motive:** a first resource-timing decision. Does Cleave separate the
  policies?
- **Run:** commit `11a88c2`; PPO 200k on seeds 0–2; held-out 3000–3999; fresh
  range 4000–8999.
- **Outcome:**
  - Held-out: random 28.4%, heuristic 65.0%, PPO 66.6 / 64.8 / 66.3%.
  - Fresh range: heuristic 66.32%, PPO 67.76 / 67.18 / 68.28%.
- **Learned:**
  - The clearest behavioral difference is the opening: in 537 of the losses or
    truncations separating the heuristic from PPO seed 0, the heuristic opened
    with Cleave where PPO attacked. Whether that choice costs fights is tested
    in H5.
  - Two PPO seeds Cleave in under a quarter of fights, mostly against a single
    enemy.
  - Cleave itself didn't raise win rates: the heuristic scored below its M1B
    rate.
- **Irregularities:** none recorded.
- **Open leads:** was 200k steps enough? Answered in H5.
- **Artifacts:** `b082262:reports/m2_heldout_3000_3999.md`; `BaldurRL/runs/m2_heldout/`,
  `extended_5000/`.

## H4: M2 bounded damage shaping (2026-10-02)
- **Motive:** test whether a small dense reward improves sample efficiency.
- **Run:**
  - Commit `11a88c2`.
  - Terminal ±1 plus `0.2 × damage dealt / total enemy max HP`, bounded at 0.2
    per episode;
  - PPO 200k on seeds 0–2;
  - checkpoints at 25k, 50k, 100k and 200k, scored on 1000–1199;
  - final models on 3000–3999 and 4000–8999.
- **Outcome:**
  - The shaped mean never exceeded the terminal mean at any checkpoint.
  - Final on 3000–3999: terminal 65.90%, shaped 66.03%.
  - Fresh range: terminal 67.76 / 67.18 / 68.28%; shaped 67.62 / 68.66 / 66.30%.
- **Learned:** this shaping improved neither sample efficiency nor final win
  rate at this scale.
- **Irregularities:** 200-episode checkpoint evaluations are noisy.
- **Open leads:** none.
- **Artifacts:** `b082262:reports/m2_reward_shaping.md`; `BaldurRL/runs/m2_ppo_damage_s*/`,
  `extended_5000/`.

## H5: M2 training budget and the opening-Cleave diagnostic (2026-10-02)
- **Motive:** find whether H3's 200k runs stopped before learning.
- **Run:**
  - commit `aa08df0`;
  - PPO 1M from scratch, seeds 0–2;
  - fresh seeds 9000–13999, n=5,000;
  - a matched-start diagnostic on seeds 2000–2199: 3 opening actions × 50
    shared roll seeds, with heuristic continuation and Cleave disabled.
- **Outcome:**
  - 1M PPO scored 68.68 / 69.48 / 69.82% against the heuristic's 65.12%.
  - The same heuristic without Cleave scored 67.98%.
  - The best-validation checkpoints scored 69.48 / 69.46 / 69.74%.
  - Opening with Attack enemy 1 beat opening with Cleave by +0.59 to +3.01
    points (95% interval over the 200 encounters).
- **Learned:**
  - 200k left win rate on the table. Checkpoint selection wasn't the cause.
  - PPO's gain comes from avoiding the opening Cleave, not from using Cleave
    well: the 1M models often Cleave a single enemy, which is dominated.
- **Irregularities:** none recorded.
- **Open leads:** the optimum. Answered in H6.
- **Artifacts:** `b082262:reports/m2_ppo_1m_followup.md`;
  `BaldurRL/runs/m2_ppo_1m_s*/`, `m2_1m_fresh_9000_13999/`,
  `m2_initial_action_diagnostic.json`.

## H6: M2 exact optimum (2026-10-09)
- **Motive:** an exact ceiling for M2, against which to measure the regret of
  every policy.
- **Run:** a dynamic-programming solver over the M2 state space, on seeds
  9000–13999. No commit: the scripts are in the run folder.
- **Outcome:**
  - Expected optimal win rate 71.79%. The heuristic's exact expected win rate on the same seeds is 66.03%; it won 65.12% when played (H5).
  - Regret per episode on 9000–9399: heuristic 0.0561; PPO 1M seed 2 0.0162.
- **Learned:**
  - PPO comes within about 2 points of the optimum.
  - The solver later checked H10's exact transition model to 6e-16. The
    maintained oracle, `tests/m2_oracle.py`, is a separate value-iteration
    implementation.
- **Irregularities:** the solver itself lives only in ignored `runs/`, not in
  the repository.
- **Open leads:** none.
- **Artifacts:** `BaldurRL/runs/m2_exact_optimum_9000_13999/` (`m2_dp.py`,
  `m2_regret.py`, `regret.json`).

## H7: M3 2v2 and M4 2v3 stages, PPO 1M (2026-10-06 to 2026-10-09)
- **Motive:** multi-ally stages. M4 was scaled so that random play rarely wins.
- **Run:** stages and reports added in commit `a56d477` on a side branch,
  squashed into the main line as `4060b68`; PPO 1M on seeds 0–2; held-out
  3000–3999, n=1,000.
- **Outcome:**
  - **M3:** random 86.3%, heuristic 99.2%, PPO 99.0 / 98.7 / 98.8%.
  - **M4:** random 15.6%, heuristic 61.4%, PPO 58.2 / 61.1 / 62.2%.
- **Learned:**
  - M3 is easy: PPO matches the heuristic while barely using Cleave (6–33% of
    fights).
  - M4 is a fair fight. Cleave and Second Wind matter, and survivors keep about
    14 of 40 HP.
- **Irregularities:** none recorded.
- **Open leads:** M4 training budget (H9) and search (H8, H10).
- **Artifacts:** `b082262:reports/m3_heldout_3000_3999.md`,
  `m4_heldout_3000_3999.md`; `BaldurRL/runs/m3_*`, `m4_*`.

## H8: M4 rollout planner pilot (2026-10-06)
- **Motive:** check whether one-step rollouts under the stage heuristic improve
  on the heuristic.
- **Run:**
  - `RolloutAgent` with 32 and 64 rollouts per action, common random numbers
    across actions;
  - M4 seeds 18000–18099, n=100.
- **Outcome:** heuristic 64/100; rollout-32 47/100; rollout-64 51/100.
  - A 2,048-rollout recheck of one state reversed a premature END_TURN choice.
- **Learned:** small rollout budgets lose to their own continuation policy.
- **Irregularities:** n=100 is a functional and performance pilot, not an
  estimate.
- **Open leads:** the same failure at larger scale on M5/M6, H13.
- **Artifacts:** tag `archive/rollout-planning-2026-10-06`. It holds
  `reports/m4_rollout_pilot_18000_18099.json` and the original
  `docs/ROLLOUT_PLANNING.md`. The planner itself reached `main` through #9.

## H9: M4 longer PPO training and a threat-aware heuristic (2026-10-06)
- **Motive:** check whether more PPO training, or better targeting, beats the
  M4 heuristic.
- **Run:**
  - a policy diagnostic: candidate rules designed on seeds 4000–4999 and
    evaluated on 5000–9999 (n=5,000), paired against the stage heuristic;
  - PPO continued to 5M steps on seeds 0–2, with best-validation checkpoints;
  - the resulting improved-targeting ("threat-aware") heuristic;
  - test seeds 10000–14999, n=5,000;
  - no commit: the scripts and `source_hashes.json` are in the run folder.
- **Outcome:**
  - **Diagnostic** (5000–9999): stage heuristic 62.34%; targeting only 65.14%
    (+2.80, SE 0.50); targeting plus heal at 14 and Cleave at 3 enemies 65.26%
    (+2.92, SE 0.62); PPO 1M 59.54 / 61.02 / 60.20%. The targeting rule
    picks the enemy with the highest expected damage per attack against our AC
    16, divided by the number of our attacks needed to kill it.
  - **Test** (10000–14999): stage heuristic 61.24%; threat-aware heuristic
    63.66%.
  - PPO 1M mean 61.45%; 5M mean 61.79% (+0.33 [−0.61, +1.27]).
  - Best-validation checkpoints: mean 62.53%; best 63.22% (seed 0).
  - 5M mean minus threat-aware: −1.87 [−3.23, −0.51].
  - H10 used seed 0's best-validation checkpoint, the highest of the three on
    these test seeds.
- **Learned:**
  - Extra PPO training barely helps on M4.
  - The threat-aware heuristic became the M4 reference policy:
    `M4ThreatAwareAgent` reproduces it exactly.
- **Irregularities:** the diagnostic's seeds 4000–9999 were not registered as
  an M4 use until 2026-10-20. The test intervals condition on these three
  fitted models.
- **Open leads:** search with an exact model, H10.
- **Artifacts:** `BaldurRL/runs/m4_policy_diagnostic_5000_9999/` (`results.json`,
  `m4_policy_probe.py`); `BaldurRL/runs/m4_longer_training/` (`comparison.json`).

## H10: M4 expectimax with exact chance nodes and a learned leaf (2026-10-10)
- **Motive:** M4 has exact, enumerable dice. Check whether exact search plus a
  learned value beats every baseline.
- **Run:**
  - Exact `RosterTransitionModel`, verified three ways:
    - exhaustive RNG-path enumeration through `env.step` (14 states, to 1e-12);
    - 65 cases × 200k samples, with no sample outside the enumerated support;
    - the H6 solver (to 6e-16) and the independent `tests/m2_oracle.py` (to
      1e-12).
  - Expectimax at depth 1 and depth 2 (10k-node budget).
  - Leaves: the race leaf, a hand-built damage-race estimate; V0, a value
    network trained on 22,000 heuristic and PPO games; V1, V0's data plus
    4,000 of the planner's own games.
  - Commit `8fe23ac`. Seeds: M4 data 20000–39999 and 60000–67999; M2 data
    20000–41999; validation loss 40000–41999; selection 42000–43999; test
    10000–14999, n=5,000.
- **Outcome**, paired against threat-aware (63.66%):

  | Planner | Win rate | Points vs threat-aware [95% CI] | ms per planned decision |
  | --- | ---: | --- | ---: |
  | Depth 1, race leaf | 48.08% | −15.58 [−17.32, −13.84] | 66 |
  | Depth 1, V0 | 62.88% | −0.78 [−2.45, +0.89] | 39 |
  | Depth 1, V1 | 65.50% | +1.84 [+0.19, +3.49] | 42 |
  | Depth 2 (10k-node budget), V1 | 66.64% | +2.98 [+1.34, +4.62] | 299 |
  | Stage heuristic | 61.24% | −2.42 [−3.40, −1.44] | <0.01 |
  | PPO 1M seed 2 | 62.26% | −1.40 [−2.88, +0.08] | 0.1 |
  | PPO 5M, best checkpoint (H9) | 63.22% | −0.44 [−2.11, +1.23] | 0.1 |

  - The worst-case decision took about 4.6 s, at depth 1.
  - On M2 (9000–9999, n=1,000, regret per episode against H6's optimum): exact
    optimum 71.16%; depth 1 + learned M2 V0 70.13% (0.0193); depth 1 + race
    leaf 60.14% (0.0948); M2 heuristic 63.94% (0.0594). On 9000–9399, depth 2
    cut the learned leaf's regret about 3× (0.0130 → 0.0040).
- **Learned:**
  - Search wins only where exact enumeration comes before the leaf.
  - The search exploits leaf errors.
  - Lower validation loss can play worse: V0's was 0.4135 against V1's 0.4159,
    yet V0 played 4.65 points worse on the selection seeds. Select on play.
  - Search-value and λ-return targets never beat outcome targets. A third round
    and an ensemble didn't help.
- **Irregularities:** winner's curse. V1's edge shrank from +2.90 on selection
  seeds to +1.84 on test.
- **Open leads:** carry the method to M5/M6, H14.
- **Artifacts:** `b082262:reports/m4_expectimax.md`;
  `~/BaldurRL-archive/planning-search/` (PROGRESS, `runs/`). The code is on `main`.

## H11: M5/M6 decision-rich stages: design, tuning, heuristics and PPO (2026-10-10)
- **Motive:** M4 left little room between policies. Add decisions with more
  headroom: enemy roles, Trip/Prone, Dodge, advantage, and on M6 ranks,
  Advance and opportunity attacks.
- **Run:**
  - Targets set before measuring: naive heuristic between 35% and 65%; a spread
    of at least 8–10 points to the best heuristic.
  - v1 failed them (M5 naive 75.9%, Trip spam 94.0%), so there was one retune
    on dev seeds 20000–21999:
    - Trip limited to 2 charges per ally;
    - Healer 2d6+2;
    - +1 attack and damage for the Brute and Archer;
    - on M6, Brute HP 20–26 and 2 Healer charges.
  - Commit `6316abe`. Held-out 3000–7999, n=5,000; PPO 1M on seeds 0–2.
  - The heuristic variants (naive, trip_focus, healer_first, tuned; hold,
    dive_all, adaptive) are defined in `VARIANTS` in
    `agents/tactical_heuristics.py`.
- **Outcome:**
  - **M5:** random 6.70%, naive 61.44%, trip_focus 76.62%, healer_first 73.86%, tuned
    78.52%; PPO 74.90 / 74.68 / 73.56%.
  - **M6:** random 6.20%, naive 40.92%, hold 78.88%, dive_all 79.36%, adaptive 80.60%; PPO
    75.74 / 75.36 / 74.52%.
  - Paired differences in points (SE), on the same 5,000 seeds:
    - M5: tuned − naive +17.08 (0.86); Trip alone +15.18 (0.88); lowest-HP
      focus − Healer-first +2.76 (0.69); tuned − trip_focus +1.90 (0.43);
      pooled PPO − tuned −4.14 (0.60).
    - M6: adaptive − naive +39.68 (0.87); dive_all − hold +0.48 (0.81);
      adaptive − hold +1.72 (0.59); pooled PPO − adaptive −5.39 (0.61).
- **Learned:**
  - Most of the spread is "use Trip". Among strong policies the headroom is
    1–5 points.
  - Tactics that won in these rules, as BG3 hypotheses:
    - focus the lowest-HP target; Healer-first loses 2.8 points because the
      Healer's self-heal absorbs focus fire;
    - follow a Trip with an attack on the Prone target the same turn: the
      competent policies do so about 90% of the time, PPO 60–76%;
    - on M6, dive against durable Brutes (HP divided by our hit chance at least
      42) and hold against soft ones; a single diver was weak;
    - never Disengage: diving with Disengage won 51.4% on development seeds
      against 70.4% without; no final heuristic Dodges.
  - PPO plateaus by about 500k steps, and best-checkpoint selection doesn't
    close the gap.
- **Irregularities:** seeds 3000–7999 informed the retune, so they aren't an
  untouched test set. See H12.
- **Open leads:** PPO's gap (H12), planners (H13, H14), headroom (H18).
- **Artifacts:** `b082262:reports/m5_m6_decision_stages.md`;
  `~/BaldurRL-archive/decision-stages/` (PROGRESS; `runs/v2/` traces and
  `analysis/`; PPO models in `runs/{m5,m6}_ppo_1m_s*/`; `runs/v1/` is the
  superseded design and mustn't be pooled with v2).

## H12: M5/M6 adversarial review, fresh seeds and PPO diagnostics (2026-10-10)
- **Motive:** an independent check of H11 before anything was built on it.
- **Run:**
  - Commit `6316abe`. All 16 policies rerun seed by seed, 80,000 outcomes in
    all;
  - restoration, rebuilding a simulator from an observation, checked against a
    live environment on 7,285 transitions from seeds 51000–51099;
  - fresh seeds 40000–44999;
  - an M6 ablation;
  - a corrected stratified counterfactual check of PPO's disagreements, plus
    whole-fight repairs.
- **Outcome:**
  - **Reproduction:** every outcome reproduced exactly, and no simulator defect
    was found.
  - **Fresh seeds:** tuned 77.84%, adaptive 81.02%; the ranking held. Adaptive − hold was only
    +1.14 (SE 0.60).
  - **M6 headline** (adaptive − naive, +39.68): counting only reachable Cleave
    targets and using Trip takes
    naive from 40.9% to 77.7%. That is 36.8 of the 39.7 points.
  - **Supported PPO mistakes:**
    - Trip → Attack on M5: +1.53 [+0.82, +2.23];
    - Trip → Cleave on M5: +2.70 [+1.82, +3.58];
    - Trip → Attack on M6: +0.94 [+0.37, +1.51].
  - **Whole-fight repair:** fixing the M5 Trip and Cleave choices recovers 1.05
    of the 4.14-point gap. Forcing adaptive's opening plan on M6 recovers +0.44
    (SE 0.68).
- **Learned:**
  - The M6 headline is mostly two baseline fixes.
  - Most of PPO's gap remains unexplained by local decisions.
- **Irregularities:** the first diagnostic had a sampling defect. It was
  corrected and both stages rescored the same day.
- **Open leads:** none beyond H18.
- **Artifacts:**
  - `b082262:reports/`: `m5_m6_adversarial_review.txt`, `review_m5_m6.py`,
    `review_m6_ablation.py`, `review_ppo_interventions.py`, and
    `audit_m5_m6.py` (the trace audit and restoration stress check);
  - the PPO diagnostic, whose code documents its reservoir sampling and
    continuation-policy assumptions: `evaluation/ppo_diagnostics.py` at tag
    `archive/simulator-research-2026-10-15`;
  - `~/BaldurRL-archive/decision-stages/runs/review/`.

## H13: M5/M6 rollout planner pilot (2026-10-10)
- **Motive:** use rollouts as a headroom probe on the new stages.
- **Run:** commit `6316abe`; 32 rollouts per action with heuristic
  continuation; predeclared 600 M5 and 300 M6 fights.
- **Outcome:**
  - M5: 58.0% against the heuristic's 78.8% (−20.8, SE 2.5).
  - M6: 65.3% against 81.3% (−16.0, SE 3.3).
  - Rescoring the deviations with 1,024 rollouts mostly favored the heuristic.
- **Learned:** this is finite-sample selection error. Taking the best of 8–11
  noisy estimates selects the errors. It isn't evidence of a ceiling.
- **Irregularities:** none recorded.
- **Open leads:** a statistically gated probe, H18.
- **Artifacts:** `b082262:reports/m5_m6_decision_stages.md` ("Rollout planner
  pilot"); `~/BaldurRL-archive/decision-stages/runs/`.

## H14: M5/M6 exact transitions and planner pilots (2026-10-10)
- **Motive:** carry H10's exact-search method to M5/M6.
- **Run:**
  - The replay model `combat/tactical_transitions.py` drives the real
    environment with an enumerating random source. It was verified by raw path
    enumeration (241 cases, about 1e-16) and by 20,000 samples per case (288
    cases).
  - Planners: one-decision lookahead, exact rest-of-turn rollout, afterstate
    value networks trained on outcomes, and a 3-point deviation margin.
  - Commit `e146d0d`. Seeds: training 100000–119999 and 140000–199999;
    validation loss 120000–121999; selection from 122000; frozen test
    130000–132999; transition validation 51200–51202 and tests 51300–51302.
- **Outcome**, on selection seeds:

  | Stage | Variant | Fights | Planner win | Heuristic win |
  | --- | --- | ---: | ---: | ---: |
  | M5 | Lookahead, first networks | 1,000 | 45.3% | 79.4% |
  | M5 | After one self-play round | 400 | 52.0% | 81.8% |
  | M5 | Rest-of-turn rollout | 100 | 54% | 81% |
  | M5 | With the 3-point margin | 100 | 80% | 81% |
  | M6 | Rest-of-turn rollout | 20 | 75% | 70% |

- **Learned:**
  - Searching a whole turn, or through the enemy phase, was too slow with the
    replay model.
  - The search chose actions the networks misjudged, such as Dodge or ending a
    turn with actions unused; the gaps behind those choices, about 2 points,
    were within the networks' error.
  - The 3-point margin removed the damage only by reverting to the heuristic.
  - Ten times more data improved validation loss but not play.
- **Irregularities:** none recorded. The frozen test range 130000–132999 was
  not used.
- **Open leads:** a faster table model (H15), better value targets and expert
  iteration (H16).
- **Artifacts:** `b082262:reports/m5_m6_planning.md`;
  `~/BaldurRL-archive/integration/` (PROGRESS with the seed plan,
  `runs/m5m6_planner_prototypes/`, `runs/eval_m5/`, value datasets).

## H15: M5/M6 table transition model (2026-10-10 to 2026-10-15)
- **Motive:** make exact enemy-phase search affordable on M5/M6.
- **Run:**
  - `combat/table_tactical_transitions.py` (default from commit `ac74529`),
    checked against the replay model, now its oracle, on seeds 210000–210009;
  - an adversarial audit with constructed edge states, seeds 211000–211029.
- **Outcome:**
  - Same successor sets, with a largest probability difference of 1.0e-14
    (345 M5 and 346 M6 cases).
  - 16× faster on M5 and 53× on M6.
  - No mismatches in 118,089 audited cases.
- **Learned:** the per-transition cost is solved, but the branching factor
  remains.
- **Irregularities:** none recorded.
- **Open leads:** none. This became the default model.
- **Artifacts:** `b082262:reports/m5_m6_planning.md` ("Follow-up"). The audit
  and benchmark scripts are at
  `bff5f88:evaluation/tactical_transition_audit.py` and
  `bff5f88:evaluation/tactical_transition_benchmark.py`.

## H16: Guided planning and expert iteration, shelved (2026-10-15)
- **Motive:** fix H14's leaf exploitation with a policy prior, the anchor (the
  heuristic the planner extends, which it leaves only by a margin), exact enemy
  phases (via H15) and expert iteration.
- **Run:**
  - Bootstrap on 600 training and 100 validation fights per stage;
  - one expert-iteration round on 60 training and 20 validation fights,
    epsilon .10, TD(.5);
  - selection seeds 122000–122099.
- **Outcome:** after one iteration, M5 66/100 against the anchor's 81
  (−15 [−25.9, −4.1]); M6 64/100 against 83 (−19 [−29.7, −8.3]).
- **Learned:** the conclusion is "this configuration loses", not "expert
  iteration can't work".
- **Irregularities:** the review found three flaws in the test:
  - **Teacher mismatch:** collection scored every action (37/100 on M5), while
    evaluation used top-3 proposals plus the anchor (70/100).
  - **Too little data:** the network was trained on only 60 fights.
  - **Margin didn't hold:** the fixed margin still let it deviate at about 13%
    of decisions.
- **Open leads:** revisit only with one planner configuration for collection
  and evaluation, pooled training data, significance-gated deviations, and
  breadth-first budget use.
  - **Removal condition** for the archived code: when M5/M6 learned-leaf
    planning is revived in a fresh branch, or ruled out for the BG3 planner
    design.
- **Artifacts:**
  - tag `archive/m5-m6-guided-planning-2026-10-15` (code, `reports/m5_m6_guided_planning/`
    with the review and handoff);
  - tag `archive/m5-m6-planning-wip-2026-10-10` (the uncommitted working copy
    found in the worktree);
  - `~/BaldurRL-archive/m5-m6-planning/` (PROGRESS, `runs/`).

## H17: Tactics mining for BG3 hypotheses (2026-10-15)
- **Motive:** the first BG3 agent will be rule-based. Mine the simulator traces
  for an evidence-graded list of tactics, stating which might hold in BG3.
- **Run:**
  - smoke tests on 230000–230007;
  - 13 fixed paired contrasts, 4,000 encounters each, seeds 232000–235999;
  - tuned M5 or adaptive M6 against targeted changes;
  - Bonferroni simultaneous intervals and adjusted exact McNemar tests;
  - plan `e2e25d3`, source `b082262`.
- **Outcome:** concentrated damage, early Surge and coordinated positioning
  helped these simulator policies. Each row is the baseline (tuned on M5, 78.45%;
  adaptive on M6, 79.75%) minus the variant, in points, with the simultaneous
  95% interval:

  | Stage | Variant | Variant win rate | Baseline − variant |
  | --- | --- | ---: | --- |
  | M5 | Surge never used | 47.27% | +31.18 [+28.31, +34.04] |
  | M5 | split focus | 56.07% | +22.38 [+19.78, +24.97] |
  | M5 | Surge withheld to round 2 | 66.75% | +11.70 [+8.91, +14.49] |
  | M5 | Healer first | 74.40% | +4.05 [+1.93, +6.17] |
  | M5 | Trip the threat target, not the focus | 76.33% | +2.12 [+0.01, +4.24] |
  | M5 | emergency Dodge at 75% of HP incoming | 76.75% | +1.70 [+0.48, +2.92] |
  | M5 | Second Wind at 10/20 HP, not 12 | 76.80% | +1.65 [+0.51, +2.79] |
  | M5 | Cleave at 2 targets, not 3 | 77.48% | +0.97 [−0.09, +2.04] |
  | M6 | Disengage before a planned dive | 70.58% | +9.18 [+7.11, +11.24] |
  | M6 | stop after one diver | 72.97% | +6.78 [+4.88, +8.67] |
  | M6 | emergency Dodge at 75% of HP incoming | 76.17% | +3.57 [+2.07, +5.08] |
  | M6 | hold everywhere | 77.75% | +2.00 [+0.03, +3.97] |
  | M6 | dive everywhere | 77.95% | +1.80 [+0.06, +3.54] |

  How each finding is graded as a BG3 hypothesis is in `docs/GAME_KNOWLEDGE.md`
  on `main`.
- **Learned:** these are hypotheses for BG3, not verified tactics.
- **Irregularities:** the bundle mixes historical trace summaries with
  confirmation results; they aren't interchangeable samples.
- **Open leads:** the Cleave threshold, earlier Advance timing, and an
  optimal-policy ceiling.
- **Artifacts:**
  - tag `archive/simulator-research-2026-10-15`:
    - `evaluation/sim_mining_summary.json`, SHA-256
      `ee8f8840b949c1632a921d239dc72004d2b336468d30cd803a9ab628c6c2fe75`;
    - the mining CLI, which depends on historical worktree paths.
  - The 22 traces in its inventory moved from
    `BaldurRL-worktrees/decision-stages/runs/v2/` to
    `~/BaldurRL-archive/decision-stages/runs/v2/`. Size and SHA-256 were
    verified on 2026-10-20.
  - `~/BaldurRL-archive/sim-tactics-playbook/` (`runs/`, `reports/sim_mining/`);
    the prompt is in `~/BaldurRL-archive/prompts/`.

## H18: Paired headroom probe (2026-10-15)
- **Motive:** before spending more on M5/M6 policies, measure how much above the
  best heuristic is reachable, using a one-step improvement probe that switches
  away from the heuristic only on strong evidence.
- **Run:**
  - Predeclared revision `6c2162e`;
  - smoke tests on 220000–220007, throughput pilots on 220010–220013, and
    configuration selection on 221000–221011 and 221100–221111;
  - alpha .05 with sampling looks at 32, 128 and 1,024 rollouts per action
    (the gate may decide at each look);
  - common rollout seeds, and exact paired gates adjusted over all alternatives;
  - a once-only confirmation on 223000–223999, 1,000 encounters per stage;
  - 18.44 CPU-hours, excluding tests and process startup.
- **Outcome:**

  | Stage | Probe | Heuristic | Difference | Conservative 95% CI | Approximate 95% CI | Exact McNemar p |
  | --- | ---: | ---: | ---: | --- | --- | ---: |
  | M5 | 80.3% | 78.6% | +1.70 | [−1.23, +4.61] | [−0.08, +3.48] | .07842 |
  | M6 | 81.6% | 79.6% | +2.00 | [−1.03, +5.02] | [+0.14, +3.86] | .04460 |

  - The probe changed 421/28,557 M5 and 387/28,615 M6 decisions, touching 32.6%
    and 29.4% of fights.
  - Both conservative intervals include zero; their upper bounds apply to this
    tested policy. M6's exact test and approximate interval are nominal
    pointwise evidence.
  - Rescoring 12 deviations per stage with 1,024 fresh pairs gave mean local
    gains of +4.96 and +3.17 points, each with conditional Monte Carlo SE 0.45.
- **Learned:**
  - No whole-fight improvement and no general ceiling is established.
  - The per-decision gate is not a whole-episode guarantee.
- **Irregularities:** p values are unadjusted for the two stages. Local
  rechecks exclude state-sampling uncertainty.
- **Open leads:** the local exceptions are worth revisiting only once real BG3
  observations support the action/value interface. Don't rerun the
  confirmation as fresh evidence, and don't select another configuration on
  these encounters.
- **Artifacts:**
  - tag `archive/simulator-research-2026-10-15`: `agents/improved_heuristic_agent.py`,
    and `evaluation/headroom_probe_results.json`, SHA-256
    `9ec50f42ecd9ca228546f7014e3098a83af55d739b7e8af32e261a8dbaa1cbcd`;
  - `~/BaldurRL-archive/headroom-probe/runs/`;
  - prompt in `~/BaldurRL-archive/prompts/`.

## H19: Simulator phase review (2026-10-15)
- **Motive:** decide whether more abstract simulator difficulty brings the
  project closer to playing BG3.
- **Run:** commit `ac74529`; a synthesis of H1–H18.
- **Outcome:** the simulator phase closed. M0–M6 stay as a test bed and as a
  stand-in game behind the harness. The next simulator work should be calibrated
  to logged BG3 fights. The PPO tuning sweep and the scale-up of the planner
  were never started.
- **Learned:**
  - PPO is competent, not superior: within about 2 points of the M2 optimum,
    and 4–5 points below good heuristics on M5/M6, with default hyperparameters
    only.
  - Search with an exact model was the only method that beat the heuristics,
    and only on M4.
  - Search exploits value errors, whether the leaves are hand-built, noisy
    rollouts or learned networks.
  - Good hand heuristics are strong baselines.
  - Evaluation discipline caught several conclusions a 200-fight check would
    have gotten wrong.
- **Irregularities:** none recorded. The retirement inspection was a
  consistency check, not a rerun:
  - neither committed JSON bundle contains non-finite values;
  - the mining inventory's 22 trace paths existed;
  - the headroom audit counters reconcile, with no outcome mismatch where the
    policies never switched.

  Before relocating or deleting raw traces, use the tagged inventory and hashes
  to keep the smallest useful reproducer. (On 2026-10-20 the traces moved to
  `~/BaldurRL-archive/` with sizes and hashes verified; see H17.)
- **Open leads:** a planner for BG3 needs a transition model of the real game,
  which needs logged fights first.
- **Artifacts:** `b082262:reports/sim_phase_summary.md`; the never-run
  planning-scale task is in `~/BaldurRL-archive/prompts/`, and the data staged
  for it in `~/BaldurRL-archive/planning-scale/runs/value/` is a copy of H10's.
