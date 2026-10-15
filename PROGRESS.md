# Simulator tactics mining — predeclared plan (2026-10-15)

Base b082262. Existing runs are read-only. No combat rules or existing policy
variants change; named research variants live in the one-off mining script.

## Confirmation protocol (declared before any new experiment)

Run each fixed policy once on every encounter 232000–235999, identically paired
by seed within stage. Baselines are M5 tuned and M6 adaptive. Two-sided exact
McNemar tests, paired normal 95% CIs, and Bonferroni simultaneous 95% CIs across
the following 13 comparisons; only exclusions of zero in the simultaneous CI
and adjusted p<.05 merit a new grade-A claim. No reruns, threshold tuning or
optional stopping. Counts and full per-seed results are retained.

M5 single-rule changes: (1) highest-HP target for ally 1, lowest for ally 0
(split focus); (2) Healer-first targeting with tuned resources; (3) Action Surge
held until round 2; (4) Second Wind threshold 10 rather than 12; (5) Cleave at
2 rather than 3 living targets; (6) Trip greatest estimated threat rather than
focus; (7) existing incoming-damage Dodge trigger enabled; (8) never spend
Action Surge (turn-hygiene cost).
M6 changes: (9) adaptive opening trigger, but stop advancing once one living ally is deep
(single-diver plan);
(10) Disengage before an otherwise planned dive when Surge remains;
(11) incoming-damage Dodge trigger; (12) hold everywhere; (13) dive everywhere.
Direction hypotheses: baseline better except Dodge (unknown); conditional plan
should benefit durable Brutes but not soft ones. The two prespecified M6 strata
(opening Brute HP / normal hit chance >=42, <42) are descriptive secondary
analyses, not additional confirmed claims.

## Read-only mining

Stream all final-policy traces, not v1; summarize first kills, damage in excess
of target's remaining HP, END_TURN with legal Attack/Surge/Second Wind/Trip,
and encounter features. PPO final models s0–s2 remain separate for outcomes;
pooled comparisons average models within encounter. Report both unique PPO
wins and unique heuristic wins. Inspect opening-feature strata (durable Brute,
Archer HP<=14, >=2 WEAKEST enemies, total opening enemy HP>=60); all trace
stratification is grade C, with pointwise paired CIs, not causal or confirmatory.
Recompute quoted historical contrasts and corrected local diagnostic values
from preserved raw outputs wherever available.

## Fixed-policy dice replay (not optimal loss attribution)

Choose 32 baseline wins and 32 baseline losses per stage by evenly spaced
indices in sorted old encounter seeds (3000–7999). Restore each opening
observation and replay the same baseline with the same 128 combat RNG seeds
236000–236127. This estimates fixed-policy win probabilities and whether old
losses can win under other dice; it cannot identify truly unwinnable fights or
fraction of losses attributable to decisions. No action-value maxima and no
training. Record binomial uncertainty, sampled-state selection and costs.

Seeds: exploration 230000–230007 smoke validation; 230008–231999 unused; confirmation 232000–235999;
spare 236000–236127 combat RNG only (no newly sampled encounters).
Compute: serial worker, compact outcomes only, target <3 CPU-hours.

Deliverables: <=2-page main playbook, separate inventory/provenance appendix,
committed reproduction script; raw outputs runs/sim_mining (ignored).

Pre-run clarification: split_focus measures ally-1 target-priority change,
not a pure general focus-fire effect. one_diver retains adaptive durability
trigger; only the number of divers changes.
