# M5/M6 headroom probe

2026-10-15: work starts at b082262. No combat mechanics changes.

## Method and development plan

At every nonforced decision, score every legal action against the tuned M5 or
adaptive M6 anchor under that same heuristic continuation. Common simulation
seeds come only from the planner's RNG. Exact one-sided McNemar tests use
Bonferroni alpha/(alternatives × prespecified looks); adaptive futility can
only return the anchor. This controls a decision's familywise type-I error,
not an episode-wide guarantee. No normal zero-variance significance gate.

Initial pilot: seeds 220010–220013, each stage, alpha .01, looks 16/64/256,
futility margin zero, planner master seed 20261015. Tests use 220000–220004.
Each episode gets an ordinal-based planner seed independent of combat seed,
invariant to worker scheduling. Measure process CPU and wall time before
choosing selection configurations and confirmation sample sizes.

Confirmation analysis (configuration and sample size not yet locked): paired
encounter outcomes, two-sided exact McNemar, paired normal approximate CI plus
conservative boundary-aware >=95% CI from Bonferroni Clopper–Pearson bounds on
the two discordant probabilities. Record decisions, fight deviations, action
pairs and estimated advantages. Uniformly sample 12 deviations for independent
1024-pair rescoring using master seed 20261016. Encounter pairing fixes initial
encounters, not all realized dice after policy divergence.

Only 220000–229999 is authorized for encounter collection; 130000–132999 is
untouched. No confirmation data will be collected until configuration and
analysis are committed here. Raw diagnostics stream to ignored
runs/headroom_probe/, including per-episode process CPU time.

## Selection declaration

M5 pilot: 4 fights, 62.21 CPU seconds, 15.55 seconds/fight, no deviations.
Compare two conservative budgets on the same fresh 12 selection encounters per
stage (221000–221011): A alpha=.01, looks16/64/256; B alpha=.05,
looks16/64/512. Both futility margins zero. Selection favors a configuration
that generates deviations independently validated with fresh rollouts, subject
to projected confirmation cost <22 CPU-hours (leaving ~2 hours for pilots,
selection, tests and checks); no promise to optimize 12-fight win rates.

Selection tie rule (declared before inspecting selection outcomes): prefer B's
larger budget/higher power if its projected 1,000-fight cost per stage fits the
study budget and its independently rescored deviations do not have a mean
advantage whose normal 95% Monte Carlo interval is wholly negative. Otherwise
choose A under the same check; if neither passes report the issue instead of
running a nominally confirmatory unsafe probe. If there are no selection
deviations, prefer B and explicitly report low observed deviation/power.
Confirmation is 1,000 encounters/stage, fixed before confirmation outcomes.

## Additional power/efficiency selection (before collecting fresh C data)

A produced two deviations on M5 (one encounter), so the gate is capable of
switching. Early A/B outputs suggest low deviation rates. Add configuration C,
alpha=.05, looks32/128/1024, candidate-wise nonpositive futility: score every
legal candidate at 32, then only keep sampling candidates with positive paired
sample advantage. The anchor receives the full stream; candidates always use
its exact matching seed prefixes. Keep the original alternatives × three looks
Bonferroni denominator. This only suppresses rejection events and preserves
per-decision error control. Diagnostics record sample counts per action.

C uses fresh selection encounters 221100–221111 per stage. Prefer C over B if
its projected two-stage 1,000-fight confirmation fits within 22 CPU-hours and
its fresh-rollout deviation validation is not significantly negative; choose
B/A by the prior rule otherwise. This preference favors power/efficiency,
not selection encounter win rates. No confirmation encountered yet.
