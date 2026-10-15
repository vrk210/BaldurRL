# RL and simulation work alongside Windows preparation

Assessment based on integration cd7dca1 and the existing M4/M5/M6 reports. This is an experiment proposal, not a claim of newly trained models or game transfer. The Windows-preparation branch remains based on main and does not change simulator mechanics, models, or training code.

## Why further simulation work is useful

M5/M6 final PPO models win about 74-75% versus about 79-81% for the strongest hand heuristics. Checkpoint curves plateau, but only default PPO configurations were tried: this does not establish a limit of PPO or of learning. Existing whole-fight interventions explain only part of the gap. The simulator can generate labeled decisions and counterfactual continuations cheaply without waiting for human gameplay.

The purpose of live data initially is to validate IDs, legality, resources, event timing, and the simulator's supported subset. It is not automatically a clean offline RL dataset: snapshot intervals are not necessarily decisions, legal sets and terminal conditions can be unknown, and accepted inputs are not resolved actions.

## Ordered, bounded experiments

1. Audit data provenance after cd7dca1. Check non-finite search_value rates conditioned on whether a planner actually selected the decision. NaN is expected for random/non-search decisions and is not by itself corruption. Regenerate only affected search-target data. Rerun M6 comparisons affected by the turn-depth fix. Keep historical reports identified by the code revision that generated them.
2. Train a legal-mask-aware behavior-cloning policy from M5 tuned / M6 adaptive teachers, using the same policy network size as a baseline PPO run first. Evaluate whole episodes as well as held-out action agreement. Reserve new seeds before collection; do not consume the already reserved final test range 130000-132999 during model selection.
3. Compare cold-start PPO, cloning alone, and cloning-initialized PPO with matched environment-step budgets and at least three training seeds. Select configurations on selection seeds only, then use one final paired encounter-seed evaluation. Report uncertainty over encounter seeds and variation across retraining seeds separately. A clone matching the teacher establishes a competent starting point, not an improvement over the teacher.
4. If the clone fails under its own trajectories, aggregate teacher labels on learner-visited simulator states (DAgger-style), rather than only fitting a fixed teacher-state dataset. Teacher labels remain simulator labels. DAgger reference: https://arxiv.org/abs/1011.0686 .
5. Only then test a bounded policy-guided search/expert-iteration prototype. Current M5/M6 search can be slow and exploits inaccurate value estimates; a search policy is not an expert until measured to beat its teacher. Use M2's exact oracle to validate the training/evaluation pipeline, then focus search effort on ambiguous M5/M6 decisions. Record per-action ranking/regret or paired rollout evidence, not value calibration alone. Expert Iteration reference: https://arxiv.org/abs/1705.08439 .

## Defer until motivated by evidence

Do not expand to more invented simulator mechanics solely to create new benchmarks. Do not spend the pre-Windows period on unbounded PPO sweeps, universal recurrent policies, or a rewrite into a learned world model. Simulator observations are currently sufficient for their specified state; memory is not the first diagnosed bottleneck. Entity-centric policies may later help variable parties and targets, but must preserve initiative/order semantics; blindly permuting slots changes the problem.

Real BG3 traces should become conformance fixtures: measure known preconditions, action identity, costs, outcomes and turn boundaries; specify and implement mismatches one at a time. Do not expect one sampled BG3 roll to match an independently seeded simulator roll. No broad sim-to-real policy claim follows from improved simulator win rate.

## Allocation

Primary effort before Windows: deployable, inspectable observation/trial harnesses. One bounded research lane in parallel: audit -> clone -> PPO fine-tune. No new training is launched by this plan. A different RL algorithm is a later controlled comparison after these baselines, not a prerequisite for the first autonomous combat encounter.

## Read-only local artifact spot check

On 2026-10-15, inspected only `search_value` arrays with `numpy.load(..., allow_pickle=False)` in the integration worktree's git-ignored artifacts:

- Five M4 files under `runs/value/data_r1` and `data_r2`: 203,421 decision rows, all search values finite.
- Five M5 files `runs/value_m5/data/r1_x_{113000,113600,114200,114800,115400}.npz`: 93,372 decision rows across 3,000 episodes, all search values non-finite.

This is a spot check, not a complete provenance audit. The M5 filenames alone do not prove the epsilon bug caused missing values; a producer without diagnostics or non-search decisions can also yield NaNs. Outcomes and actions remain separately present. Check the actual collector configuration before deciding whether search-target retraining or regeneration is warranted. Outcome-only training is not invalidated merely by missing search diagnostics. Do not overwrite or relabel historical data.
