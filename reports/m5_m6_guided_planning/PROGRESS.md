# M5/M6 planning follow-up

Implementation branch: m5-m6-planning, based on integration e146d0d. The original
integration worktree has merge conflicts and is left untouched.

Seed allocation declared before this follow-up's experiments:
- Transition verification / performance only: 210000–210099.
- Training / explicit exploration: 180000–189999 (inside the original training allocation).
- Value/policy validation, loss only: 120000–121999.
- Configuration selection: 122000–123999 (already used for selection by the original pilots).
- Frozen final test 130000–132999 remains unused unless a candidate improves selection performance.

Work: table-driven exact transitions validated against the replay oracle;
bounded search with exact enemy phases, policy priors and conservative heuristic
anchor; expert-iteration data collection and fitting value/policy networks.

Completed pilot:
- Oracle/table comparison: M5 345 cases, M6 346; successor sets equal, max error ~1e-14.
- Bootstrap: 600 training fights + 100 validation fights per stage.
- Expert iteration: 60 training fights + 20 validation fights per stage, epsilon .10, TD(.5).
- Selection 122000–122099: anchors M5 81%, M6 83%; v1 planners 66%, 64%.
- No final frozen-test evaluation; no PPO updates.
- Full suite: 385 passed, 1 skipped. See reports/m5_m6_planning.md for interpretation.
