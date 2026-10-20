# M5 M6 implementation review and PPO tuning handoff

Review the local planning follow-up adversarially, reproduce its main claims, and assess the proposed PPO tuning study below. The goal is to maximize combat win probability in the existing M5/M6 simulator while establishing which improvements are computational and which improve policy quality. The faster transition model has encouraging correctness and timing evidence; the learned planner pilot still loses to the heuristics. No PPO tuning from this plan has been implemented or run.

> **Review comment R0 (2026-10-11).** The review asked for here has been done. Inline comments marked **Review comment** carry its findings. The full list, with the open tasks for the next agent, is in [Review comments for the next agent](#review-comments-for-the-next-agent) at the end. No code was changed and no training was run during the review. Review outputs and scripts are in `runs/review_followup/` (gitignored).

## Reviewer prompt

You are reviewing BaldurRL work produced in an earlier session. Treat the reported results as claims to verify, and look for counterexamples, leakage, semantic drift, incorrect targets, and misleading performance conclusions. Review first; report findings before changing code or launching a large training study. Read `AGENTS.md` and its required specifications before any combat edits. Preserve existing uncommitted work and artifacts.

### Locations and PR context

| Item | Location and status at handoff |
| --- | --- |
| Open PR | [PR #9](https://github.com/vrk210/BaldurRL/pull/9), `integration` into `main`, last inspected head `cd7dca17c9258b8144dddd0827111374e4406573` |
| PR worktree | `/Users/vrk93/PycharmProjects/BaldurRL-worktrees/integration` |
| Local follow-up worktree | `/Users/vrk93/PycharmProjects/BaldurRL-worktrees/m5-m6-planning` |
| Follow-up branch | `m5-m6-planning`, based on `e146d0de18be2bef5fae127e398e7f7892e76889`; implementation changes are uncommitted and unpushed |
| Original decision-stage worktree | `/Users/vrk93/PycharmProjects/BaldurRL-worktrees/decision-stages`, commit `6316abe` |
| Python environment used | `/Users/vrk93/PycharmProjects/BaldurRL/.venv/bin/python` |
| Follow-up artifacts | `runs/planning_followup/` inside the follow-up worktree; gitignored and local only |

PR #9 already includes the original M5/M6 stages, adversarial report review, corrected full-trace PPO diagnostics, replay-based exact tactical transition model, expectimax, and original planning pilots. The table model, guided planner, policy priors, expert-iteration pipeline, and follow-up report results are local additions and are absent from the inspected PR head. A GitHub-only review will miss them.

The PR has advanced since the follow-up branched. In particular, `cd7dca1` fixes expectimax depth consumption per successor when an M6 opportunity attack kills the acting ally, and forwards planner diagnostics through `EpsilonPolicy`. The latter overlaps a local follow-up fix. Inspect and reconcile these changes before integrating; avoid replacing current PR files with older whole-file copies. The report and `PROGRESS.md` describe an unfinished integration merge at branch creation time; that is historical, not its current status.

Start by verifying current state, because another agent may have advanced the branches:

```bash
cd /Users/vrk93/PycharmProjects/BaldurRL-worktrees/m5-m6-planning
git status --short
git diff --stat
git log --oneline e146d0d..integration
git diff e146d0d..integration -- agents/expectimax_agent.py combat/tactical_transitions.py value_training.py
```

`git diff` excludes untracked files. Inspect those explicitly; most new modules and tests are currently untracked. Record the source state you actually reviewed.

### Work breakdown

| Area | Files relative to follow-up worktree | Review focus |
| --- | --- | --- |
| Original review and PPO diagnosis, already in PR | `reports/m5_m6_adversarial_review.txt`, `reports/m5_m6_decision_stages.md`, `evaluation/ppo_diagnostics.py`, `reports/review_*.py`, `reports/audit_m5_m6.py` | Full-trace stratified reservoir sampling, occurrence weights, conditional continuation versus whole-fight effects, confidence intervals |
| Faster exact transitions | `combat/table_tactical_transitions.py`, `combat/tactical_transitions.py`, `combat/tactics.py` | Equality to the replay oracle; shared rules; HP caps; probability mass; lifecycle and scheduling; default registration |
| Bounded guided search | `agents/guided_planner.py`, `evaluation/planning_eval.py` | Decision horizon, exact enemy phase, root/interior top-k plus heuristic anchor, equal budget per root candidate, batched leaves, deviation margin |
| Priors and expert iteration | `agents/policy_prior.py`, `expert_iteration.py`, `value_training.py` | Masked training/inference, expert versus behavior labels, selected-action Q targets, successor TD targets, validation split, reproducibility |
| Evidence and documentation | `evaluation/tactical_transition_benchmark.py`, `reports/m5_m6_planning.md`, `docs/EXPECTIMAX_PLANNING.md`, `docs/ARCHITECTURE.md`, `PROGRESS.md` | Reproducible claims, fair timing, stated limitations, stale documentation |
| Tests | `tests/test_table_tactical_transitions.py`, `tests/test_guided_planning.py`, modified `tests/test_tactical_transitions.py` | Independent checks and missing adversarial cases |

The corrected PPO diagnostic samples are drawn across complete recorded encounters, stratified by model and disagreement kind; they are not limited to the first encounters or early turns. These are samples of disagreement occurrences, not a new PPO training distribution. Paired continuations use heuristic follow-on play, so local effects cannot simply be multiplied by disagreement frequency to explain the overall PPO gap. Check the report's whole-fight interventions separately.

### Claims to reproduce

The transition benchmark sampled every legal action at every fifth decision in random-play encounters on seeds 210000–210009. It checked 345 M5 and 346 M6 state/action cases. Successor sets matched the replay oracle, with maximum probability differences about 1e-14. Reported aggregate transition speedups were 16.1× for M5 and 52.5× for M6. These are timings for one process/pass with initialization and caching, not demonstrated full-game speedups or a guarantee for every state/action.

The small learning pilot used, per stage:

- Bootstrap: 600 explored heuristic fights, seeds 180000–180599, epsilon .15; 100 validation fights, 120000–120099.
- One expert iteration: 60 training fights, 180600–180659, epsilon .10; 20 validation fights, 120100–120119; successor TD(lambda=.5) value targets.
- Selection: 100 paired encounters, seeds 122000–122099. M5 tuned heuristic won 81, bootstrap planner 62, iterated planner 66. M6 adaptive heuristic won 83, bootstrap planner 68, iterated planner 64.
- Iterated planner minus anchor: M5 −15 percentage points, approximate paired 95% CI [−25.92, −4.08]; M6 −19 points, [−29.68, −8.32]. Mean decision times were .74 ms and 2.43 ms respectively.

The standalone follow-up suite previously returned 385 passed, 1 skipped. PR `cd7dca1` separately reports 530 passed. Neither count validates the combination of current PR plus local follow-up. No follow-up candidate used frozen planning-test seeds 130000–132999, and no PPO checkpoint changed.

The original pooled PPO gaps remain historical reference points: M5 74.38% versus tuned 78.52%, a 4.14-point gap; M6 75.21% versus adaptive 80.60%, a 5.39-point gap. Those rates come from a different, much larger evaluation than the 100-encounter planner pilot.

### Adversarial checks

1. Compare table and replay distributions on fresh verification seeds and targeted reachable states, including rare late-fight states. Exercise all legal actions, full enemy phases, lethal Advance/opportunity attacks, spent Reactions, Disengaged, Prone/Dodge clearing, healer charges, healing/damage caps, dead-actor skipping, round wrap, and round-limit truncation. Check state encoding and observation restoration. Matching a shared helper alone is weaker evidence than matching the actual environment.
2. Check that `combat/tactics.py` extraction preserves environment behavior. Verify no observations, masks, indices, reward semantics, or combat rules changed inadvertently. Ensure current PR's turn-boundary fix survives integration.
3. Inspect timing order and caches. Compare cold/warm timings, action-specific costs, successor counts, and mean/tail planner latency. A node budget limits decision-state expansion, not chance branching or wall-clock work. Tens of legal actions can still make exhaustive search impractical.
4. Check guided search really includes the exact automatic phase for END_TURN, counts decisions consistently, retains the anchor in root and interior proposals, and treats budget cutoffs consistently. Equal root budgets do not make approximate Q-values equally accurate. A .03 deviation margin is no guarantee of improved play under value error.
5. Confirm collection calls the teacher before epsilon behavior, separates behavior and expert actions, stores Q(executed expert choice) rather than an unconstrained maximum, and clears stale diagnostics. Validate complete episodes and successor indexing for TD(lambda), exact terminal win/loss/timeout targets, and the distinction between `td:<lambda>` and older `lambda:<lambda>` targets.
6. Verify legal masks in fitting/inference, loss weighting, validation epoch selection, disjoint collection seeds, frozen-range rejection, and per-episode RNG independence. Check serial/chunked collection agreement. Search values and behavior successors refer to different actions on exploratory steps; determine whether each target intentionally represents a teacher value or behavior return.
7. Recalculate artifact results and intervals, and inspect `runs/planning_followup/provenance.json` hashes. Separate infrastructure success from policy failure. Do not claim that the planner has closed PPO's gap or that existing speedups establish BG3 feasibility.

Useful artifacts are `transitions.json`, `comparison.json`, `reproducibility.json`, `provenance.json`, and each stage's datasets, `v0/` and `v1/` models/fit histories, and `selection_*.json`. Source commands for the complete pilot are in `docs/EXPECTIMAX_PLANNING.md` under the table model and guided expert iteration section.

Run reproduction into a new output directory so historical artifacts remain intact:

```bash
/Users/vrk93/PycharmProjects/BaldurRL/.venv/bin/python -m pytest -q
/Users/vrk93/PycharmProjects/BaldurRL/.venv/bin/python -m evaluation.tactical_transition_benchmark --seed-start 210000 --episodes 10 --out runs/review_followup/transitions.json
/Users/vrk93/PycharmProjects/BaldurRL/.venv/bin/python -m evaluation.planning_eval --stage m5 --name review_guided_v1 --policy guided --leaf mlp:runs/planning_followup/m5/v1/value.npz --prior runs/planning_followup/m5/v1/prior.npz --root-top-k 3 --horizon 1 --deviation-margin .03 --seed-start 122000 --episodes 100 --out runs/review_followup/m5_selection_v1.json
```

Repeat selection for M6 with M6 model paths. Then broaden transition checks on unused verification seeds; do not use frozen test encounters to debug or choose configurations.

### Requested review output

Return findings ranked by severity with file/line references, evidence, and minimal reproductions. Distinguish reproduced facts, plausible concerns, and demonstrated bugs. Include what you ran, which claims held, remaining coverage gaps, integration conflicts with PR #9, and whether the PPO plan needs changes. Recommend which pieces are ready for integration and which remain experimental. Do not equate a passing suite with exhaustive correctness or a failed small pilot with proof that the method cannot work.

## Proposed PPO environment and training tuning plan

This section is a proposed next study, not completed work. Optimize win probability on unchanged M5/M6 encounter distributions. Keep combat mechanics, action space, observations, opponent behavior, and the base evaluation environment fixed. Put experimental rewards and timeout handling in explicit training wrappers/configurations. Longer-game superiority, policy/search hybrids, and richer combat remain hypotheses requiring separate experiments.

### Phase 0 establish controls and objective semantics

Reproduce current PPO controls before comparing changes. Existing `train.py` exposes gamma, learning rate, entropy, rollout length, batch size, and epochs; it does not yet expose all knobs below. Record resolved settings rather than relying on library defaults. The prior setup used approximately 1,003,520 actual steps, four environments, 64×64 policy/value networks, gamma .99, GAE lambda .95, learning rate 3e-4, entropy coefficient .01, n_steps 1024 per environment, batch size 256, and ten epochs.

> **Review comment PPO-7: log every resolved setting.** `train.py` doesn't expose GAE lambda, clip range, target KL, network size or value-loss coefficient, and advantage normalization is left at the library default. Add them as explicit config fields, and have Phase 0 write the resolved values of every setting (including library defaults) into run metadata. Otherwise the plan's metrics (KL, clip fraction, explained variance) have no defined source.

Audit the installed SB3/sb3-contrib version and timeout path. The inspected MaskablePPO bootstraps `TimeLimit.truncated` transitions by adding gamma times terminal-state value to reward. The current tactical environment gives +1 on victory, −1 on defeat, and zero on ongoing steps or round-limit truncation; evaluation counts a truncation as a non-win. Thus simply setting gamma=1 does not establish a strict win-probability objective if timeout transitions still bootstrap.

Add an explicit experimental training wrapper that makes the round limit an absorbing non-win: emit a true terminal failure and prevent time-limit bootstrapping, preserving the reason in info. Keep base evaluation termination/truncation reporting intact. Verify this through the actual vectorized rollout path, not just a direct environment step. Test victory, defeat, timeout, final observation, reward, and rollout target behavior. Record timeout rates; previous evaluations reported none, so this fix may have little empirical effect even though it clarifies the objective.

> **Review comment PPO-1: treat the timeout wrapper as a correctness fix, not a sweep factor.** Verified: MaskablePPO in sb3-contrib 2.9.0 bootstraps on `TimeLimit.truncated`, and both DummyVecEnv and SubprocVecEnv set it from the env's `truncated`. But no historical 5,000-fight evaluation recorded a truncation, and fights average about 7–8 rounds against a 50-round cap. Build and test the wrapper as planned, apply it to every new run, and don't spend sweep arms on it. The three legacy controls per stage are enough to show it changes nothing.

Compare terminal reward encodings:

| Objective | Victory | Any non-win including timeout | Intermediate reward |
| --- | --- | --- | --- |
| Signed outcome | +1 | −1 | 0 |
| Win indicator | +1 | 0 | 0 |

At gamma=1 and complete absorbing episodes, expected signed return is 2 times win probability minus 1, and expected indicator return is win probability. They have the same optimal win-rate policy but can train differently because of critic/advantage scaling and approximation. At gamma below 1, terminal returns are discounted per agent decision, introducing timing preferences. For example, .99 raised to 30 is about .74. GAE lambda changes the estimator's bias/variance; gamma changes the objective. Keep those roles distinct.

> **Review comment PPO-2: the two encodings give the same actor updates at gamma 1.** SB3 normalizes advantages per minibatch by default. At gamma = 1, ±1 and 0/1 rewards then produce identical actor gradients. They differ only in the critic's loss scale (factor 4, the same as changing the value-loss coefficient) and initial bias. They only mean different things below gamma = 1, where ±1 also rewards delaying a loss, which may encourage Dodge-style stalling. See the Phase 1 comment for a smaller design.

### Phase 1 discount and terminal objective sweep

Run a controlled factorial study with gamma in {.99, .995, .999, 1.0}, both reward encodings, three independent training seeds, and both stages: 48 runs at about one million steps each. Add legacy controls using the original reward/timeout semantics, retaining the same training seeds and compute budget. Three controls per stage bring this to 54 runs. Compare every candidate by base-environment win rate, not shaped or training return.

> **Review comment PPO-2/PPO-3: shrink the factorial and check power first.** - **Design:** gamma ∈ {0.99, 0.995, 1.0} with ±1 reward, plus 0/1 reward only at 0.99, plus one value-loss-coefficient arm. That saves about a third of the runs. Gamma 0.999 is hard to tell apart from 1.0 at ~30 decisions per fight.
> - **Power:** existing PPO seeds span about 1.3 points (73.6–74.9% on M5, 74.5–75.7% on M6), and a 5,000-fight evaluation has SE ≈ 0.6. With 3 seeds per cell, differences under about 1.5 points can't be resolved, while the whole gap to the heuristics is 4–5 points.
> - **Action:** evaluate every final model on at least 5,000 paired fights, and promote with successive halving rather than reading the full factorial once. Do a quick power calculation from the existing seed spread before launching.

Keep architecture, optimizer, GAE lambda, rollout/batch settings, and entropy fixed for this first comparison. Report learning curves and training-seed spread; do not select a winner from a single lucky run. The earlier rough seven minutes per million-step run suggests several aggregate hours for this sweep, but measure current throughput and evaluation overhead before budgeting. That estimate is not a benchmark for the proposed configurations.

### Phase 2 tune optimization and exploration

Promote promising objective/gamma settings, retaining a baseline and a slower-learning exploratory candidate. Use staged, controlled sweeps rather than the full Cartesian product:

- GAE lambda: .95, .98, 1.0.
- Learning rate: 1e-4, 3e-4, 1e-3; compare constant and decay schedules on finalists.
- Entropy coefficient: 0, .003, .01, .03; compare annealing after broad exploration. Log masked action entropy and behavior changes.
- Rollout length, batch size, update epochs, clip range, and target KL: choose a bounded grid after inspecting current KL, clip fraction, value loss, explained variance, and update stability. Account for n_steps being per environment and total batch size being n_steps times n_envs.

Reject configurations for demonstrated instability or consistently poor results; give retained candidates enough steps to distinguish slow learning from failure. Record environment steps and wall-clock cost separately. Ensure batch sizes and rollout lengths are valid for the installed implementation.

### Phase 3 imitation and stronger representations

Use proven M5 tuned/M6 adaptive heuristics as teachers. Compare random initialization, behavior-cloning initialization followed by PPO, and PPO with a decaying masked imitation auxiliary loss. Collect heuristic demonstrations across diverse full encounters, then consider DAgger labels on learner-visited states to address distribution shift. Include a clone-only baseline and paired ablations. Do not use the current weaker planner as a default teacher.

> **Review comment PPO-6: protect cloned skill, and add a stronger cheap teacher.** - **Don't use the planner:** the review confirms the current guided planner shouldn't be a teacher. It loses 15–19 points to the anchor (finding F3 below).
> - **Protect cloned skill:** behavior cloning followed by PPO is known to lose cloned skill early in fine-tuning. Pretrain the critic on teacher rollouts before unfreezing the actor, or use a decaying KL-to-teacher term rather than only a cross-entropy auxiliary.
> - **Better teacher:** a rollout-improved heuristic. For each legal action, play a few hundred full games to the end with the heuristic and pick the best. This is a policy-improvement step, so up to sampling noise it can't be worse than the heuristic. The earlier version lost with only 32 rollouts per action. At ~0.02 ms per heuristic decision it's affordable.
> - **Headroom probe:** run this before Phase 1. If it doesn't clear ~80%, the PPO budget should be reconsidered.

PPO's actor updates must use its own appropriately collected on-policy rollouts. Keep demonstration/DAgger losses explicit and separate. If search chooses executed actions, those trajectories cannot simply be treated as ordinary PPO actor rollouts without accounting for the behavior policy.

First compare simple 64×64, 128×128, and 256×256 networks. If flat features/actions appear limiting, test entity encoders and shared action/target scoring while preserving the simulator's legal decisions. Additional memory is a lower priority until observation aliasing or hidden state is demonstrated. Inspect whether representation gains persist across training seeds and justify inference cost.

### Phase 4 longer training and optional search hybrid

Extend configurations showing progress to 3–5 million steps; extend the strongest stable candidates to 10–20 million only if curves justify it. Separate insufficient training from an optimization plateau. Larger budgets are proposed experiments, not evidence that PPO will overtake the heuristics.

Once policy/value quality improves, consider policy-guided bounded search with the existing simulator: policy supplies candidates, critic supplies leaf estimates, and search can supply separate distillation labels. A learned MuZero dynamics model is not necessary while exact simulator access is cheap. Evaluate policy alone, search alone, and the hybrid under matched decision-time budgets. For richer future action spaces, candidate generation and selective search remain necessary; current transition speedups do not remove exponential branching.

### Evaluation and promotion protocol

Audit all existing seed usage before allocating fresh development, checkpoint-selection, configuration-selection, and final-test ranges. The planning follow-up already used 120000–120119 for validation and 122000–122099 for selection, while historical PPO studies used other evaluation ranges. Do not relabel previously viewed encounters as untouched tests. Treat environment-reset seeds and independent learner initialization/training seeds as separate concepts; inspect the actual reset sequence for overlap.

> **Review comment PPO-5: seed ranges already used.** New ranges must avoid all of these:
>
> | Range | Earlier use |
> |---|---|
> | 1000–1199 | PPO training-time evaluation (checkpoint saves) |
> | 3000–7999 | PPO and heuristic held-out evaluation |
> | 9000–13999 | M2 exact-optimum / M4 test seeds |
> | 100000–123999 | Value-network data, validation and selection |
> | 140000–199999 | Planning data and expert iteration |
> | 210000–212399 | Review: transition benchmark, adversarial check, env fingerprints |
>
> 130000–132999 stays frozen. Training envs start at `reset(seed=s+rank)` and then continue their own stream. They overlap an evaluation encounter only on each worker's first fight, which is negligible, but record it.

Keep frozen planning range 130000–132999 untouched throughout tuning. Once configurations, checkpoint rules, and inference settings are fixed, evaluate finalists with at least five independent training seeds and preferably 10,000 encounters per model on a predeclared fresh final set. Use the same encounters for PPO and heuristic comparisons. If the existing frozen range is also used, identify it as a separate 3,000-encounter test, never as the 10,000-encounter set.

Use separate encounter sets for checkpoint selection and configuration comparison to reduce selection bias. Report per-model paired win-rate differences and confidence intervals plus variability across learner seeds; do not treat repeated encounters across multiple models as wholly independent evidence. State whether policy inference is deterministic or stochastic and choose that setting before final testing.

> **Review comment PPO-4: drop best-of-100 checkpoint selection during training.** `train.py` evaluates 200 fights every 10k steps (about 100 evaluations per run) and keeps the best. That maximum is biased upward by roughly 2–3 SE, and the history shows it: seed-0 checkpoints at 250k–1M steps ranged 72.8–76.2% at SE 1.0. These evaluations also cost about as many policy steps as training. Use the final model, or save checkpoints and score them afterwards on a separate selection range.

Log win rate, timeout rate, decisions/rounds, remaining HP, role targeting/kill order, ability/positioning usage, full-trace disagreement patterns, masked entropy, KL, clip fraction, value loss/explained variance, training throughput, and inference latency. Preserve resolved configs, commit/source hashes, package versions, model/checkpoint paths, seed manifests, and evaluation artifacts.

A successful result improves paired win rate against both reproduced PPO controls and the best heuristic, with consistent evidence across learner seeds and acceptable inference cost. A negative result should identify whether discount/objective alignment, optimization, exploration, imitation, representation, or sample budget was tested adequately; it should not claim PPO is intrinsically inferior from one setup.

### Implementation deliverables before large runs

1. Review and reconcile the local planning implementation with current PR #9 without losing either branch's fixes.
2. Add reproducible PPO experiment configs and the missing training knobs, explicit reward/timeout wrappers, complete metadata, and resumable sweep orchestration.
3. Validate terminal targets, vectorized timeout handling, masks, seed allocation, and legacy-control reproducibility with focused tests and smoke runs.
4. Run Phase 1, report results, and use evidence to select the next bounded sweep. Keep speculative imitation/architecture/hybrid changes separate so their effects remain attributable.

> **Review comment deliverable 1 status.** A trial apply of this worktree onto PR #9 head `cd7dca1` gave two trivial conflicts, and the combined tree passes 542 tests.
> - **`value_training.py`:** the epsilon fix is identical on both sides, so keep the PR side.
> - **`docs/ARCHITECTURE.md`:** keep the PR paragraph and append this branch's new section.
>
> PR #9's `turn_passed` fix survives: the table model inherits it, and the M6 lethal-Advance test passes with the table model as default. See the readiness table below for what to integrate.

This handoff itself does not authorize publishing, merging, or starting the full sweep. Its immediate task is implementation review and plan assessment; subsequent implementation/run scope should follow the user's next instruction.

## Review comments for the next agent

Reviewer: Claude Code, 2026-10-11, against base `e146d0d` with this worktree's uncommitted and untracked files, compared with PR #9 head `cd7dca1`. The pilot's provenance hashes match the current files.

### Claims that held

| Claim | Reproduced |
|---|---|
| Suite 385 passed / 1 skipped | 386 passed, none skipped |
| Benchmark 345 M5 / 346 M6 cases, error ~1e-14 | Same cases; max error 9.7e-15 / 1.0e-14 |
| Speedup 16.1× M5 / 52.5× M6 | 16.1× / 53.4×. Per-transition with cold caches, not whole-game |
| Selection M5 anchor / v0 / v1 | 81 / 62 / 66, identical seed by seed |
| Selection M6 anchor / v0 / v1 | 83 / 68 / 64, identical seed by seed |
| Paired CIs | M5 v1 −15 [−25.9, −4.1], M6 v1 −19 [−29.7, −8.3]; exact McNemar p ≤ 0.014 |
| Frozen 130000–132999 unused | Confirmed |

Extra evidence:
- **Adversarial transition check:** 118,089 table-model cases on seeds 211000–211029 had 0 mismatches against the replay oracle. They covered lethal Advance, spent reactions, Prone+Dodge, heal caps and charges, dead actors, and round-50 truncation (`runs/review_followup/adversarial_transitions.{py,json}`).
- **`tactics.py` extraction:** behavior is unchanged. 800 seeded episodes give identical fingerprints at `e146d0d`, in this worktree, and at `cd7dca1`.

### Findings to address, most severe first

- [ ] **F1 (bug, demonstrated): the expert-data teacher isn't the evaluated planner.**
  - **Cause:** `expert_iteration.py:139` defaults collection to `--root-top-k 0` (score every action). `docs/EXPECTIMAX_PLANNING.md` (~line 272) collects with that default but evaluates with top-3 plus anchor (~line 283).
  - **Evidence:** v0 on expert seeds 180600–180699, no exploration, wins per 100:

    | Stage | Full width | Top 3 + anchor | Anchor heuristic |
    |---|---:|---:|---:|
    | M5 | 37 | 70 | 87 |
    | M6 | 52 | 65 | 81 |

    This explains the expert data's 19/60 and 26/60 win rates.
  - **Fix:** make collect and evaluate share one planner config, and record it in the run metadata. Add a test that fails if they diverge.
- [ ] **F2 (pipeline design): v1 is fit from scratch on 60 fights.**
  - **Data:** `expert_train.npz` is 1,890 rows on M5 and 1,640 on M6, with 20 validation fights. The 600-fight bootstrap data is discarded and nothing is carried from v0. v0's own best epoch was 2 of 25.
  - **Consequence:** 62→66 / 68→64 is noise.
  - **Fix:** pool bootstrap and expert data (or warm-start from v0), use far more expert fights, and use fresh seeds outside the ranges in PPO-5.
- [ ] **F3 (demonstrated): the deviation margin doesn't stop value exploitation.**
  - **Control:** with margin 1.0 the planner reproduces the anchor exactly (81 / 83, same decision counts), so the whole loss comes from deviations.
  - **Rate (v1):** 426/3,164 decisions on M5 (83 fights) and 355/2,779 on M6 (70 fights), with a mean estimated gain of only ~0.05.
  - **Main swaps:** M5 Attack→Dodge 169, Attack→End Turn with action unused 74, Action Surge→End Turn 35. M6 Advance→End Turn 104, Advance→Trip 91.
  - **Pattern:** exploitation grows with the number of actions compared, since the max over noisy Q picks errors.
  - **Next:** consider calibrated margins (e.g. from value-net validation error per action type), or restrict deviations to actions with support in the training data.
  - **Data:** `runs/review_followup/deviations.{py,json}`.
- [ ] **F4 (plausible, not triggered at horizon 1): the node budget is spent depth-first in action order.**
  - At `agents/guided_planner.py:100–110`, once the budget runs out, later siblings are scored at shallower depth, and the max then mixes depths.
  - Fix it (breadth-first or iterative deepening, or a per-child budget) before using horizon > 1, and add a test.
- [ ] **F5 (minor, documented): TD targets on exploratory steps follow the behavior policy.** With 5–8% exploration this mixes two policies' values. State it wherever results are reported.

### Readiness

| Ready to integrate into PR #9 / `integration` | Experimental |
|---|---|
| `combat/table_tactical_transitions.py` as the M5/M6 default | `agents/guided_planner.py` |
| `combat/tactics.py` helpers | `agents/policy_prior.py` |
| Epsilon-diagnostics reconciliation | `expert_iteration.py`, until F1, F2 and F4 are fixed |
| TD-target validation in `value_training.py` | The pilot result. Report it as "this configuration loses", not "expert iteration fails" |
| `evaluation/tactical_transition_benchmark.py`, optionally the adversarial checker as a slow test | |

### Suggested order of work

1. Fix F1 and F2. Rerun one expert iteration with a matched teacher, pooled data and fresh seeds (declare them in `PROGRESS.md` first).
2. Run the rollout-improved-heuristic headroom probe (PPO-6) before committing the PPO budget.
3. PPO Phase 0: config knobs and resolved-settings logging (PPO-7), timeout wrapper with tests (PPO-1), and no best-checkpoint selection during training (PPO-4).
4. Phase 1 with the reduced design and power check (PPO-2, PPO-3).

### Coverage gaps in this review

- **Not run:** bootstrap/expert collection, refits, PPO controls, any training.
- **Not exhaustive:** the transition checks are extensive, and some mutated states may not be reachable in real play. The oracle drives the real env, so it can't catch a rule that is itself wrong in `tactics.py`.
- **Not exercised:** deeper guided search (horizon > 1) and its budget behavior.

Nothing here authorizes merging, publishing, or starting the full sweep. Preserve this worktree's uncommitted work.
