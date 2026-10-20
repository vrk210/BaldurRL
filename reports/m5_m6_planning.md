# M5/M6 exact transitions and planner pilots

The original exact M5/M6 transition model works and is verified. The follow-up
implements a faster table model, bounded policy-guided search, and explored
expert iteration. Neither the original planners nor the follow-up pilot beat
the best hand heuristic on configuration-selection seeds. The frozen test range
(130000–132999) remains unused.

## Transition model

`combat/tactical_transitions.py` enumerates exact successor distributions for
every M5/M6 decision by driving the real environment code with an enumerating
random source. It doesn't restate any rule. Details are in
[EXPECTIMAX_PLANNING.md](../docs/EXPECTIMAX_PLANNING.md#m5m6-transition-model).

| Check | Cases | Result |
|---|---:|---|
| Raw path enumeration through unmodified `env.step` | 241 | Matches to about 1e-16 |
| 20,000 real-generator samples per case | 288 | No unseen successor; largest z 4.6 across thousands of outcomes |
| Normal-mode attack distributions vs M0–M4 derivation | 6 | Equal to 1e-15 |

Cost per transition ranges from under a millisecond to several seconds. A
three-target Cleave has up to about 700 merged outcomes, and a three-enemy
`END_TURN` up to about 3,400.

## Planner pilots (selection seeds only)

All runs used seeds from 122000 upward. Value networks were trained on seeds
100000–121999 and 140000–199999, as declared in `PROGRESS.md` before use.

| Stage | Planner | Seeds | Planner wins | Best heuristic wins | Mean time per decision |
|---|---|---:|---:|---:|---:|
| M5 | One-decision lookahead, first networks | 1,000 | 45.3% | 79.4% | 29 ms |
| M5 | One-decision lookahead, after one self-play round | 400 | 52.0% | 81.8% | 29 ms |
| M5 | Exact rest-of-turn rollout of the heuristic, larger afterstate network | 100 | 54% | 81% | 0.6–1.3 s |
| M5 | Same, deviating from the heuristic only by more than 3 points | 100 | 80% | 81% | 0.6 s |
| M6 | Exact rest-of-turn rollout, larger afterstate network | 20 | 75% | 70% | 4.3 s, max 124 s |

PPO seed 0 won 75.6% on the same 1,000 M5 seeds.

## What went wrong

- **Full-turn search is too expensive.** Searching a whole turn exactly, as on
  M4, did not finish one decision in 300 seconds. Cleave and Advance branch into
  hundreds of outcomes, and each one runs real environment code.
- **Searching through the enemy phase is too expensive.** Doing it at every
  turn end, as the M4 planner does, would mean hundreds of enemy-phase
  enumerations per decision.
- **The learned values get exploited.** The cheaper designs score turn ends with
  outcome-trained value networks. The search then picks actions whose successor
  states the networks misjudge:
  - It chooses Dodge, which the heuristics never use, in place of attacks.
  - It ends turns with an Action or Action Surge unused.

  The gaps behind these choices average about 2 points, within the networks'
  error. Ten times more training data improved validation loss but not play. A
  3-point deviation margin removed the damage, but only by reverting to the
  heuristic.

On M4 the planner gained because exact enemy-phase search added real
information before each value estimate. On M5/M6 that step is unaffordable with
this model, and the remaining value networks are not accurate enough to improve
on a heuristic that already wins about 80%.

## Next steps

1. **A faster transition model.** A table-driven M5/M6 model like the M0–M4 one
   would restore exact enemy-phase search. This model would then serve as its test oracle.
2. **Better value targets.** Train on search-backed or temporal-difference
   targets with explicit exploration, and keep a conservative deviation rule
   until values are accurate.
3. **A policy prior and expert iteration.** These are planned in the Codex
   planning-scale task, and they address exactly this exploitation problem.

Scripts and raw outputs for these pilots are in `runs/m5m6_planner_prototypes/`
and `runs/eval_m5/sel/`. Those folders are git-ignored.

## Implemented follow-up (2026-10-10)

Implemented on branch `m5-m6-planning`, based on integration commit `e146d0d`,
in a separate worktree because the integration worktree has an unfinished merge.
The original pilot findings above remain historical.

### Faster exact transitions

`combat/table_tactical_transitions.py` is now the default M5/M6 transition model.
The original `TacticalTransitionModel` remains an independent replay oracle.
The table model convolves attack/heal outcomes and merges HP caps before expanding
states. It uses shared costs, targeting, heal eligibility, roll modes, reach,
opportunity-attack eligibility, lifecycle functions and turn order; no combat
mechanics, observations, action indices, or rewards changed.

A seeded random-play comparison on seeds 210000–210009 checked every legal
action at every fifth decision (including complete enemy phases):

| Stage | State/action cases | Largest probability difference | Oracle total | Table total | Speedup |
|---|---:|---:|---:|---:|---:|
| M5 | 345 | 9.7e-15 | 2.764 s | 0.172 s | 16.1× |
| M6 | 346 | 1.0e-14 | 12.165 s | 0.232 s | 52.5× |

All successor sets matched, and all probability sums were within 1e-12 of one.
These are one-process pass timings, including distribution initialization and
normal per-encounter caching; speedup varies by action and state. Additional
unit tests cover death during Advance (including terminal defeat), Disengaged
and spent Reactions, Healer charges/HP caps, retaliation against dead allies,
Prone/Dodge lifecycle, dead-actor skipping, round-50 truncation, terminal
observations, and Cleave's fixed pre-roll reach set.

### Search and learning

- `agents/guided_planner.py` adds bounded decision search. END_TURN includes the
  exact enemy phase. Policy top-k proposals retain the tuned/adaptive heuristic
  anchor; each root candidate gets an equal decision-node budget. Chance outcomes
  are never capped or sampled. Horizon/budget cutoffs use batched leaf values.
- `agents/policy_prior.py` adds heuristic and learned masked priors.
- `expert_iteration.py` collects a teacher label/Q-values before random exploration,
  separating behavior and expert actions. It fits a masked policy prior and a
  value network on outcome, conservative search-value, or successor TD(lambda)
  targets. Subsequent collect/fit rounds load the previous value/prior checkpoints.
- The deviation rule retains the heuristic unless the proposal's estimated gain
  exceeds a configurable margin (3 points in this pilot). This rule does **not**
  guarantee improved play when value errors are large.
- The existing epsilon wrapper now forwards only current search diagnostics and
  clears them on random actions, preventing missing/stale search labels.

Training and validation must use disjoint seeds; the expert CLI rejects the
frozen test range for collection and fitting. Command examples and target
semantics are in [EXPECTIMAX_PLANNING.md](../docs/EXPECTIMAX_PLANNING.md#m5m6-table-model-and-guided-expert-iteration).

### End-to-end pilot

For each stage, a 64×64 value/prior pair was bootstrapped from 600 explored
heuristic fights (seeds 180000–180599, epsilon .15), with validation loss on 100
separate fights (120000–120099). One expert iteration used 60 explored teacher
fights (180600–180659, epsilon .10), 20 validation fights (120100–120119), and
TD(lambda=.5) value targets; the policy distilled the conservative teacher.
This is a small pipeline pilot, not a scaled training study.

The teacher scored all legal root actions. Selection evaluated three highest
learned-prior proposals plus the heuristic anchor, one decision of lookahead,
a 64-node budget per candidate, and a 3-point deviation margin. All policies
used the same 100 configuration-selection seeds 122000–122099 (already used
for selection in the original study). Validation win rates were not used to
select epochs; validation target loss was.

| Stage | Anchor | Bootstrap planner v0 | After one iteration v1 | v1 − anchor, paired approximate 95% CI | v1 mean time/decision |
|---|---:|---:|---:|---:|---:|
| M5 | tuned 81/100 | 62/100 | 66/100 | −15 points [−25.92, −4.08] | 0.74 ms |
| M6 | adaptive 83/100 | 68/100 | 64/100 | −19 points [−29.68, −8.32] | 2.43 ms |

The transition-speed problem is substantially reduced. The learned-value problem
remains: this pilot does not improve on the anchor, and one expert iteration does
not reliably improve even the bootstrap planner. The sample sizes and training
coverage are too small to conclude that the approach cannot improve with better
training. Unlimited full-turn search also remains combinatorial; node budgets
bound expanded decision states, not wall-clock time or chance-successor count.
No candidate advanced to frozen-test evaluation, and no PPO checkpoint was trained
or changed. PPO's previously reported overall gaps therefore remain unchanged.

Full-suite verification: 385 passed, 1 skipped. Artifact checks confirm expert
search labels equal Q(expert action), all legal actions are scored in teacher
data, terminal TD targets are exact, and no frozen-test seeds appear. Raw
outputs, networks, datasets, fit histories, and paired comparisons live under
`runs/planning_followup/` (git-ignored); source and reproducible commands are retained in the repository.
