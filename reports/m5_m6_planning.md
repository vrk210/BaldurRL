# M5/M6 exact transitions and planner pilots

The exact M5/M6 transition model works and is verified, and a faster table-driven
version is now the default. No planner built on either beat the best hand
heuristic on configuration-selection seeds, so none was run on the frozen test
range (130000–132999), which remains unused. The planner work is shelved; see
[Follow-up](#follow-up-2026-10-10-to-2026-10-15) for what was kept.

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

## Follow-up (2026-10-10 to 2026-10-15)

### Kept: the table transition model

Next step 1 above is done. `combat/table_tactical_transitions.py` is now the
registered M5/M6 model, and the original model is kept as its test oracle.

| Check | Cases | Result |
|---|---:|---|
| Benchmark against the oracle, seeds 210000–210009 | 345 M5, 346 M6 | Same successor sets; largest probability difference 1.0e-14 |
| Speed, same benchmark | | 16× on M5, 53× on M6, per transition |
| Adversarial audit with constructed edge states, seeds 211000–211029 | 118,089 | No mismatches |
| Environment behavior after the `tactics.py` helper extraction | 800 seeded episodes | Identical masks, observations, rewards and outcomes |

Details are in [EXPECTIMAX_PLANNING.md](../docs/EXPECTIMAX_PLANNING.md#m5m6-table-model-default).

### Shelved: guided search and expert iteration

Next steps 2 and 3 were piloted: a guided planner (policy-prior proposals plus
the heuristic as an anchor, one decision of lookahead with the exact enemy phase,
a learned leaf value, and a 3-point deviation margin) and one round of expert
iteration with TD(λ) value targets. On the selection seeds 122000–122099:

| Stage | Anchor heuristic | Bootstrap planner | After one iteration | Iteration − anchor (paired 95% CI) |
|---|---:|---:|---:|---:|
| M5 | 81/100 | 62/100 | 66/100 | −15 points [−25.9, −4.1] |
| M6 | 83/100 | 68/100 | 64/100 | −19 points [−29.7, −8.3] |

An independent review reproduced these numbers seed by seed and found that the
pilot did not fairly test expert iteration:

1. **The teacher was not the evaluated planner.** Expert data was collected with
   every legal action scored, while evaluation used the top 3 prior proposals
   plus the anchor. On the expert-data seeds, the collection setting won 37/100
   (M5) and 52/100 (M6), against 70 and 65 for the evaluated setting and 87 and
   81 for the anchor.
2. **The iterated value network saw only 60 fights.** It was fit from scratch on
   1,890 (M5) and 1,640 (M6) decisions, dropping the 600-fight bootstrap data.
   The bootstrap network itself stopped improving on validation at epoch 2.
3. **The deviation margin did not stop value exploitation.** With deviations
   disabled, the planner reproduces the anchor exactly, so the whole loss comes
   from deviations. It deviated at about 13% of decisions, mostly by Dodging
   instead of attacking or ending turns with actions unused on M5, and ending
   turns or Tripping instead of Advancing on M6. These are the same failure
   modes as the earlier pilots.

The supported conclusion is therefore "this configuration loses to the
heuristic", not "expert iteration cannot work here". The work is shelved rather
than fixed because further policy gains on invented M5/M6 rules no longer serve
the project's main goal. The planner, priors, expert-iteration pipeline, TD
targets, their tests, and the full review are preserved on branch
`archive/m5-m6-guided-planning`.
