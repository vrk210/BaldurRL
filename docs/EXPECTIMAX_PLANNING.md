# Expectimax planning

`ExpectimaxAgent` (`agents/expectimax_agent.py`) uses the simulator as an exact
model. Instead of sampling rollouts (see [rollout planning](ROLLOUT_PLANNING.md)),
it enumerates every chance outcome of a decision with its exact probability and
maximizes expected win probability:

```text
V(s, d) = 1 / 0                         if s is a victory / defeat or truncation
        = leaf(s)                       if s is ongoing and d == 0
        = max_a sum_s' P(s'|s,a) V(s', d - [a is END_TURN])   otherwise
```

Depth `d` counts controlled-turn boundaries. `END_TURN` includes every
automatic actor that follows it (the whole enemy phase), so depth 1 searches the
rest of the active ally's turn, resolves the following enemy phase exactly when
there is one, and scores the next controlled turn start with the leaf value.
`depth=None` searches to termination and is exact (feasible only for small
states). The agent keeps the `choose_action(observation, action_mask)` policy
interface and contains no combat rules.

```python
from agents.expectimax_agent import ExpectimaxAgent
from agents.value_functions import MLPValue

agent = ExpectimaxAgent("m4", depth=1, leaf_value=MLPValue.load("runs/value/v1_outcome.npz"))
action = agent.choose_action(observation, action_mask)
agent.last_diagnostics  # q_values, value, expanded_nodes, leaf_evaluations, planning_seconds, depth_used
```

## Exact transition model

`combat/transitions.py` defines the planner-facing protocol:

| Method | Meaning |
| --- | --- |
| `state_from_observation(obs)` / `same_encounter(obs)` | compact hashable state of a live allied decision; encounter check |
| `status(state)` | `ONGOING`, `WIN`, `LOSS`, `TRUNCATED` |
| `legal_actions(state)` | indices of legal decisions (same as the environment mask) |
| `outcomes(state, action)` | exact merged `(probability, next_state)` list; `END_TURN` includes the automatic phase |
| `ends_turn(action)` | whether the decision is a controlled-turn boundary |
| `observations(states)` | batch of environment observation vectors (for leaf values) |

`transition_model(stage, observation)` builds a model bound to one encounter.
`RosterTransitionModel` covers M0–M4. Its state is
`(status, active ally, round, ally blocks, enemy HP)`, each ally block being HP
and the five resource counts. Enemy per-turn resources are omitted because the
environment refreshes them at the start of every enemy turn before use.
Identical successors are merged, so HP outcomes collapse (an M4 Cleave on three
enemies has at most 729 successors; an M4 enemy phase rarely more than ~60).

No rule is restated. Dice, hit/critical rules, Cleave halving (minimum 1) and
Second Wind healing come from `combat/distributions.py`
(`attack_damage_distribution`, `second_wind_healing_distribution`), which
enumerates die faces through the same small rule functions in
`combat/mechanics.py` (`attack_hit`, `damage_dice_count`, `damage_from_dice`,
`scale_damage`, `SECOND_WIND_HEALING`) that the sampling mechanics call. Costs come
from the ability catalog, legality from `legal_mask_for`, enemy targeting from
`automatic_attack_decision` in `combat/legality.py` (the environments'
first-legal-attack automatic turns, factored out; exhaustive tests confirm they
agree),
and turn order and refresh from the environment's `TurnManager` order and
`turn_refresh` mappings. Termination, the round-50 truncation check, dead-actor
skipping, and refresh mirror the environment's `END_TURN` loop. Legality and
enemy targeting are memoized on the fields they read; enemy-phase distributions
are memoized on ally HP, enemy liveness, and whether the round is the last.
No capping or bucketing is used: every distribution is exact.

### Adding a stage

A new stage registers its own enumeration:

```python
from combat.transitions import register_transition_model
register_transition_model("m5", lambda stage, observation: M5TransitionModel(...))
```

The model may use any hashable state encoding. Planners only call the protocol
methods above, so roles, conditions, or ranks belong entirely to the new model.
Reusing `RosterTransitionModel` is possible only if the stage keeps the M0–M4
rules (fixed first-legal-attack enemies, the same resources and effects);
otherwise a stage-specific model should enumerate its own outcomes from shared
distribution helpers added next to its mechanics.

### Verification

- `tests/test_transitions.py` drives the real `env.step` with a scripted RNG
  through every RNG path (odometer enumeration) for 14 M0–M4 states and all
  legal actions, including round-50 truncation and dead-ally skipping; the
  enumerated distributions match to 1e-12. Three-enemy M4 Cleave and END_TURN
  steps (too many paths to enumerate) are checked against 20,000 samples.
- `runs/transition_validation/sampling_200k.json` (script alongside): 65
  (state, action) cases, 200,000 sampled environment transitions each, M0–M4.
  No sampled successor was missing from the enumeration; probability sums are
  1 within 4.4e-16; chi-square z-scores have mean 0.19 and maximum 3.26 (that
  case was then shown exact by full path enumeration).
- `tests/test_expectimax_agent.py`: unbounded search reproduces an independent
  M2 value-iteration oracle (condensed from the supervisor's `m2_dp.py`) to
  1e-12; depth-1 search with the exact optimal value as leaf reproduces Q* at
  every decision of an episode. A separate check against the original
  `m2_dp.py` on 12 random small states agreed to 6e-16.

### M5/M6 transition model

`combat/tactical_transitions.py` registers `TacticalTransitionModel` for M5 and
M6. Their rules (roles, Trip/Prone, Dodge, healing, three targeting policies,
ranks, opportunity attacks) are not restated. Instead the model drives the real
environment code with an enumerating random source (`_PathRng`): every sequence
of draws is replayed in odometer order, and identical successor states are
merged. Three things keep this tractable:

- **One draw per attack.** While enumerating, `combat.tactics.resolve_attack_with_mode`
  and `combat.tactics.roll_attack` are swapped for versions that draw once from
  the attack's exact (hit, damage) distribution. Those distributions are computed
  by enumerating every die path through the original `combat.rolls` functions
  with exact fractions, and cached. Callers in `combat.tactics` use only the
  damage dealt and, for Trip, whether it hit. The swap is undone after every call.
- **One actor at a time.** `END_TURN` runs `TacticalCombatEnv._end_current_turn`,
  then `_pass_step` once per actor, merging identical intermediate states.
  `_pass_turn` is built from the same two methods, so the environment and the
  model share one loop.
- **Caching.** Outcomes are cached per (state, action), and enemy steps per state.

States are tuples of (status, current side, current slot, round, per-ally HP,
conditions and resources, per-enemy HP, conditions and resources). Enemy Action
is omitted because every enemy turn refreshes it before use. Every M5/M6 state at
an allied decision is fully observable, so states convert to observations and back.

Cost is the limitation. A three-target Cleave has hundreds of merged outcomes, a
three-enemy `END_TURN` up to a few thousand, and each enumerated path runs real
environment code (tens of microseconds). Single transitions take from under a
millisecond to several seconds, so full-turn search on M5/M6 is impractical
without a faster, table-driven model.

Verification (`tests/test_tactical_transitions.py`, plus a larger scratch run):
- **Raw path enumeration.** 241 (state, action) cases from random-play M5/M6
  episodes, with at most 20,000 raw paths, matched enumeration through the
  unmodified `env.step` to floating-point rounding (about 1e-16).
- **Sampling.** 288 cases, each with 20,000 real-generator samples, produced no
  successor missing from the enumeration. The largest per-outcome z-score was 4.6,
  across many thousands of outcomes.
- **Cross-check.** Normal-mode attack distributions equal the independent M0–M4
  `attack_damage_distribution` derivation.

## Search implementation

Each decision expands the tree depth-first with a transposition table keyed by
`(state, remaining depth)`, collects all depth-0 states, evaluates them in one
batch with the leaf value, and backs values up in post-order. The table is kept
across the decisions of one controlled turn (later decisions of the turn are
usually free) and cleared when the agent ends a turn or the encounter changes.
Ties within 1e-12 pick the lowest action index. With `node_budget=N`, a search
deeper than one turn that would expand more than `N` decision nodes is abandoned
and the next shallower depth is used; `depth_used` reports the outcome.
`always_search=True` also searches forced decisions (used to record root values
for value training).

Two options trade exactness for cost on stages with long enemy phases:

- `afterstate_value`: an `END_TURN` that would use the last unit of depth is not
  expanded through the enemy phase. It is scored by a learned afterstate value
  of the state in which the turn ends (Sutton and Barto, section 6.8). Afterstate
  networks are fitted on `END_TURN` rows only (`value_training.py fit --afterstate`),
  with every ally's per-turn fields zeroed.
- `decision_horizon=1`: search only the next decision. Ongoing successors are
  scored by `leaf_value`, terminal ones exactly, and `END_TURN` by `afterstate_value`.

## Leaf values

`agents/value_functions.py` maps observation batches to win probabilities:

- `RaceValue` (heuristic): a deterministic mean-field race. Allies, with
  expected Second Wind healing folded into HP, spend attacks round by round on
  the living enemy with the lowest attacks-to-kill per unit threat; Action Surge
  adds an attack and Cleave deals expected halved damage to every living enemy;
  enemies deal their expected damage per round to the lowest living ally slot;
  only allies at or after the active slot still act in the current round. The
  margin (surviving ally HP share on a win, minus surviving enemy HP share on a
  loss) goes through `sigmoid(scale * margin + bias)`, fitted by two-parameter
  logistic regression on heuristic and PPO states (`value_training.py fit-race`).
- `MLPValue`: a ReLU MLP run in NumPy. Inputs are the observation divided by the
  observation-space highs, with inactive allies' Action and Bonus Action counts
  zeroed (both are set, not added, at the start of that ally's next turn, so they
  cannot matter and zeroing keeps search leaves on the training distribution).
- `EnsembleValue`: averages several `MLPValue` logits.

`value_training.py collect` records every decision observation with the episode
outcome (and, for the planner, the search root value); `fit` trains the MLP by
binary cross-entropy on outcome, search-value, mixed, or lambda-return targets
with early stopping on a separate validation seed range.

## Evaluation

```bash
.venv/bin/python -m evaluation.planning_eval --name x_v1_d1 --policy expectimax \
    --leaf mlp:runs/value/v1_outcome.npz --depth 1 --seed-start 42000 --episodes 2000 \
    --workers 4 --out runs/m4_eval/val/x_v1_d1.json
```

`evaluation/planning_eval.py` splits seeds into contiguous chunks across worker
processes (results do not depend on the worker count), records per-seed
outcomes and per-decision wall time, and provides paired win-rate differences.

See [the M4 report](../reports/m4_expectimax.md) for results, cost, and failure
modes.
