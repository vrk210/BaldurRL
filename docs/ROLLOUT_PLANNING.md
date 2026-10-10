# Rollout planning

`RolloutAgent` chooses a legal action by evaluating complete simulated episodes
from the current observation. It supports M0 through M6 and uses the existing
terminal-reward environment and combat mechanics. Policy weights stay fixed;
the planner performs no training.

```bash
.venv/bin/python -m evaluation.evaluate --stage m4 --agent rollout --rollout-policy heuristic --rollouts-per-action 32 --agent-seed 0 --seed-start 3000 --episodes 10 --trace-dir runs/m4_rollout
.venv/bin/python -m evaluation.evaluate --stage m4 --agent rollout --rollout-policy ppo --model runs/m4_ppo_s0/final_model.zip --rollouts-per-action 32 --agent-seed 0 --seed-start 3000 --episodes 10
```

The default budget is **32 full simulations per legal candidate action**.
Seven legal actions therefore require 224 simulated episodes for one actual
decision. A simulation first applies its candidate action, then follows the
continuation policy until termination or round-limit truncation. Victory scores
1; defeat and truncation score 0. The candidate's score is its simulated win
fraction. Exact ties prefer the continuation policy's action if it is among
the tied actions, then the lowest tied action index. A single legal action
returns immediately without simulation or RNG consumption.

## Python interface

```python
from agents.heuristic_agent import make_heuristic_agent
from agents.rollout_agent import RolloutAgent

planner = RolloutAgent(
    "m4",
    continuation_factory=lambda seed: make_heuristic_agent("m4"),
    rollouts_per_action=32,
    seed=0,
)
action = planner.choose_action(observation, action_mask)
diagnostics = planner.last_diagnostics
# diagnostics.scores: tuple of (legal action index, estimated win probability)
# diagnostics.simulated_episodes, simulated_decisions, planning_seconds
```

The factory receives a policy seed and must provide independently initialized
policy state for every simulation and for tie resolution. For stochastic
continuation, `lambda seed: RandomAgent(seed)` provides isolated generators.
Factories must not return a shared mutable policy or capture the live policy's
RNG. They may share immutable model weights. The CLI's PPO continuation loads
one feed-forward MaskablePPO model and creates fresh deterministic `SB3Policy`
adapters; it never copies or trains a network per simulation. Arbitrary recurrent
or stateful policy histories are not restored automatically. A custom factory
would need to supply correct isolated history itself.

## State and seed isolation

`combat.simulation.simulation_from_observation(stage, observation, seed=...)`
reconstructs independent mutable characters and resource pools. It restores
all observed HP, enemy statistics, ability resources, round, and active ally.
Restoring a partially spent turn does not refresh it. Stage presets and known
abilities come from existing constructors. Enemy Action resources need not be
observed because the environment refreshes them before the next automatic turn.

M5/M6 restoration also rebuilds roles, targeting policies, Prone, Dodging,
last attackers, heal and Trip charges, and (M6) positions, Disengaged, Movement,
and enemy Reactions; malformed one-hots, out-of-profile statistics, and
inconsistent ranks raise errors. The CLI's `--heuristic-variant` selects an M5/M6
heuristic as the continuation policy.

This API accepts **ongoing allied decision observations only**. Terminal or
truncated status cannot always be recovered from the vector: in particular,
a round-50 observation may represent a live decision or a truncated episode.
Callers must know that the supplied observation is live. Malformed vectors,
defeated sides, and dead active allies raise errors. Simulator rewards always
use the terminal scheme, including when the actual M2 environment uses damage
shaping.

The planner accepts only observations and masks. It never reads the actual
environment's RNG, episode seed, future rolls, or unobserved state. Its own
generator supplies fresh simulation and continuation seeds. For each rollout
number, candidate actions use the same pair of seeds (common random numbers).
Those streams can consume different numbers of draws after actions diverge.
Simulation initialization discards constructor sampling and starts future
combat rolls from the supplied simulation seed. Supplied observations/masks
and actual episode state remain unchanged by planning.

Evaluation remains `evaluate(agent, seeds, stage=...)`. Rollout CLI run cards add
`rollout_settings` with continuation policy, budget, planner seed, objective,
common-seed convention, and tie rule. Existing agents keep their run-card format.
Diagnostics describe the last decision, including first actions in the simulated
decision count; the immediate single-action path reports zero episodes and no
estimated scores. Planning time is wall-clock time and is not reproducible.

These estimates depend on the fixed continuation policy and finite budget;
they are not optimal-play guarantees. More simulations increase compute cost
and can change the chosen action. There is no tree search or parallel rollout.
The default 32 is an experimental starting budget. Finite-sample action
selection can underperform the continuation policy, including prematurely
ending turns with usable resources. No confidence-aware fallback is implemented.

## Initial M4 pilot

On the same 100 encounter seeds (18000–18099), with planner seed 0 and the
original heuristic as continuation:

| Policy | Wins | Total evaluation time | Mean time per planned decision |
| --- | ---: | ---: | ---: |
| Original heuristic | 64/100 | 0.2 seconds | — |
| Rollout, 32 simulations per action | 47/100 | 204.9 seconds | 120.6 ms |
| Rollout, 64 simulations per action | 51/100 | 412.7 seconds | 233.7 ms |

Planned-decision timing excludes the single-legal-action fast path; timings are
observed local CPU measurements. These budgets did not improve the heuristic
on this small sample. At one state, 32 simulations favored ending the turn
(27 wins) over Cleave (26 wins). A separate 2,048-simulation check estimated
76.1% for ending the turn versus 86.3% for the best attack and 86.0% for Cleave,
supporting finite-sample selection error at that state. This is not evidence
for the overall performance of a larger-budget planner or the scenario's
optimal win rate. See the [pilot data](../reports/m4_rollout_pilot_18000_18099.json).

For planning without sampling noise, see [expectimax planning](EXPECTIMAX_PLANNING.md),
which enumerates exact chance outcomes and scores the frontier with a learned value.
