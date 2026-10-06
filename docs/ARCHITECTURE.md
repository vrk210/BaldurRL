# Architecture

## Long-term abstraction

ML code should eventually work through the same conceptual environment interface against either a Python combat simulator or actual Baldur's Gate 3:

```text
                    Agent
                      │
                   Action
                      │
                      ▼
                 BaldurEnv
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Python Simulator           BG3 Adapter
          │                       │
          └───────────┬───────────┘
                      ▼
          observation / reward /
          termination / metadata
```

The simulator therefore uses **semantic state and semantic actions**, rather than mouse coordinates or UI-specific behavior. This is a direction for later milestones, not an implemented interface. Keep the object graph shallow: characters store stats, resource counts, and known ability IDs; shared ability definitions live outside characters; mechanics remain functions.

## Module responsibilities

| Module | Owns | Must not own |
| --- | --- | --- |
| `combat/characters.py` | Entity state: `Character`, `Fighter`, `Goblin`; known `Action` IDs, per-turn `turn_refresh` mapping, not copied ability definitions | Rewards, turn loops, policies, attack-resolution orchestration |
| `combat/actors.py` | Stable `Side`/`ActorRef` addresses and `CombatRoster` ally/enemy organization | Combat rules, rewards, turn advancement |
| `combat/turns.py` | Deterministic `TurnManager`: order, current actor, round, wrap detection, dead-actor skipping | Dice, HP mutation, rewards, policies, attacks |
| `combat/legality.py` | `DecisionSpec` decisions, side-aware `resolve_target`, turn-agnostic per-actor `legal_mask_for` | Turn order, dice, HP mutation, rewards, policies |
| `combat/actions.py` | Semantic action intent: `ATTACK`, `SECOND_WIND`, `ACTION_SURGE`, `END_TURN` | Execution rules |
| `combat/resources.py` | `Resource` enum and `ResourcePool` counts; shared affordability, atomic spending, gaining, and setting counts | Effect-specific legality or effects |
| `combat/damage.py` | Immutable `DamageSpec(dice_count, die_size, bonus)` | Rolls or damage application |
| `combat/abilities.py` | Immutable `AbilitySpec(action, costs, target)` definitions and shared M0 catalog; excludes `END_TURN` | Execution methods or turn control |
| `combat/mechanics.py` | Game rules: dice rolls, attack resolution, damage, healing, critical hits, Action Surge | Rewards or policy choice |
| `combat/env.py` | Gymnasium coordinator: reset, flat observation encoding, action-index mapping and masks, one Fighter decision per step, fixed Goblin turn, reward call, termination, truncation | Duplicated combat formulas; call mechanics |
| `combat/simulation.py` | Independent M0–M4 simulator reconstruction from live observations and caller-supplied seeds | Live RNG access, policies, duplicated combat formulas |
| `agents/` | Action-selection policies, beginning with `RandomAgent` | Direct state mutation or combat formulas |
| `evaluation/` | Reproducible seeded episodes and external metrics: win rate, final HP, rounds, action usage | Combat rules |

## Stage progression

M0 remains the fixed 1v1 regression environment specified in `M0_SPEC.md`. The later stages are specified in `STAGES_SPEC.md`: M1A varies one opponent's stats; M1B has two fixed melee enemy slots and explicit target decisions; M2 adds one-use Cleave; M3 controls two allies against two enemies with an active-actor observation; M4 extends that to three scaled enemies. A small stage factory selects an environment for training and evaluation. Each environment publishes action labels and observation field names in index order so traces and run cards do not encode M0 assumptions.

Gym action indices represent decisions. A decision carries a semantic `Action` and, for a targeted attack, an enemy slot index. Target selection is separate from the shared ability identity. The environment owns target legality, turn order, outcomes, and reward selection; mechanics still own attack and ability effects. Character state holds resources and known abilities. Agents receive only the flat observation and action mask and return an index.

All three later stages use terminal win/loss reward for their main comparison. M2 also supports an isolated bounded damage-shaping training experiment. External evaluation always measures combat outcomes and behavior on seeded episodes, independent of the training reward scheme.

## Dependency direction

```text
actions + resources → abilities
actions + resources + damage → characters
abilities + characters → mechanics → environment → agents/evaluation
```

Avoid circular dependencies. Characters must not import ability execution or the RL environment. Mechanics must not import trained agents. Agents must not implement combat formulas. The environment must call mechanics rather than duplicate their rules. Generic resource affordability uses `ResourcePool.has`; living-target and missing-HP rules stay with the corresponding mechanics. `can_use_ability` checks known IDs and costs, not full effect-specific legality. `END_TURN` stays in environment turn control.

For M0, `BaldurCombatEnv.step()` handles exactly one Fighter decision. It dispatches mechanics and returns immediately for Fighter abilities; `END_TURN` alone triggers the automatic Goblin attack and round advance. Legal-action masks stay separate from the seven-field `float32` Box observation. The action space is an explicit four-index Discrete space. Rewards use an injectable callable receiving an immutable before snapshot, the selected semantic `Action`, an immutable after snapshot, and termination/truncation flags; the default is terminal-only. The environment alone owns episode completion and the round-50 truncation limit.

`RandomAgent` is the first baseline policy. It chooses uniformly from the legal-action mask using its own seeded NumPy generator; the environment's separate seeded generator handles combat rolls. The evaluation harness runs policies over explicit environment episode seeds (by default `0..9999`) and reports win rate as the primary baseline metric, along with outcomes, final HP, rounds, and action usage. Run it with `python -m evaluation.evaluate`.

`RolloutAgent` evaluates every legal first action by following independent fixed
continuation policies through terminal-reward simulators restored from the
observation. Combat code owns reconstruction; the agent only chooses and scores
actions through `step()`. A small `agents.policy.Policy` structural protocol is
shared with evaluation. See [rollout planning](ROLLOUT_PLANNING.md) for factory
isolation, budgets, diagnostics, and CLI usage.

For decision-level inspection, evaluation can write one JSONL record per episode. Each record contains the outcome and every Fighter decision with the observation, legal-action mask, chosen action, reward, next observation, termination flags, and diagnostic `info` (including attack rolls). A run card records the policy or model, exact episode seeds, field and action names, aggregate metrics, and loss seeds. This is evaluation data only; it does not change the policy observation or combat transitions. Training also saves traces and a run card for its final evaluation. Use a separate seed range for final comparison because periodic model selection already uses the training run's evaluation seeds.
`evaluation.compare` compares two run cards evaluated on the same seeds and reports losses and the first trace divergence. Later dice rolls can differ after policies choose different actions, so the report does not attribute outcome differences solely to that first decision.

## Future BG3 integration

The platform-neutral [integration contract](BG3_INTEGRATION.md) separates rich BG3 facts from policy observations:

```text
BG3 collection → integration snapshots/events → observation adapter → policy vector
```

The collection harness preserves BG3 entity, ability, and resource IDs. The adapter later chooses fields and maps IDs for M0, M1, M2, or richer policies. The harness must not emit the current PPO vector directly. The `integration/` package is independent of `combat/`, Gymnasium, and training code; `combat/` does not import it. The current simulator evaluation JSONL traces remain a separate format containing already encoded observations and masks. BG3 collection, observation adaptation, and action execution are not implemented yet.
