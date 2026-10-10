# BaldurRL: instructions for coding agents

Prefer stable abstractions and implementations expected to survive future milestones when doing so adds little complexity. Do not overengineer speculative future functionality solely to avoid hypothetical rewrites.

## Project goal and current stages

BaldurRL's long-term goal is an autonomous agent capable of completing Baldur's Gate 3. **M0** is the fixed 1v1 regression baseline. **M1A** randomizes one opponent, **M1B** adds a second opponent and target choice, **M2** adds the simplified Cleave decision, **M3** controls two allies against two enemies, **M4** is 2v3 with scaled enemies, **M5** adds enemy roles, varied enemy targeting, advantage/disadvantage, Trip/Prone, and Dodge, and **M6** adds a two-rank positioning abstraction to M5. Implement only these stages unless a later task explicitly expands scope.

## Documentation priority

Before modifying combat code, read in order:

1. [docs/M0_SPEC.md](docs/M0_SPEC.md)
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. [docs/GAME_KNOWLEDGE.md](docs/GAME_KNOWLEDGE.md)

For M1A through M6, also read the stage specification before editing their rules.

`docs/M0_SPEC.md` is authoritative for M0 behavior; the stage specification governs M1A, M1B, M2, M3, M4, M5, and M6. If code and documentation disagree, report the inconsistency; do not silently choose one.

## Scope

M0 has exactly one Fighter and one Goblin at fixed melee range. M1A retains one enemy with randomized stats. M1B and M2 have two enemies at fixed melee range. M3 has two controlled Fighter allies and two enemies at fixed melee range. M4 has two allies and three scaled enemies at fixed melee range. M5 has two allies against a Brute, an Archer, and a Healer with no geometry. M6 adds front/back ranks, a per-turn Movement resource, Advance/Disengage, and opportunity attacks; it is an abstract two-rank model, not movement on a map. All stages omit terrain, elevation, line of sight, distances, inventory, consumables, party members, dialogue, quests, BG3 API integration, computer vision, and LLM calls, and every stage except M6 also omits movement and opportunity attacks. Do not add these systems unless explicitly requested. M5/M6 mechanics are owner-authorized simulator choices documented in the stage specification, not verified BG3 rules.

## Design and module boundaries

```text
Agent
  ↓
Action
  ↓
Environment
  ↓
Mechanics
  ↓
Character state
```

The agent chooses actions. The environment coordinates episodes, turns, legality, rewards, and termination. Mechanics implement combat rules. Character classes store stats, `ResourcePool` state, and known `Action` IDs. Shared immutable `AbilitySpec` definitions describe costs and targets, not execution. Agents must never directly modify character state. RL algorithms must not contain combat mechanics.

| Module | Responsibility | Exclusions |
| --- | --- | --- |
| `combat/characters.py` | State representations: `Character`, `Fighter`, `Goblin`, with resources, known ability IDs, and per-turn refresh | Attack resolution, rewards, environment turn loops, RL algorithms |
| `combat/actions.py` | Semantic ability intent, including `CLEAVE`, and `END_TURN` turn control | Action execution |
| `combat/resources.py` | Typed `Resource` counts and shared `ResourcePool` affordability/spending | Ability-specific effects |
| `combat/damage.py` | Immutable damage dice and bonus data | Damage rolls |
| `combat/abilities.py` | Immutable `AbilitySpec` catalogs for stage abilities and their costs/targets | Executable effects; `END_TURN` remains turn control |
| `combat/actors.py` | Stable `Side`/`ActorRef` addresses and `CombatRoster` ally/enemy organization | Combat rules, rewards, turn advancement |
| `combat/turns.py` | Deterministic `TurnManager`: order, current actor, round, wrap detection, dead-actor skipping | Dice, HP mutation, rewards, policies, attacks |
| `combat/legality.py` | `DecisionSpec` decisions, side-aware targets, turn-agnostic per-actor `legal_mask_for` and M5/M6 `tactical_legal_mask` | Turn order, dice, HP mutation, rewards, policies |
| `combat/mechanics.py` | d20 and damage rolls, critical hits, attack resolution, Second Wind, Action Surge, Cleave | RL rewards |
| `combat/rolls.py` | Advantage/disadvantage roll modes and moded attack resolution (M5/M6) | RL rewards, environment loops |
| `combat/tactics.py` | M5/M6 rules: roles' tuning data, catalog costs, healing, Trip/Prone, Dodge, Healer, enemy targeting, ranks and reach, Advance/opportunity attacks, Disengage, turn-start/turn-end conditions | Environment loops, rewards, policies |
| `combat/env.py` | Gymnasium M0 environment: one Fighter decision per step, fixed Goblin turn after `END_TURN`, observations, action masks, rewards, termination, truncation | Combat formulas; call `mechanics.py` |
| `combat/stages.py` | M1A/M1B/M2/M3/M4 environments, shallow target decisions, stage factory (M0–M6) and metadata | Combat formulas; call `mechanics.py` |
| `combat/tactical_env.py` | M5/M6 Gymnasium environment: sampling, turns, masks, observations, rewards, termination, per-stage `STAGE_RULES` | Combat formulas; call `tactics.py`/`mechanics.py` |
| `combat/simulation.py` | Restore independent simulators (M0–M6) from live observations | Live RNG access, policies, combat formulas |
| `agents/` | Random, stage heuristic (including named M5/M6 variants), rollout, and trained policies; action selection only | Direct environment-state mutation, combat rules |
| `evaluation/` | Reproducible stage evaluation: win rate, remaining HP, rounds, action and target usage, kill order by role, paired comparisons, M5/M6 behavior summaries | Combat rules |

## Coding rules

- Use Python type hints; prefer simple dataclasses and functions.
- Avoid speculative abstractions and unnecessary dependencies.
- Keep functions small and testable. Write deterministic tests whenever possible.
- Use caller-supplied `numpy.random.Generator` instances, not global randomness, for combat rules.
- Check known ability IDs and shared resource costs before applying an effect; keep target and HP restrictions specific to that effect.
- Keep the four M0 action indices and seven M0 observation fields in `docs/M0_SPEC.md` stable. Stage-specific indices and fields are specified separately. Keep masks separate from observations, and use only the environment's seeded RNG for combat transitions.
- Do not print inside reusable mechanics functions.
- Do not silently catch programming errors or prematurely optimize.

## Missing-mechanic rule

> If a mechanic required for implementation is absent from the relevant stage specification, do not infer it from general D&D or BG3 knowledge.

Identify the missing specification, do not guess, and propose or request clarification before implementing it. M0 deliberately simplifies real BG3.

## Implementation order

```text
combat mechanics
→ environment
→ deterministic tests
→ random baseline
→ heuristic baseline
→ evaluation harness
→ MaskablePPO
```

Do not jump ahead unless explicitly requested.

## Documentation maintenance

Changes to combat mechanics, actions, observations, legal-action rules, rewards, termination, or architecture boundaries must update the appropriate documentation in the same change.
