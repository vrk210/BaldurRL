# BaldurRL: instructions for coding agents

Prefer stable abstractions and implementations expected to survive future milestones when doing so adds little complexity. Do not overengineer speculative future functionality solely to avoid hypothetical rewrites.

## Project goal and current milestone

BaldurRL's long-term goal is an autonomous agent capable of completing Baldur's Gate 3. The current milestone, **M0**, is intentionally smaller: train and evaluate an agent in a simplified 1v1 Level 2 Fighter-vs-Goblin combat simulator. Do not implement future milestones unless explicitly requested.

## Documentation priority

Before modifying combat code, read in order:

1. [docs/M0_SPEC.md](docs/M0_SPEC.md)
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. [docs/GAME_KNOWLEDGE.md](docs/GAME_KNOWLEDGE.md)

`docs/M0_SPEC.md` is authoritative for current simulator behavior. If code and documentation disagree, report the inconsistency; do not silently choose one.

## M0 scope

M0 has exactly one Fighter and one Goblin at fixed melee range. It has no movement, terrain, elevation, line of sight, opportunity attacks, inventory, consumables, party members, dialogue, quests, BG3 API integration, computer vision, or LLM calls. Do not add these systems unless explicitly requested.

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
| `combat/characters.py` | State representations: `Character`, `Fighter`, `Goblin`, with resources and known ability IDs | Attack resolution, rewards, environment turn loops, RL algorithms |
| `combat/actions.py` | Semantic intent: `ATTACK`, `SECOND_WIND`, `ACTION_SURGE`, `END_TURN` | Action execution |
| `combat/resources.py` | Typed `Resource` counts and shared `ResourcePool` affordability/spending | Ability-specific effects |
| `combat/damage.py` | Immutable damage dice and bonus data | Damage rolls |
| `combat/abilities.py` | Immutable `AbilitySpec` catalog for M0 abilities and their costs/targets | Executable effects; `END_TURN` remains turn control |
| `combat/mechanics.py` | d20 and damage rolls, critical hits, attack resolution, Second Wind, Action Surge | RL rewards |
| `combat/env.py` | Gymnasium M0 environment: one Fighter decision per step, fixed Goblin turn after `END_TURN`, observations, action masks, rewards, termination, truncation | Combat formulas; call `mechanics.py` |
| `agents/` | Future `RandomAgent`, `HeuristicAgent`, and trained policies; action selection only | Direct environment-state mutation |
| `evaluation/` | Future reproducible evaluation: win rate, remaining HP, turns to victory, resource usage | Combat rules |

## Coding rules

- Use Python type hints; prefer simple dataclasses and functions.
- Avoid speculative abstractions and unnecessary dependencies.
- Keep functions small and testable. Write deterministic tests whenever possible.
- Use caller-supplied `numpy.random.Generator` instances, not global randomness, for combat rules.
- Check known ability IDs and shared resource costs before applying an effect; keep target and HP restrictions specific to that effect.
- Keep the four action indices and seven observation fields in `docs/M0_SPEC.md` stable. Keep masks separate from observations, and use only the environment's seeded RNG for combat transitions.
- Do not print inside reusable mechanics functions.
- Do not silently catch programming errors or prematurely optimize.

## Missing-mechanic rule

> If a mechanic required for implementation is absent from `docs/M0_SPEC.md`, do not infer it from general D&D or BG3 knowledge.

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
