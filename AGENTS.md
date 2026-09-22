# BaldurRL: instructions for coding agents

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

The agent chooses actions. The environment coordinates episodes, turns, legality, rewards, and termination. Mechanics implement combat rules. Character classes store state. Agents must never directly modify character state. RL algorithms must not contain combat mechanics.

| Module | Responsibility | Exclusions |
| --- | --- | --- |
| `combat/characters.py` | State representations: `Character`, `Fighter`, `Goblin` | Attack resolution, rewards, environment turn loops, RL algorithms |
| `combat/actions.py` | Semantic intent: `ATTACK`, `SECOND_WIND`, `ACTION_SURGE`, `END_TURN` | Action execution |
| `combat/mechanics.py` | d20 and damage rolls, critical hits, attack resolution, Second Wind, Action Surge | RL rewards |
| `combat/env.py` | Future Gymnasium environment: episode reset, observations, legal actions and masks, action dispatch, turn progression, rewards, termination, truncation | Combat formulas; call `mechanics.py` |
| `agents/` | Future `RandomAgent`, `HeuristicAgent`, and trained policies; action selection only | Direct environment-state mutation |
| `evaluation/` | Future reproducible evaluation: win rate, remaining HP, turns to victory, resource usage | Combat rules |

## Coding rules

- Use Python type hints; prefer simple dataclasses and functions.
- Avoid speculative abstractions and unnecessary dependencies.
- Keep functions small and testable. Write deterministic tests whenever possible.
- Use supplied seeded RNG objects, not global randomness, for combat rules.
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
