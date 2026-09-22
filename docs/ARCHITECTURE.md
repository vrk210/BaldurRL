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

The simulator therefore uses **semantic state and semantic actions**, rather than mouse coordinates or UI-specific behavior. This is a direction for later milestones, not an implemented interface.

## Module responsibilities

| Module | Owns | Must not own |
| --- | --- | --- |
| `combat/characters.py` | Entity state: `Character`, `Fighter`, `Goblin` | Rewards, turn loops, policies, attack-resolution orchestration |
| `combat/actions.py` | Semantic action intent: `ATTACK`, `SECOND_WIND`, `ACTION_SURGE`, `END_TURN` | Execution rules |
| `combat/mechanics.py` | Game rules: dice rolls, attack resolution, damage, healing, critical hits, Action Surge | Rewards or policy choice |
| `combat/env.py` | Future coordinator: reset, observations, action masks and legality, turn progression, rewards, termination, truncation | Duplicated combat formulas; call mechanics |
| `agents/` | Future action-selection policies: `RandomAgent`, `HeuristicAgent`, MaskablePPO | Direct state mutation or combat formulas |
| `evaluation/` | Future reproducible experiments: win rate, remaining HP, turns to victory, resource usage | Combat rules |

## Dependency direction

```text
actions ──────────────┐
                      ▼
characters → mechanics → environment → agents/evaluation
```

Avoid circular dependencies. Characters must not import the RL environment. Mechanics must not import trained agents. Agents must not implement combat formulas. The environment must call mechanics rather than duplicate their rules.

## Future BG3 integration

```text
BG3
 │
 ├── Osiris events / queries
 ├── Script Extender / Lua / entity state
 └── action execution
          │
          ▼
       BG3Adapter
          │
          ▼
same semantic observations/actions used by simulator
```

The intended policy input is abstract combat state, not UI input. Human-play data should eventually be converted into the same semantic state/action representation. The exact BG3 bridge and API mapping are **TODO_VERIFY**; do not design them yet.
