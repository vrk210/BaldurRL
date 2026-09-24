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
| `combat/characters.py` | Entity state: `Character`, `Fighter`, `Goblin`; known `Action` IDs, not copied ability definitions | Rewards, turn loops, policies, attack-resolution orchestration |
| `combat/actions.py` | Semantic action intent: `ATTACK`, `SECOND_WIND`, `ACTION_SURGE`, `END_TURN` | Execution rules |
| `combat/resources.py` | `Resource` enum and `ResourcePool` counts; shared affordability, atomic spending, and setting counts | Effect-specific legality or effects |
| `combat/damage.py` | Immutable `DamageSpec(dice_count, die_size, bonus)` | Rolls or damage application |
| `combat/abilities.py` | Immutable `AbilitySpec(action, costs, target)` definitions and shared M0 catalog; excludes `END_TURN` | Execution methods or turn control |
| `combat/mechanics.py` | Game rules: dice rolls, attack resolution, damage, healing, critical hits, Action Surge | Rewards or policy choice |
| `combat/env.py` | Gymnasium coordinator: reset, flat observation encoding, action-index mapping and masks, one Fighter decision per step, fixed Goblin turn, reward call, termination, truncation | Duplicated combat formulas; call mechanics |
| `agents/` | Future action-selection policies: `RandomAgent`, `HeuristicAgent`, MaskablePPO | Direct state mutation or combat formulas |
| `evaluation/` | Future reproducible experiments: win rate, remaining HP, turns to victory, resource usage | Combat rules |

## Dependency direction

```text
actions + resources → abilities
actions + resources + damage → characters
abilities + characters → mechanics → environment → agents/evaluation
```

Avoid circular dependencies. Characters must not import ability execution or the RL environment. Mechanics must not import trained agents. Agents must not implement combat formulas. The environment must call mechanics rather than duplicate their rules. Generic resource affordability uses `ResourcePool.has`; living-target and missing-HP rules stay with the corresponding mechanics. `can_use_ability` checks known IDs and costs, not full effect-specific legality. `END_TURN` stays in environment turn control.

For M0, `BaldurCombatEnv.step()` handles exactly one Fighter decision. It dispatches mechanics and returns immediately for Fighter abilities; `END_TURN` alone triggers the automatic Goblin attack and round advance. Legal-action masks stay separate from the seven-field `float32` Box observation. The action space is an explicit four-index Discrete space. Rewards use an injectable callable receiving immutable before/after state snapshots; the default is terminal-only. The environment alone owns episode completion and the round-50 truncation limit.

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
