# M0 specification (authoritative)

## Objective

M0 is one preset Level 2 Fighter versus one preset Goblin in a fixed melee encounter. Its goal is to validate `state → action → transition → reward → policy learning → evaluation` in a small, understandable, testable environment, not to perfectly simulate BG3.

## Encounter

- Exactly one learning-agent-controlled Fighter and one fixed-policy Goblin.
- Both start within melee range. The Fighter always acts first; initiative is not modeled.
- No movement, terrain, elevation, line of sight, opportunity attacks, inventory, consumables, or additional creatures.
- The Goblin's entire policy is: if the Fighter is alive, `ATTACK` the Fighter, then `END_TURN`. Its attack and damage rolls remain stochastic. It cannot move, flee, disengage, use items, or choose an alternative attack.

The fixed M0 presets are prototype choices, not verified BG3 stat blocks:

| Character | HP | AC | Attack bonus | Damage |
| --- | ---: | ---: | ---: | --- |
| Level 2 Fighter | 20 | 16 | +5 | 1d8 + 3 |
| Goblin | 15 | 15 | +4 | 1d6 + 2 |

## Character state

The shared `Character` state is:

```text
name: str
max_hp: int
hp: int
armor_class: int
attack_bonus: int
damage: DamageSpec
resources: ResourcePool
known_abilities: frozenset[Action]
```

`alive` is true exactly when `hp > 0`. HP must never fall below zero.

`DamageSpec` is immutable and contains `dice_count: int`, `die_size: int`, and `bonus: int`. It replaces three separate character damage fields.

`ResourcePool` stores nonnegative integer counts keyed by `Resource`. M0 uses exactly `ACTION`, `BONUS_ACTION`, `SECOND_WIND`, and `ACTION_SURGE`. Missing entries count as zero. Callers use `get`, `has`, `spend`, `gain`, and `set`; an unaffordable spend fails without changing any count. `gain` adds a nonnegative amount to the current count.

Initial Fighter resources are `ACTION = 1`, `BONUS_ACTION = 1`, `SECOND_WIND = 1`, `ACTION_SURGE = 1`. At the start of each normal Fighter turn, set `ACTION = 1` and `BONUS_ACTION = 1`. Second Wind and Action Surge are once-per-encounter resources and do not refresh each turn. The Goblin has no additional class-specific state; its initial resource pool has `ACTION = 1` for its attack, and that Action refreshes at the start of each normal Goblin turn so it can follow its fixed policy every turn.

Known ability IDs are immutable. The Fighter knows `ATTACK`, `SECOND_WIND`, and `ACTION_SURGE`; the Goblin knows `ATTACK`. Characters do not own ability definitions.

## Ability definitions and legality

The shared immutable M0 ability catalog maps an `Action` to an `AbilitySpec(action, costs, target)`. It contains `ATTACK` targeting `ENEMY`, `SECOND_WIND` targeting `SELF`, and `ACTION_SURGE` targeting `SELF`. `END_TURN` is environment turn control and is not an ability. `AbilitySpec` contains no executable effect.

Generic ability eligibility requires that the character knows the action and its `ResourcePool` can pay all catalog costs. Spending those costs uses the same pool operation for every ability. Additional semantic conditions, such as a living attack target or missing HP for Second Wind, remain separate. `can_use_ability` checks only known IDs and resource costs; it is not a complete action mask.

## Fighter action space and legality

Exactly four actions exist:

| Action | Legal exactly when | Effect |
| --- | --- | --- |
| `ATTACK` | Fighter knows it; `ACTION >= 1`; `goblin.alive` | Spend `ACTION: 1`; resolve attack. |
| `SECOND_WIND` | Fighter knows it; `BONUS_ACTION >= 1`; `SECOND_WIND >= 1`; `fighter.hp < fighter.max_hp` | Spend `BONUS_ACTION: 1` and `SECOND_WIND: 1`; heal `1d10 + 2`, capped at `max_hp`. The `+2` is for the M0 Level 2 Fighter. |
| `ACTION_SURGE` | Fighter knows it; `ACTION_SURGE >= 1` | Spend `ACTION_SURGE: 1`; gain one additional `ACTION`. The Fighter can have up to two Actions in M0. |
| `END_TURN` | `fighter.alive` and `goblin.alive` | End Fighter turn; automatically execute Goblin turn. |

Every selected action must be legal at the moment of selection. No other actions exist.

## Gymnasium interface

`BaldurCombatEnv` exposes `spaces.Discrete(4)` with this fixed index mapping: `0 → ATTACK`, `1 → SECOND_WIND`, `2 → ACTION_SURGE`, `3 → END_TURN`. Enum numeric values do not define the mapping.

One `step()` is **one Fighter decision**. `ATTACK`, `SECOND_WIND`, and `ACTION_SURGE` apply their effects and return immediately while it remains the Fighter's turn if combat continues. Only `END_TURN` runs the fixed Goblin policy. The Goblin has no separate agent step. State changes are observable after each decision.

`action_masks()` returns a separate boolean array in the same index order. Its first three entries combine known ability, catalog cost affordability, and any effect-specific rule (living Goblin for Attack; missing Fighter HP for Second Wind). `END_TURN` is legal while both combatants live. A terminal or truncated episode has an all-false mask. The mask is not part of the observation. An invalid or masked action index raises `ValueError`; it does not receive a reward penalty. Calling `step()` after an episode ends requires `reset()` first.

`reset(seed=None, options=None)` creates fresh presets and starts round 1 with a Fighter turn. The environment uses Gymnasium's `self.np_random` as its only RNG and passes it to mechanics. The initial resource counts are those specified above.

## Attack resolution and damage

Roll `1d20` using a supplied RNG:

- Natural 1: automatic miss.
- Natural 20: automatic hit and critical hit.
- Otherwise: hit if `d20 + attack_bonus >= defender.armor_class`.

`AttackResult.total_attack` records `d20_roll + attack_bonus` even for natural 1 and natural 20; those natural-roll rules still decide whether the attack hits.

Normal damage is `damage.dice_count` rolls of a `damage.die_size` die, plus `damage.bonus` (for example, `1d8 + 3`). A critical hit doubles the **number of damage dice**, not the static bonus (for example, `2d8 + 3`). Damage cannot reduce HP below zero.

All stochastic behavior uses a `numpy.random.Generator` supplied by the environment or caller. `Generator.integers(low, high)` has an exclusive upper bound: use `integers(1, 21)` for a d20, `integers(1, die_size + 1)` for damage dice, and `integers(1, 11)` for Second Wind's d10. Mechanics return ordinary Python `int` values, not NumPy integer scalars. Core combat mechanics must not use global randomness.

## Turn lifecycle and rounds

```text
Fighter turn begins
  → ACTION = 1; BONUS_ACTION = 1
  → agent selects one or more legal actions
  → agent eventually selects END_TURN
  → Goblin attacks Fighter if both remain alive, then ends its turn
  → next Fighter turn
```

Valid sequences, subject to legality at each step: `ATTACK → ACTION_SURGE → ATTACK → END_TURN`; `ACTION_SURGE → ATTACK → ATTACK → END_TURN`; `SECOND_WIND → ATTACK → END_TURN`.

Round 1 is the first Fighter/Goblin round. `END_TURN` causes exactly one Goblin Attack if both live. If combat continues after that attack, the round completes: increment `round_number`, refresh only Fighter `ACTION` and `BONUS_ACTION`, and begin the next Fighter turn. The Goblin's `ACTION` is made available at the start of its fixed turn. No turn resource refresh occurs after victory or defeat.

## Episode outcome and reward

- Victory: `goblin.hp == 0`; terminal reward `+1`.
- Defeat: `fighter.hp == 0`; terminal reward `-1`.
- Otherwise: reward `0`.
- `MAX_ROUNDS = 50`. If round 50 finishes without victory or defeat, return `terminated = False`, `truncated = True`, and leave `round_number = 50`. Never enter round 51.
- Death ends the episode immediately after the attack that caused it; a Fighter victory needs no subsequent `END_TURN`.
- Do not add damage or healing rewards, turn penalties, resource bonuses, or other reward shaping. The objective is to win; shaping may be tested later.

The default `terminal_reward(before, action, after, terminated, truncated)` callable implements these rewards. `action` is the semantic Fighter `Action` selected for this step; the terminal reward does not use it. `before` and `after` are separate immutable snapshots of Fighter HP, Goblin HP, round number, and Fighter resource counts. A different reward callable may be injected into the environment later without changing combat transitions or mechanics. No shaping function is currently implemented.

## M0 observation and info

The observation is one unnormalized `numpy.float32` `spaces.Box` vector in this exact stable order:

```text
0 fighter_hp
1 fighter_action_count
2 fighter_bonus_action_count
3 fighter_second_wind_count
4 fighter_action_surge_count
5 goblin_hp
6 round_number
```

HP ranges from zero to the preset maximum. Fighter Action count ranges from zero to two; each other listed resource count is zero or one in M0. Round number ranges from 1 to 50. Constant preset statistics (maximum HP, AC, attack bonuses, damage dice, and known ability IDs) are omitted. Numerical normalization is not used. The agent may know normal Goblin statistics but never future RNG outcomes.

The separate `info` dictionary reports current round, semantic action name (or `None` on reset), Fighter HP, and Goblin HP. Attack steps also report the relevant `AttackResult` fields, with Goblin attack details under a separate key after `END_TURN`. `info` is diagnostic, not part of the policy observation.

## Completion criteria

M0 is complete when:

1. Combat mechanics have deterministic unit tests.
2. The Gymnasium environment passes validation.
3. A random-action policy can complete episodes.
4. A heuristic baseline exists.
5. Thousands of seeded evaluation episodes can run.
6. MaskablePPO can train against the environment.
7. Trained performance can be compared with the baselines.

Movement, multiple enemies, party combat, imitation learning, and real BG3 integration belong to later milestones.
