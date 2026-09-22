# M0 specification (authoritative)

## Objective

M0 is one preset Level 2 Fighter versus one preset Goblin in a fixed melee encounter. Its goal is to validate `state → action → transition → reward → policy learning → evaluation` in a small, understandable, testable environment, not to perfectly simulate BG3.

## Encounter

- Exactly one learning-agent-controlled Fighter and one fixed-policy Goblin.
- Both start within melee range. The Fighter always acts first; initiative is not modeled.
- No movement, terrain, elevation, line of sight, opportunity attacks, inventory, consumables, or additional creatures.
- The Goblin's entire policy is: if the Fighter is alive, `ATTACK` the Fighter, then `END_TURN`. Its attack and damage rolls remain stochastic. It cannot move, flee, disengage, use items, or choose an alternative attack.

## Character state

The shared `Character` state is:

```text
name: str
max_hp: int
hp: int
armor_class: int
attack_bonus: int
damage_dice_count: int
damage_die_size: int
damage_bonus: int
```

`alive` is true exactly when `hp > 0`. HP must never fall below zero.

`Fighter` adds `action_available`, `bonus_action_available`, `second_wind_available`, and `action_surge_available` (all `bool`). At the start of each normal Fighter turn, set `action_available = True` and `bonus_action_available = True`. Second Wind and Action Surge are once-per-encounter resources; they do not refresh each turn. `Goblin` has no additional class-specific state.

## Fighter action space and legality

Exactly four actions exist:

| Action | Legal exactly when | Effect |
| --- | --- | --- |
| `ATTACK` | `fighter.action_available` and `goblin.alive` | Set `action_available = False`; resolve attack. |
| `SECOND_WIND` | `fighter.second_wind_available` and `fighter.bonus_action_available` and `fighter.hp < fighter.max_hp` | Consume bonus action; set `second_wind_available = False`; heal `1d10 + 2`, capped at `max_hp`. The `+2` is for the M0 Level 2 Fighter. |
| `ACTION_SURGE` | `fighter.action_surge_available` | Set `action_surge_available = False` and `action_available = True`. May restore a consumed action; does not consume the restored action. |
| `END_TURN` | `fighter.alive` and `goblin.alive` | End Fighter turn; automatically execute Goblin turn. |

Every selected action must be legal at the moment of selection. No other actions exist.

## Attack resolution and damage

Roll `1d20` using a supplied RNG:

- Natural 1: automatic miss.
- Natural 20: automatic hit and critical hit.
- Otherwise: hit if `d20 + attack_bonus >= defender.armor_class`.

Normal damage is `damage_dice_count` rolls of a `damage_die_size` die, plus `damage_bonus` (for example, `1d8 + 3`). A critical hit doubles the **number of damage dice**, not the static bonus (for example, `2d8 + 3`). Damage cannot reduce HP below zero.

All stochastic behavior uses an RNG supplied by the environment or caller. Core combat mechanics must not use global randomness.

## Turn lifecycle

```text
Fighter turn begins
  → action_available = True; bonus_action_available = True
  → agent selects one or more legal actions
  → agent eventually selects END_TURN
  → Goblin attacks Fighter if both remain alive, then ends its turn
  → next Fighter turn
```

Valid sequences, subject to legality at each step: `ATTACK → ACTION_SURGE → ATTACK → END_TURN`; `SECOND_WIND → ATTACK → END_TURN`.

## Episode outcome and reward

- Victory: `goblin.hp == 0`; terminal reward `+1`.
- Defeat: `fighter.hp == 0`; terminal reward `-1`.
- Otherwise: reward `0`.
- Future safety limit: `MAX_ROUNDS = 50`. If reached without victory or defeat, report **truncation**, not ordinary termination.
- Do not add damage or healing rewards, turn penalties, resource bonuses, or other reward shaping. The objective is to win; shaping may be tested later.

## Initial observation candidates

The future environment should provide enough information for a knowledgeable player to decide. Candidates are:

```text
fighter_hp, fighter_max_hp, fighter_armor_class, fighter_attack_bonus
action_available, bonus_action_available
second_wind_available, action_surge_available
goblin_hp, goblin_max_hp, goblin_armor_class, goblin_attack_bonus
round_number
```

Damage statistics may be added when enemy statistics vary. Numerical normalization is **not yet specified**. The agent may know normal Goblin combat statistics (for example, attack bonus `+4`) but never future RNG outcomes (for example, its next d20 result).

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
