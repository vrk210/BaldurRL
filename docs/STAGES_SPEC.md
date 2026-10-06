# M1A, M1B, M2, M3, and M4 simulator specifications

M0 remains specified by [M0_SPEC.md](M0_SPEC.md). M1A, M1B, and M2 reuse its Fighter preset, d20 attack rules, Second Wind, Action Surge, turn resources, fixed melee range, round limit of 50, and one Fighter decision per Gymnasium step. The enemies always attack in slot order after `END_TURN`. Victory occurs when all enemies die; defeat occurs when the Fighter dies. Terminal reward is +1 for victory, -1 for defeat, and 0 otherwise; truncation gives 0. M3 and M4 extend the same rules to two controlled allies as specified below.

These stages are **simulator choices**, not verified BG3 encounter or enemy stat blocks. On every reset, each enemy independently samples max HP 8–20, AC 12–18, attack bonus 2–6, one damage die from d4/d6/d8, and damage bonus 0–4. Current HP starts at max HP. Sampling and subsequent rolls use the environment's seeded RNG.

## M1A

One randomized enemy. Actions are `0 ATTACK`, `1 SECOND_WIND`, `2 ACTION_SURGE`, `3 END_TURN`.

Float32 observation order:

```text
fighter_hp, fighter_action_count, fighter_bonus_action_count,
fighter_second_wind_count, fighter_action_surge_count,
enemy_hp, enemy_max_hp, enemy_ac, enemy_attack_bonus,
enemy_damage_die_size, enemy_damage_bonus, round_number
```

## M1B

Two independently randomized enemies. Actions are `0 ATTACK_ENEMY_0`, `1 ATTACK_ENEMY_1`, `2 SECOND_WIND`, `3 ACTION_SURGE`, `4 END_TURN`. A dead target's attack is illegal. Both enemies attack in slot order after `END_TURN`; if the first kills the Fighter, the second does not attack.

Float32 observation order:

```text
fighter_hp, fighter_action_count, fighter_bonus_action_count,
fighter_second_wind_count, fighter_action_surge_count,
enemy_0_hp, enemy_0_max_hp, enemy_0_ac, enemy_0_attack_bonus,
enemy_0_damage_die_size, enemy_0_damage_bonus,
enemy_1_hp, enemy_1_max_hp, enemy_1_ac, enemy_1_attack_bonus,
enemy_1_damage_die_size, enemy_1_damage_bonus, round_number
```

## M2

M1B plus a simplified Cleave. The Fighter knows `CLEAVE` and starts with `CLEAVE = 1`, once per encounter. This represents a short-rest-recharge weapon ability only as a simulator simplification. Cleave costs one Action and one Cleave resource, and attacks each living enemy separately. Each attack follows normal hit, miss, and critical rules. On hit, resolve normal weapon damage, then deal `max(1, floor(damage / 2))` to that target. Critical dice are doubled before halving. No range cone or geometry is modeled. These are simulator choices, not a claim of exact BG3 Cleave rules.

Actions are `0 ATTACK_ENEMY_0`, `1 ATTACK_ENEMY_1`, `2 CLEAVE`, `3 SECOND_WIND`, `4 ACTION_SURGE`, `5 END_TURN`.

Float32 observation order:

```text
fighter_hp, fighter_action_count, fighter_bonus_action_count,
fighter_second_wind_count, fighter_action_surge_count, fighter_cleave_count,
enemy_0_hp, enemy_0_max_hp, enemy_0_ac, enemy_0_attack_bonus,
enemy_0_damage_die_size, enemy_0_damage_bonus,
enemy_1_hp, enemy_1_max_hp, enemy_1_ac, enemy_1_attack_bonus,
enemy_1_damage_die_size, enemy_1_damage_bonus, round_number
```

`reward_mode="damage"` is an optional M2 training experiment: each Fighter decision adds `0.2 * (total enemy HP before - total enemy HP after) / initial total enemy max HP` to the terminal reward. Enemy attacks do not contribute. External evaluation should use terminal reward and outcome metrics.

## M3

Two preset Fighters (same M0 preset, each with its own once-per-encounter Cleave, Second Wind, and Action Surge pools) versus two independently randomized enemies sampled exactly like M1B. Fixed turn order is `ALLY 0, ALLY 1, ENEMY 0, ENEMY 1`; dead actors are skipped. The same policy controls whichever ally is active. Each ally takes one or more decisions, then `END_TURN` passes play to the next actor: `ALLY 0` hands to `ALLY 1` with no enemy phase, and `ALLY 1`'s `END_TURN` runs both living enemies in slot order before the round advances. Enemies use the fixed first-legal-`ATTACK` policy, which targets the lowest living ally slot. Victory occurs when all enemies die; defeat requires all allies dead, so combat continues with one surviving ally. Terminal reward is +1 for victory, -1 for defeat, and 0 otherwise; truncation gives 0. Only terminal reward is supported.

Actions are `0 ATTACK_ENEMY_0`, `1 ATTACK_ENEMY_1`, `2 CLEAVE`, `3 SECOND_WIND`, `4 ACTION_SURGE`, `5 END_TURN`, evaluated for the active ally. Masks are all-false unless the current actor is a controlled ally.

Float32 observation order:

```text
active_actor_index,
ally_0_hp, ally_0_action_count, ally_0_bonus_action_count,
ally_0_second_wind_count, ally_0_action_surge_count, ally_0_cleave_count,
ally_1_hp, ally_1_action_count, ally_1_bonus_action_count,
ally_1_second_wind_count, ally_1_action_surge_count, ally_1_cleave_count,
enemy_0_hp, enemy_0_max_hp, enemy_0_ac, enemy_0_attack_bonus,
enemy_0_damage_die_size, enemy_0_damage_bonus,
enemy_1_hp, enemy_1_max_hp, enemy_1_ac, enemy_1_attack_bonus,
enemy_1_damage_die_size, enemy_1_damage_bonus, round_number
```

`active_actor_index` is the acting ally's slot mid-episode; on terminal observations it reports the first living ally slot, or 0 when no ally survives. Info reports `ally_hps` instead of `fighter_hp`. Evaluation counts a loss only when every ally is dead, and its final-HP metric is total remaining ally HP.

## M4

M3 expanded to three enemies with scaled sampling, as a first fairness guess for 2v3. Each enemy independently samples max HP 12–24, AC 12–18, attack bonus 3–7, one damage die from d4/d6/d8, and damage bonus 2–6, for means near Fighter parity (18 HP, +5 attack, ~7.5 damage). These ranges are provisional simulator tuning, not verified BG3 creature distributions. Turn order is `ALLY 0, ALLY 1, ENEMY 0, ENEMY 1, ENEMY 2`; dead actors are skipped, combat continues while any ally lives, and enemies attack the lowest living ally slot in slot order. Victory, defeat, terminal rewards, and truncation match M3. Only terminal reward is supported.

Actions are `0 ATTACK_ENEMY_0`, `1 ATTACK_ENEMY_1`, `2 ATTACK_ENEMY_2`, `3 CLEAVE`, `4 SECOND_WIND`, `5 ACTION_SURGE`, `6 END_TURN`, evaluated for the active ally. Cleave attacks each living enemy with the same halved-damage rule as M2.

Float32 observation order:

```text
active_actor_index,
ally_0_hp, ally_0_action_count, ally_0_bonus_action_count,
ally_0_second_wind_count, ally_0_action_surge_count, ally_0_cleave_count,
ally_1_hp, ally_1_action_count, ally_1_bonus_action_count,
ally_1_second_wind_count, ally_1_action_surge_count, ally_1_cleave_count,
enemy_0_hp, enemy_0_max_hp, enemy_0_ac, enemy_0_attack_bonus,
enemy_0_damage_die_size, enemy_0_damage_bonus,
enemy_1_hp, enemy_1_max_hp, enemy_1_ac, enemy_1_attack_bonus,
enemy_1_damage_die_size, enemy_1_damage_bonus,
enemy_2_hp, enemy_2_max_hp, enemy_2_ac, enemy_2_attack_bonus,
enemy_2_damage_die_size, enemy_2_damage_bonus, round_number
```

Info, `active_actor_index` terminal reporting, and evaluation accounting match M3.
