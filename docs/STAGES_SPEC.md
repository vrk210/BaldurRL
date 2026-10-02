# M1A, M1B, and M2 simulator specifications

M0 remains specified by [M0_SPEC.md](M0_SPEC.md). These stages reuse its Fighter preset, d20 attack rules, Second Wind, Action Surge, turn resources, fixed melee range, round limit of 50, and one Fighter decision per Gymnasium step. The enemies always attack in slot order after `END_TURN`. Victory occurs when all enemies die; defeat occurs when the Fighter dies. Terminal reward is +1 for victory, -1 for defeat, and 0 otherwise; truncation gives 0.

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
