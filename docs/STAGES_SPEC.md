# M1A–M6 simulator specifications

M0 remains specified by [M0_SPEC.md](M0_SPEC.md). M1A, M1B, and M2 reuse its Fighter preset, d20 attack rules, Second Wind, Action Surge, turn resources, fixed melee range, round limit of 50, and one Fighter decision per Gymnasium step. The enemies always attack in slot order after `END_TURN`. Victory occurs when all enemies die; defeat occurs when the Fighter dies. Terminal reward is +1 for victory, -1 for defeat, and 0 otherwise; truncation gives 0. M3 and M4 extend the same rules to two controlled allies as specified below.

For every staged environment, terminal outcome depends on the roster sides: all enemies dead with any ally surviving gives +1; all allies dead gives -1; otherwise it gives 0. An individual ally's death does not imply defeat. Current mechanics cannot defeat both sides in one transition, but if an injected transition does so, it explicitly counts as a defeat with terminal reward -1. M2 damage shaping is added to this terminal outcome using the unchanged formula below.

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

## M5 — roles and conditions

M5 and M6 are decision-rich stages added because M0–M4 are dominated by dice: almost every decision is forced or a near-tie. **Every rule in the M5 and M6 sections is a SIMULATOR CHOICE authorized by the project owner, not a verified BG3 rule.** BG3 has related ideas (advantage, prone, dodge, opportunity attacks, ranged and support enemies), but the numbers, durations, triggers, and enemy behaviors below were designed for this simulator and must not be cited as BG3 behavior.

M5 keeps the Fighter preset, the d20 attack rules, critical hits, Second Wind, Action Surge, the M2 Cleave rule, the two-ally turn model of M3/M4, terminal reward, and the round limit of 50. It adds heterogeneous enemy roles, varied enemy targeting, an advantage/disadvantage primitive, a Prone condition created by an ally Trip, and an ally Dodge.

### Roster and sampling

- Allies: two preset Fighters (20 HP, AC 16, +5, 1d8+3) exactly as in M4, each with its own `SECOND_WIND = 1`, `ACTION_SURGE = 1`, and `CLEAVE = 1`. They additionally know `TRIP` and `DODGE` and each start with `TRIP = 2` once-per-encounter Trip charges (a simulator stand-in for a limited maneuver budget; charges never refresh).
- Enemies: three slots. On reset the roles are a uniform random permutation of `BRUTE`, `ARCHER`, `HEALER` (exactly one of each, in random slots).
- Each enemy independently samples its statistics from its role profile (inclusive ranges) and its targeting policy uniformly from `WEAKEST`, `RETALIATE`, `RANDOM`. Current HP starts at max HP.

| Role | Max HP | AC | Attack bonus | Damage | Extra |
| --- | --- | --- | --- | --- | --- |
| Brute | 24–32 | 14–16 | +5–+7 | 1d10 + 3–5 | — |
| Archer | 10–16 | 12–14 | +6–+8 | 1d8 + 3–5 | ranged attacks |
| Healer | 12–18 | 12–14 | +2–+4 | 1d6 + 0–2 | `HEAL = 3` charges; heal 2d6+2 |

These are the final (v2) M5 ranges; see the tuning history below for v1.

RNG order at reset: `np_random.permutation(3)` assigns roles in the role order `BRUTE, ARCHER, HEALER` to the permuted slots; then for slot 0, 1, 2 in order: max HP, AC, attack bonus, damage bonus, targeting policy (one `integers` draw each). The damage die is fixed by role.

### Advantage and disadvantage (mechanics primitive)

An attack roll (or Trip roll) has a roll mode. `NORMAL` draws one d20. `ADVANTAGE` draws two d20 in order and keeps the higher; `DISADVANTAGE` draws two and keeps the lower. Sources do not stack: if at least one advantage source and at least one disadvantage source apply, the roll is `NORMAL` and draws a single d20. Natural 1 (automatic miss), natural 20 (automatic hit and critical), and the hit comparison all use the kept die. Diagnostics record every drawn die.

### Conditions

- **Prone** (enemies only). A successful Trip makes an enemy Prone. While Prone: (a) every ally attack roll against it (Attack, each Cleave roll) has advantage — all ally attacks are melee; (b) its own attack rolls (normal attacks and, in M6, opportunity attacks) have disadvantage; (c) it cannot be tripped again. Prone does not affect healing. An enemy stops being Prone at the end of its next turn. Because every enemy turn in a round follows both allied turns, Prone always lasts through the rest of the current allied turns and the enemy's own turn in that round. Dead enemies have no condition effects.
- **Dodging** (allies only). After `DODGE`, every enemy attack roll against that ally has disadvantage until the start of that ally's next turn.

### Ally actions

The same policy controls whichever ally is active. Index mapping:

| Index | Decision | Cost | Legal exactly when (active ally alive, at least one enemy alive) | Effect |
| ---: | --- | --- | --- | --- |
| 0–2 | `ATTACK_ENEMY_i` | Action | enemy `i` alive | One melee attack; advantage if the target is Prone. |
| 3–5 | `TRIP_ENEMY_i` | Bonus Action + Trip charge | enemy `i` alive and not Prone | Roll `d20 + attack_bonus` vs target AC with normal natural-1/natural-20 rules (no damage; a natural 20 has no extra effect). Success makes the target Prone. Trip rolls are always `NORMAL`. |
| 6 | `CLEAVE` | Action + Cleave | any enemy alive | M2 rule: one attack per living enemy, halved damage (min 1); each roll has advantage if that target is Prone. |
| 7 | `DODGE` | Action | ally not already Dodging | The ally is Dodging until the start of its next turn. |
| 8 | `SECOND_WIND` | Bonus Action + Second Wind | HP below max | Heal 1d10+2 (M0 rule). |
| 9 | `ACTION_SURGE` | Action Surge | — | Gain one Action (M0 rule). |
| 10 | `END_TURN` | — | always | End this ally's turn. |

Trip and Second Wind share the single Bonus Action, so a turn can use at most one of them, and each ally can Trip at most twice per encounter. Every ally attack roll and Trip roll against enemy `i` records that ally's slot as enemy `i`'s **last attacker**, regardless of the result. Masks are all-false unless the current actor is a living controlled ally; masked or invalid indices raise `ValueError`.

### Enemy turns

On an enemy's turn it acts once, then its turn ends (clearing Prone):

1. **Healer heal.** A Healer with `HEAL >= 1` heals if any living enemy (including itself) has `2 * hp <= max_hp`. It heals the eligible enemy with the most missing HP (`max_hp - hp`), ties to the lowest slot, for `2d6 + 2` capped at max HP, and spends one `HEAL`. Healing is not an attack and ignores Prone and Dodge. Otherwise the Healer attacks.
2. **Attack.** Brutes, Archers, and non-healing Healers make one attack against the living ally chosen by their targeting policy:
   - `WEAKEST`: lowest current HP; ties to the lowest slot.
   - `RETALIATE`: its last attacker if that ally is alive; otherwise the `WEAKEST` choice.
   - `RANDOM`: if two allies live, one uniform draw `np_random.integers(0, 2)` picks among living allies in slot order; with one living ally no draw is made.
   The attack roll has disadvantage if the attacker is Prone or the target is Dodging (M6 adds one more source below). Enemies never have advantage.

All enemy roles may attack any living ally (the Archer's attack is ranged; the Healer's attack works at any range). Enemy RNG order per turn: targeting draw (if any), attack d20(s), damage dice; or the heal dice.

### Turn order, lifecycle, outcome

Fixed order `ALLY 0, ALLY 1, ENEMY 0, ENEMY 1, ENEMY 2`; dead actors are skipped. At the start of an ally's turn: `ACTION = 1`, `BONUS_ACTION = 1`, Dodging ends. At the start of an enemy's turn: `ACTION = 1`. `ALLY 0`'s `END_TURN` hands to `ALLY 1`; `ALLY 1`'s `END_TURN` runs all living enemies in slot order, then the round advances. Victory (all enemies dead, any ally alive) gives +1; defeat (both allies dead) gives -1; otherwise reward is 0. If round 50 finishes without an outcome, the episode truncates with reward 0. Only terminal reward is supported — no shaping. `active_actor_index` and evaluation accounting match M3/M4.

### M5 observation

Unnormalized `float32` Box, 63 fields in this exact order:

```text
active_actor_index,
ally_{0,1}_hp, ally_{0,1}_action_count, ally_{0,1}_bonus_action_count,
ally_{0,1}_second_wind_count, ally_{0,1}_action_surge_count, ally_{0,1}_cleave_count,
ally_{0,1}_trip_count, ally_{0,1}_dodging,            (8 fields per ally, ally 0 first)
enemy_{0,1,2}_hp, _max_hp, _ac, _attack_bonus, _damage_die_size, _damage_bonus,
enemy_{j}_is_brute, _is_archer, _is_healer,
enemy_{j}_targets_weakest, _targets_retaliate, _targets_random,
enemy_{j}_prone, _heal_count, _last_attacker,         (15 fields per enemy, slot order)
round_number
```

`last_attacker` is 0 when no ally has attacked or tripped that enemy, otherwise ally slot + 1. Role and targeting are one-hot. Bounds: low is 0 everywhere except `round_number` (1). Highs: `active_actor_index` 1; ally HP 20, Action 2, `trip_count` 2, other ally counts and `dodging` 1; enemy HP and max HP 32, AC 16, attack bonus 8, die size 10, damage bonus 5, one-hot flags and `prone` 1, `heal_count` 3, `last_attacker` 2; `round_number` 50. Highs are derived from the stage's role profiles and charges.

Info reports `round`, `decision`, `action`, `target_index`, `enemy_hps`, `ally_hps`, `active_actor_index`, `enemy_roles`, `enemy_targeting`, `enemy_prone`, `ally_dodging`, `turn_order`, and the step's events: `fighter_attack` (roll mode and all d20s), `trip`, `cleave_attacks`, `healed`, and `enemy_turns` (one record or `None` per enemy slot: kind `attack` or `heal`, target, roll details).

## M6 — two-rank positioning

M6 is M5 plus a simple two-rank geometry (not a grid). Every M5 rule, roster, sampling range, enemy behavior, condition, reward, and truncation rule applies unless changed here. **All M6 rules are SIMULATOR CHOICES.**

**M6 sampling differences.** To offset the rank wall, the M6 Brute samples max HP 20–26 (M5: 24–32) and the M6 Healer starts with `HEAL = 2` charges (M5: 3). All other ranges, dice, and Trip charges match M5.

### Ranks, positions, and reach

- **Enemy ranks** are fixed by role: the Brute is in the **front** rank; the Archer and Healer are in the **back** rank. Enemies never move and ranks never change.
- **Ally positions**: each ally is at the **line** (initial position, facing the enemy front rank) or **deep** (moved past the front rank into the enemy back rank). Moving deep is one-way; there is no retreat.
- **Front broken**: no living enemy is in the front rank.
- **Reach**: an ally can make melee attacks (Attack, Trip, a Cleave roll) against a living enemy exactly when the enemy is in the front rank, or the ally is deep, or the front is broken. A deep ally reaches every enemy. Cleave attacks each living reachable enemy and is legal only with at least one reachable living enemy; its set of targets is fixed before its first roll, so killing the Brute with a Cleave roll does not add back-rank targets to that Cleave.
- **Engaged archer**: an Archer is engaged while any living ally is deep or the front is broken. An engaged Archer's attack rolls have disadvantage (combining with the M5 sources by the no-stacking rule).
- Enemies attack any living ally regardless of rank or position.

### New resources

- `MOVEMENT` (allies): 1 at reset, refreshed to 1 at the start of each ally turn.
- `REACTION` (enemies): 1 at reset, refreshed to 1 at the start of each enemy turn. Only opportunity attacks spend it.

### New actions and opportunity attacks

| Index | Decision | Cost | Legal exactly when | Effect |
| ---: | --- | --- | --- | --- |
| 10 | `ADVANCE` | Movement | ally at line; front not broken; a back-rank enemy alive | Unless the ally is Disengaged, each living front-rank enemy with `REACTION >= 1`, in slot order, spends its Reaction and makes one **opportunity attack** against the advancing ally (normal attack, damage, and roll-mode rules, including its Prone and the ally's Dodging; it ignores its targeting policy). Once the ally is at 0 HP no further opportunity attacks are made. If the ally survives, it becomes deep. |
| 11 | `DISENGAGE` | Action | ally at line; `MOVEMENT >= 1`; not already Disengaged; front not broken; a back-rank enemy alive | The ally is Disengaged until the end of its turn: its `ADVANCE` provokes no opportunity attacks. |
| 12 | `END_TURN` | — | always | As M5. |

Indices 0–9 are exactly M5's (`ATTACK_ENEMY_0..2`, `TRIP_ENEMY_0..2`, `CLEAVE`, `DODGE`, `SECOND_WIND`, `ACTION_SURGE`), with the reach rule added to Attack, Trip, and Cleave legality. Opportunity attacks do not change any enemy's last attacker. Disengaged ends at the end of the ally's turn.

**Death on an ally's own turn.** An opportunity attack can kill the active ally. Its turn then ends immediately and play continues exactly as if it had chosen `END_TURN`; if no ally survives, the episode ends in defeat at once.

### M6 lifecycle additions

At the start of an ally's turn `MOVEMENT = 1` and Disengaged ends (in addition to M5's refresh). At the start of an enemy's turn `REACTION = 1` (in addition to `ACTION = 1`).

### M6 observation

Unnormalized `float32` Box, 75 fields: M5's layout with three fields appended to each ally block and two to each enemy block:

```text
active_actor_index,
ally_{i}: hp, action_count, bonus_action_count, second_wind_count,
          action_surge_count, cleave_count, trip_count, dodging,
          movement_count, deep, disengaged              (11 fields per ally)
enemy_{j}: hp, max_hp, ac, attack_bonus, damage_die_size, damage_bonus,
           is_brute, is_archer, is_healer,
           targets_weakest, targets_retaliate, targets_random,
           prone, heal_count, last_attacker,
           back_rank, reaction_count                    (17 fields per enemy)
round_number
```

New highs are all 1; enemy HP highs follow the M6 profiles (26), and `heal_count` is at most 2. Info adds `ally_deep` and the `advance` event (its `opportunity_attacks` and whether the ally ended deep); Dodge and Disengage are recorded only through `decision`/`action` and the observation flags.

### Reconstruction

M5 and M6 observations contain every piece of state that affects future transitions at a live allied decision: HP, resources, conditions, last attackers, heal charges, positions, Disengaged, and enemy Reactions. Enemy `ACTION` is unobserved because it refreshes before the enemy acts. `combat.simulation.simulation_from_observation` restores both stages under the same live-decision contract as M0–M4.

### M5/M6 tuning history

Targets (set before measuring): a naive M4-style heuristic between roughly 35% and 65%, and a spread of at least 8–10 points between it and the best hand-written heuristic, without the best heuristic at the ceiling. The plan allowed one retune.

**v1** (first design, measured on held-out seeds 3000–7999, 5,000 episodes each): roles as in the M5 table but Brute +4–+6 / 1d10+2–4, Archer +5–+7 / 1d8+2–4, Healer heal 1d8+2 with 3 charges, the same ranges in M6, and Trip limited only by the Bonus Action. Results: M5 random 19.5%, naive 75.9%, best heuristic (Trip on the lowest-HP target) 94.0%; M6 random 11.9%, naive 44.8%, best (hold the line and kill the Brute first, with Trip) 84.6%, while diving was no better. Diagnosis: M5 was too easy, and almost all of the spread came from one free, nearly always-correct button (Trip every turn), which left the best policies at the ceiling rather than creating real choices.

**v2** (final; the single retune): (1) each ally gets only 2 Trip charges per encounter, which turns Trip from a free buff into an allocation decision; (2) the Healer heals 2d6+2 instead of 1d8+2, so its sustain matters more; (3) Brute and Archer gain +1 attack bonus and +1 damage bonus, to bring M5's naive baseline into the band; (4) in M6 only, Brute HP 20–26 and Healer 2 charges, to offset the rank wall for the naive baseline and to make "hold the line" and "dive the back rank" comparable on average, so the better choice depends on the encounter. The v2 values were selected from a small set of candidates evaluated with the hand heuristics on development seeds 20000–21999 (never the held-out range); the report lists the candidates. The `V1_RULES` constant in `combat/tactical_env.py` documents v1.
