"""Environment contracts, legality, and seeded regressions for M5 and M6."""

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from combat.characters import Rank, Role, Targeting
from combat.env import MAX_ROUNDS
from combat.resources import Resource
from combat.stages import make_env
from combat.tactical_env import STAGE_RULES, TacticalCombatEnv
from combat.turns import TurnManager

M5_ACTIONS = (
    "ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "ATTACK_ENEMY_2",
    "TRIP_ENEMY_0", "TRIP_ENEMY_1", "TRIP_ENEMY_2",
    "CLEAVE", "DODGE", "SECOND_WIND", "ACTION_SURGE", "END_TURN",
)
M6_ACTIONS = M5_ACTIONS[:-1] + ("ADVANCE", "DISENGAGE", "END_TURN")
ALLY = ("hp", "action_count", "bonus_action_count", "second_wind_count",
        "action_surge_count", "cleave_count", "trip_count", "dodging")
ENEMY = ("hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus",
         "is_brute", "is_archer", "is_healer", "targets_weakest", "targets_retaliate",
         "targets_random", "prone", "heal_count", "last_attacker")


def fixed_rng(*rolls: int) -> Generator:
    values = iter(rolls)

    def integers(low: int, high: int) -> np.int64:
        result = next(values)
        assert low <= result < high, f"Roll {result} is outside [{low}, {high})"
        return np.int64(result)

    rng = Mock(spec=Generator)
    rng.integers.side_effect = integers
    return cast(Generator, rng)


def env_for(stage: str, seed: int) -> TacticalCombatEnv:
    env = make_env(stage)
    assert isinstance(env, TacticalCombatEnv)
    env.reset(seed=seed)
    return env


def step(env: TacticalCombatEnv, name: str):
    return env.step(env.action_names.index(name))


def legal(env: TacticalCombatEnv) -> set[str]:
    return {name for name, ok in zip(env.action_names, env.action_masks()) if ok}


@pytest.mark.parametrize(("stage", "actions", "ally_extra", "enemy_extra"), [
    ("m5", M5_ACTIONS, (), ()),
    ("m6", M6_ACTIONS, ("movement_count", "deep", "disengaged"), ("back_rank", "reaction_count")),
])
def test_contracts(stage, actions, ally_extra, enemy_extra) -> None:
    env = make_env(stage)
    obs, info = env.reset(seed=1)
    assert env.action_names == actions and env.action_space.n == len(actions)
    expected = (
        ("active_actor_index",)
        + tuple(f"ally_{s}_{f}" for s in range(2) for f in ALLY + ally_extra)
        + tuple(f"enemy_{s}_{f}" for s in range(3) for f in ENEMY + enemy_extra)
        + ("round_number",)
    )
    assert env.observation_fields == expected
    assert len(obs) == {"m5": 63, "m6": 75}[stage] and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    high = dict(zip(expected, env.observation_space.high))
    assert high["ally_0_trip_count"] == 2 and high["enemy_2_attack_bonus"] == 8
    assert high["enemy_0_max_hp"] == {"m5": 32, "m6": 26}[stage]
    assert high["enemy_1_heal_count"] == {"m5": 3, "m6": 2}[stage] and high["round_number"] == MAX_ROUNDS
    assert info["ally_hps"] == [20, 20] and info["active_actor_index"] == 0
    assert info["turn_order"] == ["ALLY/0", "ALLY/1", "ENEMY/0", "ENEMY/1", "ENEMY/2"]
    for key in ("enemy_roles", "enemy_targeting", "enemy_prone", "ally_dodging", "enemy_hps"):
        assert len(info[key]) in (2, 3)
    assert ("ally_deep" in info) == (stage == "m6")
    with pytest.raises(ValueError, match="unavailable"):
        TacticalCombatEnv(stage, reward_mode="damage")


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_sampling_follows_profiles_and_is_seeded(stage) -> None:
    env = make_env(stage)
    rules = STAGE_RULES[stage]
    seen_targeting: set[Targeting] = set()
    seen_slots: set[tuple[Role, int]] = set()
    for seed in range(300):
        obs, _ = env.reset(seed=seed)
        assert sorted(e.role.value for e in env.enemies) == [0, 1, 2]
        for slot, enemy in enumerate(env.enemies):
            profile = rules.profiles[enemy.role]
            assert profile.hp[0] <= enemy.max_hp == enemy.hp <= profile.hp[1]
            assert profile.ac[0] <= enemy.armor_class <= profile.ac[1]
            assert profile.attack[0] <= enemy.attack_bonus <= profile.attack[1]
            assert enemy.damage.die_size == profile.die and enemy.damage.dice_count == 1
            assert profile.bonus[0] <= enemy.damage.bonus <= profile.bonus[1]
            assert enemy.rank is (profile.rank if stage == "m6" else Rank.FRONT)
            assert enemy.resources.get(Resource.HEAL) == (rules.heal_charges if enemy.role is Role.HEALER else 0)
            seen_targeting.add(enemy.targeting)
            seen_slots.add((enemy.role, slot))
        assert all(a.resources.get(Resource.TRIP) == 2 for a in env.allies)
    assert seen_targeting == set(Targeting) and len(seen_slots) == 9
    first, _ = env.reset(seed=42)
    again, _ = env.reset(seed=42)
    assert np.array_equal(first, again)


def test_observation_encodes_roles_conditions_and_last_attacker() -> None:
    env = env_for("m5", 7)  # BRUTE, HEALER, ARCHER
    obs = env._observation()
    f = dict(zip(env.observation_fields, obs))
    assert (f["enemy_0_is_brute"], f["enemy_1_is_healer"], f["enemy_2_is_archer"]) == (1, 1, 1)
    assert (f["enemy_0_targets_weakest"], f["enemy_1_targets_random"], f["enemy_2_targets_random"]) == (1, 1, 1)
    assert f["enemy_1_heal_count"] == 3 and f["enemy_0_heal_count"] == 0
    env.np_random = fixed_rng(20)
    obs, *_ = step(env, "TRIP_ENEMY_2")
    f = dict(zip(env.observation_fields, obs))
    assert (f["enemy_2_prone"], f["enemy_2_last_attacker"], f["ally_0_trip_count"]) == (1, 1, 1)
    assert f["ally_0_bonus_action_count"] == 0
    step(env, "END_TURN")
    env.np_random = fixed_rng(2)
    obs, *_ = step(env, "ATTACK_ENEMY_0")
    assert dict(zip(env.observation_fields, obs))["enemy_0_last_attacker"] == 2


def test_m5_seed7_golden_sequence() -> None:
    env = env_for("m5", 7)
    stats = [(e.role.name, e.targeting.name, e.max_hp, e.armor_class, e.attack_bonus, e.damage.die_size, e.damage.bonus)
             for e in env.enemies]
    assert stats == [("BRUTE", "WEAKEST", 32, 15, 7, 10, 5), ("HEALER", "RANDOM", 12, 12, 2, 6, 2),
                     ("ARCHER", "RANDOM", 10, 13, 8, 8, 3)]
    *_, info = step(env, "TRIP_ENEMY_0")
    assert info["trip"] == {"d20_roll": 3, "total": 8, "success": False}
    *_, info = step(env, "ATTACK_ENEMY_0")
    assert info["fighter_attack"]["damage"] == 10 and info["enemy_hps"] == [22, 12, 10]
    step(env, "ACTION_SURGE")
    *_, info = step(env, "ATTACK_ENEMY_0")
    assert info["fighter_attack"]["d20_roll"] == 7 and not info["fighter_attack"]["hit"]
    *_, info = step(env, "END_TURN")
    assert info["active_actor_index"] == 1 and info["enemy_turns"] == [None, None, None]
    *_, info = step(env, "CLEAVE")
    assert [a["damage"] for a in info["cleave_attacks"]] == [0, 0, 3]
    *_, info = step(env, "TRIP_ENEMY_1")
    assert info["trip"]["success"] and info["enemy_prone"] == [False, True, False]
    *_, info = step(env, "END_TURN")
    turns = info["enemy_turns"]
    assert [(t["kind"], t["target_slot"]) for t in turns] == [("attack", 0), ("attack", 1), ("attack", 1)]
    assert turns[1]["attack"]["mode"] == "disadvantage" and turns[1]["attack"]["d20_rolls"] == (12, 12)
    assert turns[2]["attack"]["critical"] and turns[2]["attack"]["damage"] == 17
    assert info["ally_hps"] == [10, 3] and info["round"] == 2
    assert info["enemy_prone"] == [False, False, False]  # stood up at the end of its turn
    *_, info = step(env, "DODGE")
    assert info["ally_dodging"] == [True, False]
    step(env, "END_TURN")
    *_, info = step(env, "ATTACK_ENEMY_1")
    assert info["enemy_hps"] == [22, 4, 7]
    obs, reward, terminated, truncated, info = step(env, "END_TURN")
    turns = info["enemy_turns"]
    assert (turns[0]["target_slot"], turns[0]["attack"]["hit"]) == (1, False)  # WEAKEST -> 3 HP ally
    assert (turns[1]["kind"], turns[1]["target_slot"], turns[1]["healed"]) == ("heal", 1, 8)
    assert turns[2]["target_slot"] == 0 and turns[2]["attack"]["mode"] == "disadvantage"  # dodging ally
    assert info["enemy_hps"] == [22, 12, 7] and info["ally_dodging"] == [False, False]
    assert (reward, terminated, truncated, info["round"]) == (0.0, False, False, 3)


def test_m6_seed5_golden_sequence() -> None:
    env = env_for("m6", 5)
    assert [e.role.name for e in env.enemies] == ["HEALER", "BRUTE", "ARCHER"]
    assert [e.rank for e in env.enemies] == [Rank.BACK, Rank.FRONT, Rank.BACK]
    assert legal(env) == {"ATTACK_ENEMY_1", "TRIP_ENEMY_1", "CLEAVE", "DODGE", "ACTION_SURGE",
                          "ADVANCE", "DISENGAGE", "END_TURN"}
    *_, info = step(env, "ADVANCE")
    oas = info["advance"]["opportunity_attacks"]
    assert [(oa["enemy_slot"], oa["attack"]["d20_roll"]) for oa in oas] == [(1, 1)]
    assert info["ally_deep"] == [True, False]
    assert {"ATTACK_ENEMY_0", "ATTACK_ENEMY_2", "TRIP_ENEMY_0"} <= legal(env)
    assert "ADVANCE" not in legal(env) and "DISENGAGE" not in legal(env)
    step(env, "TRIP_ENEMY_0")
    *_, info = step(env, "ATTACK_ENEMY_0")
    assert info["fighter_attack"]["critical"] and info["enemy_hps"] == [1, 22, 13]
    step(env, "END_TURN")
    obs, *_ = step(env, "DISENGAGE")
    assert dict(zip(env.observation_fields, obs))["ally_1_disengaged"] == 1
    *_, info = step(env, "ADVANCE")  # The Brute's Reaction was already spent on ally 0.
    assert info["advance"]["opportunity_attacks"] == [] and env.enemies[1].resources.get(Resource.REACTION) == 0
    step(env, "ACTION_SURGE")
    step(env, "ATTACK_ENEMY_2")
    obs, _, _, _, info = step(env, "END_TURN")
    turns = info["enemy_turns"]
    assert (turns[0]["kind"], turns[0]["target_slot"], turns[0]["healed"]) == ("heal", 0, 7)
    assert turns[1]["target_slot"] == 0 and turns[1]["attack"]["mode"] == "normal"
    assert turns[2]["attack"]["mode"] == "disadvantage"  # engaged Archer
    assert turns[2]["attack"]["d20_roll"] == 4
    assert info["enemy_hps"] == [8, 22, 8] and info["round"] == 2
    f = dict(zip(env.observation_fields, obs))
    assert f["ally_1_disengaged"] == 0 and f["ally_0_movement_count"] == 1


def test_m5_legality_rules() -> None:
    env = env_for("m5", 7)
    assert legal(env) == {"ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "ATTACK_ENEMY_2", "TRIP_ENEMY_0",
                          "TRIP_ENEMY_1", "TRIP_ENEMY_2", "CLEAVE", "DODGE", "ACTION_SURGE", "END_TURN"}
    env.allies[0].hp = 15
    assert "SECOND_WIND" in legal(env)
    env.enemies[2].prone = True
    env.enemies[1].hp = 0
    names = legal(env)
    assert "TRIP_ENEMY_2" not in names and "TRIP_ENEMY_1" not in names and "ATTACK_ENEMY_1" not in names
    env.np_random = fixed_rng(10)
    step(env, "TRIP_ENEMY_0")  # spends the Bonus Action shared with Second Wind
    assert not {"SECOND_WIND", "TRIP_ENEMY_0"} & legal(env)
    step(env, "DODGE")
    assert not {"DODGE", "ATTACK_ENEMY_0", "CLEAVE"} & legal(env)
    step(env, "ACTION_SURGE")
    assert "DODGE" not in legal(env) and "ATTACK_ENEMY_0" in legal(env)
    env.allies[0].resources.set(Resource.TRIP, 0)
    env.allies[0].resources.set(Resource.BONUS_ACTION, 1)
    assert not any(name.startswith("TRIP") for name in legal(env))


def test_invalid_and_post_terminal_steps() -> None:
    env = env_for("m5", 7)
    with pytest.raises(ValueError, match="Invalid"):
        env.step(11)
    env.enemies[0].prone = True
    with pytest.raises(ValueError, match="TRIP_ENEMY_0 is not legal"):
        step(env, "TRIP_ENEMY_0")
    for enemy in env.enemies:
        enemy.hp = 1
    env.enemies[1].hp = env.enemies[2].hp = 0
    env.np_random = fixed_rng(1, 19, 1)  # Prone target: advantage keeps 19
    _, reward, terminated, _, info = step(env, "ATTACK_ENEMY_0")
    assert info["fighter_attack"]["d20_rolls"] == (1, 19)
    assert terminated and reward == 1.0 and not env.action_masks().any()
    with pytest.raises(RuntimeError):
        step(env, "END_TURN")


def test_defeat_requires_both_allies_and_enemy_phase_stops() -> None:
    env = env_for("m5", 7)
    env.allies[0].hp = 1
    step(env, "END_TURN")
    env.allies[1].hp = 1
    # Brute (WEAKEST, tie -> slot 0) kills ally 0; the RANDOM Healer then has one
    # living target, so it makes no targeting draw, and kills ally 1.
    env.np_random = fixed_rng(19, 1, 19, 1)
    _, reward, terminated, _, info = step(env, "END_TURN")
    assert terminated and reward == -1.0 and info["ally_hps"] == [0, 0]
    assert info["enemy_turns"][2] is None
    assert not env.action_masks().any()


def test_round_limit_truncates_without_wrapping() -> None:
    env = env_for("m5", 7)
    env.round_number = MAX_ROUNDS
    env.turns = TurnManager(env.turns.order, MAX_ROUNDS, current=env.turns.current)
    step(env, "END_TURN")
    env.np_random = fixed_rng(1, 0, 1, 0, 1)  # every enemy misses (RANDOM draws included)
    _, reward, terminated, truncated, info = step(env, "END_TURN")
    assert truncated and not terminated and reward == 0.0
    assert info["round"] == MAX_ROUNDS and env.turns.round_number == MAX_ROUNDS
    assert not env.action_masks().any()


def test_m6_opportunity_attack_kill_passes_the_turn_or_ends_combat() -> None:
    env = env_for("m6", 4)  # BRUTE in front (slot 0)
    assert env.enemies[0].role is Role.BRUTE
    env.allies[0].hp = 3
    env.np_random = fixed_rng(15, 6)
    _, reward, terminated, _, info = step(env, "ADVANCE")
    assert info["ally_hps"][0] == 0 and not info["ally_deep"][0]
    assert not terminated and reward == 0.0
    assert info["active_actor_index"] == 1 and info["enemy_turns"] == [None, None, None]
    assert "ATTACK_ENEMY_0" in legal(env)

    env = env_for("m6", 4)
    env.allies[1].hp = 0
    env.allies[0].hp = 3
    env.np_random = fixed_rng(15, 6)
    _, reward, terminated, _, info = step(env, "ADVANCE")
    assert terminated and reward == -1.0 and "enemy_turns" not in info


def test_m6_last_ally_oa_death_runs_enemy_phase() -> None:
    env = env_for("m6", 4)
    step(env, "END_TURN")  # ally 1 acts next
    env.allies[1].hp = 2
    env.np_random = fixed_rng(15, 1, 1, 1, 1)
    # OA kills ally 1; its turn passes to the enemy phase: the Brute already reacted
    # this round but still acts; Archer and Healer follow. With one living ally no
    # RANDOM draw is made, so every remaining roll is a missed d20.
    _, reward, terminated, _, info = step(env, "ADVANCE")
    assert info["ally_hps"][1] == 0 and not terminated
    assert info["round"] == 2 and info["active_actor_index"] == 0
    assert all(turn is not None for turn in info["enemy_turns"])
    assert env.enemies[0].resources.get(Resource.REACTION) == 1  # refreshed on its turn


def test_m6_front_broken_exposes_back_rank_and_disables_movement() -> None:
    env = env_for("m6", 4)
    assert "ATTACK_ENEMY_1" not in legal(env)
    env.enemies[0].hp = 0
    names = legal(env)
    assert {"ATTACK_ENEMY_1", "ATTACK_ENEMY_2", "TRIP_ENEMY_1"} <= names
    assert not {"ADVANCE", "DISENGAGE", "ATTACK_ENEMY_0"} & names
