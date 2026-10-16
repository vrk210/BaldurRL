"""Contracts for randomized and two-target combat stages."""

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from combat.actions import Action
from combat.characters import Character
from combat.env import MAX_ROUNDS
from combat.mechanics import AttackResult, use_attack, use_cleave
from combat.resources import Resource
from combat.stages import StagedCombatEnv, make_env


def fixed_rng(*rolls: int) -> Generator:
    values = iter(rolls)
    rng = Mock(spec=Generator)
    rng.integers.side_effect = lambda low, high: np.int64(next(values))
    return rng


def stats(env: StagedCombatEnv) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple((enemy.max_hp, enemy.armor_class, enemy.attack_bonus,
                  enemy.damage.die_size, enemy.damage.bonus) for enemy in env.enemies)


def test_stage_metadata_and_seeded_generation() -> None:
    for stage, actions, fields in (("m1a", 4, 12), ("m1b", 5, 18), ("m2", 6, 19)):
        env = make_env(stage)
        assert isinstance(env, StagedCombatEnv)
        obs, _ = env.reset(seed=17)
        first = stats(env)
        assert obs.dtype == np.float32
        assert len(obs) == len(env.observation_fields) == fields
        assert len(env.decisions) == len(env.action_names) == env.action_space.n == actions
        assert env.observation_space.contains(obs)
        env.reset(seed=17)
        assert stats(env) == first
        env.reset(seed=18)
        assert stats(env) != first
        for hp, ac, attack, die, bonus in stats(env):
            assert 8 <= hp <= 20 and 12 <= ac <= 18 and 2 <= attack <= 6
            assert die in (4, 6, 8) and 0 <= bonus <= 4
        assert tuple(env.action_names) == tuple(decision.label for decision in env.decisions)
        env.close()


def test_m1a_observation_matches_enemy() -> None:
    env = StagedCombatEnv("m1a")
    obs, _ = env.reset(seed=23)
    enemy = env.enemies[0]
    assert obs.tolist()[5:11] == [enemy.hp, enemy.max_hp, enemy.armor_class,
                                 enemy.attack_bonus, enemy.damage.die_size, enemy.damage.bonus]


def test_targeting_mask_and_victory_require_both_deaths() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    env.np_random = fixed_rng(20, 8, 8)
    env.enemies[0].hp = 1
    _, _, terminated, _, info = env.step(0)
    assert terminated is False
    assert info["enemy_hps"][0] == 0
    assert info["enemy_hps"][1] > 0
    assert env.action_masks().tolist() == [False, False, False, True, True]
    env.fighter.resources.set(Resource.ACTION, 1)
    assert env.action_masks()[1] and not env.action_masks()[0]
    env.np_random = fixed_rng(20, 8, 8)
    env.enemies[1].hp = 1
    _, reward, terminated, _, _ = env.step(1)
    assert terminated and reward == 1


def test_enemy_turn_stable_order_and_dead_skip() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    env.np_random = fixed_rng(1, 1)
    _, _, _, _, info = env.step(4)
    assert [attack["d20_roll"] for attack in info["enemy_attacks"]] == [1, 1]
    assert info["round"] == 2
    env.enemies[0].hp = 0
    env.np_random = fixed_rng(1)
    _, _, _, _, info = env.step(4)
    assert info["enemy_attacks"][0] is None
    assert info["enemy_attacks"][1]["d20_roll"] == 1


def test_second_enemy_does_not_attack_after_first_kills_fighter() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.np_random = fixed_rng(20, 8, 8)
    _, reward, terminated, truncated, info = env.step(4)
    assert terminated and not truncated and reward == -1
    assert info["enemy_attacks"][0]["damage"] == 1
    assert info["enemy_attacks"][1] is None


def test_cleave_half_damage_critical_and_resource_spend() -> None:
    env = StagedCombatEnv("m2")
    env.reset(seed=2)
    fighter = env.fighter
    assert fighter is not None
    for enemy in env.enemies:
        enemy.armor_class = 1
    # First attack: normal 5 + 3 = 8 -> 4. Second: critical 4 + 6 + 3 = 13 -> 6.
    env.np_random = fixed_rng(10, 5, 20, 4, 6)
    initial = [enemy.hp for enemy in env.enemies]
    _, _, _, _, info = env.step(2)
    assert [initial[i] - enemy.hp for i, enemy in enumerate(env.enemies)] == [4, 6]
    assert info["cleave_living_targets"] == 2
    assert fighter.resources.get(Resource.ACTION) == 0
    assert fighter.resources.get(Resource.CLEAVE) == 0
    assert not env.action_masks()[2]


def test_cleave_skips_dead_target_and_minimum_hit_damage() -> None:
    env = StagedCombatEnv("m2")
    env.reset(seed=4)
    fighter = env.fighter
    assert fighter is not None
    env.enemies[0].hp = 0
    env.enemies[1].armor_class = 1
    fighter.damage = type(fighter.damage)(1, 8, 0)
    results = use_cleave(fighter, env.enemies, fixed_rng(10, 1))
    assert results[0] is None
    assert results[1] is not None and results[1].damage == 1
    assert fighter.resources.get(Resource.CLEAVE) == 0
    with pytest.raises(ValueError, match="CLEAVE is not legal"):
        use_cleave(fighter, env.enemies, fixed_rng())


def test_damage_reward_only_tracks_fighter_damage() -> None:
    env = StagedCombatEnv("m2", reward_mode="damage")
    env.reset(seed=8)
    env.np_random = fixed_rng(1, 1)
    _, reward, _, _, _ = env.step(5)
    assert reward == 0
    env.np_random = fixed_rng(20, 8, 8)
    _, reward, _, _, _ = env.step(0)
    assert 0 < reward <= 0.2


@pytest.mark.parametrize("stage", ["m1a", "m1b", "m2"])
@pytest.mark.parametrize("outcome", ["victory", "defeat", "ongoing", "truncation"])
def test_single_ally_terminal_rewards(stage: str, outcome: str) -> None:
    env = StagedCombatEnv(stage)
    env.reset(seed=2)
    assert env.fighter is not None
    if outcome == "victory":
        for enemy in env.enemies[1:]:
            enemy.hp = 0
        env.enemies[0].hp = 1
        env.np_random = fixed_rng(20, 8, 8)
        action = 0
    else:
        action = env.action_names.index("END_TURN")
        if outcome == "defeat":
            env.fighter.hp = 1
            env.np_random = fixed_rng(20, 8, 8)
        else:
            env.np_random = fixed_rng(*([1] * len(env.enemies)))
            if outcome == "truncation":
                for _ in range(MAX_ROUNDS - 1):
                    env.np_random = fixed_rng(*([1] * len(env.enemies)))
                    env.step(action)
                env.np_random = fixed_rng(*([1] * len(env.enemies)))

    _, reward, terminated, truncated, _ = env.step(action)
    assert reward == {"victory": 1.0, "defeat": -1.0}.get(outcome, 0.0)
    assert terminated is (outcome in ("victory", "defeat"))
    assert truncated is (outcome == "truncation")


@pytest.mark.parametrize("stage", ["m1a", "m1b", "m2", "m3", "m4"])
def test_injected_simultaneous_defeat_is_a_loss(stage: str, monkeypatch) -> None:
    env = StagedCombatEnv(stage)
    env.reset(seed=2)
    for enemy in env.enemies[1:]:
        enemy.hp = 0
    env.enemies[0].hp = 1
    env.np_random = fixed_rng(20, 8, 8)

    # Inject mutual defeat without adding a new mechanic to the simulator.
    def mutual_defeat(attacker: Character, defender: Character, rng: Generator) -> AttackResult:
        result = use_attack(attacker, defender, rng)
        for ally in env.allies:
            ally.hp = 0
        return result

    monkeypatch.setattr("combat.stages.use_attack", mutual_defeat)
    _, reward, terminated, truncated, _ = env.step(0)
    assert all(ally.hp == 0 for ally in env.allies)
    assert all(enemy.hp == 0 for enemy in env.enemies)
    assert (reward, terminated, truncated) == (-1.0, True, False)
    assert not env.action_masks().any()


def test_m2_damage_shaping_is_added_to_terminal_victory() -> None:
    env = StagedCombatEnv("m2", reward_mode="damage")
    env.reset(seed=2)
    initial_hp = sum(enemy.max_hp for enemy in env.enemies)
    for enemy in env.enemies:
        enemy.hp = 1
    env.np_random = fixed_rng(20, 8, 8, 20, 8, 8)

    _, reward, terminated, truncated, _ = env.step(2)  # Cleave kills both.
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(1.0 + 0.2 * 2 / initial_hp)


def test_m2_damage_shaping_does_not_change_terminal_defeat() -> None:
    env = StagedCombatEnv("m2", reward_mode="damage")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.np_random = fixed_rng(20, 8, 8)

    _, reward, terminated, truncated, _ = env.step(5)
    assert (reward, terminated, truncated) == (-1.0, True, False)
