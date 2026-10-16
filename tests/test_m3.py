"""Contracts for the M3 two-ally stage."""

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from agents.heuristic_agent import M3HeuristicAgent, make_heuristic_agent
from agents.random_agent import RandomAgent
from combat.actors import ActorRef, Side
from combat.stages import StagedCombatEnv, make_env
from evaluation.evaluate import evaluate


def fixed_rng(*rolls: int) -> Generator:
    values = iter(rolls)
    rng = Mock(spec=Generator)
    rng.integers.side_effect = lambda low, high: np.int64(next(values))
    return rng


def stats(env: StagedCombatEnv) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple((e.max_hp, e.armor_class, e.attack_bonus, e.damage.die_size, e.damage.bonus)
                 for e in env.enemies)


def test_m3_contracts() -> None:
    env = make_env("m3")
    assert isinstance(env, StagedCombatEnv)
    obs, info = env.reset(seed=2)
    assert env.action_names == ("ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "CLEAVE",
                                "SECOND_WIND", "ACTION_SURGE", "END_TURN")
    assert env.observation_fields == (
        "active_actor_index",
        "ally_0_hp", "ally_0_action_count", "ally_0_bonus_action_count",
        "ally_0_second_wind_count", "ally_0_action_surge_count", "ally_0_cleave_count",
        "ally_1_hp", "ally_1_action_count", "ally_1_bonus_action_count",
        "ally_1_second_wind_count", "ally_1_action_surge_count", "ally_1_cleave_count",
        "enemy_0_hp", "enemy_0_max_hp", "enemy_0_ac", "enemy_0_attack_bonus",
        "enemy_0_damage_die_size", "enemy_0_damage_bonus",
        "enemy_1_hp", "enemy_1_max_hp", "enemy_1_ac", "enemy_1_attack_bonus",
        "enemy_1_damage_die_size", "enemy_1_damage_bonus",
        "round_number",
    )
    assert env.action_space.n == 6 and len(obs) == 26
    assert env.observation_space.contains(obs)
    assert obs[0] == 0 and obs[1:7].tolist() == obs[7:13].tolist() == [20, 1, 1, 1, 1, 1]
    assert info["ally_hps"] == [20, 20] and info["active_actor_index"] == 0
    assert "fighter_hp" not in info  # multi-ally stages report ally_hps instead
    assert info["turn_order"] == ["ALLY/0", "ALLY/1", "ENEMY/0", "ENEMY/1"]
    assert env._controlled_refs == frozenset({ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1)})
    assert all(ally.hp == 20 for ally in env.allies)
    env.close()


def test_m3_seeded_enemies_match_m1b() -> None:
    m3, m1b = StagedCombatEnv("m3"), StagedCombatEnv("m1b")
    for seed in (2, 17):
        m3.reset(seed=seed)
        m1b.reset(seed=seed)
        assert stats(m3) == stats(m1b)
    m3.reset(seed=2)
    first = stats(m3)
    m3.reset(seed=2)
    assert stats(m3) == first
    m3.reset(seed=3)
    assert stats(m3) != first


def test_m3_seed2_two_ally_golden() -> None:
    env = StagedCombatEnv("m3")
    env.reset(seed=2)
    assert stats(env) == ((18, 13, 2, 4, 2), (18, 15, 2, 6, 3))

    _, _, _, _, info = env.step(0)  # A0 ATTACK enemy 0: 17 for 9
    assert info["fighter_attack"]["d20_roll"] == 17
    assert info["enemy_hps"] == [9, 18]
    env.step(4)  # A0 ACTION_SURGE
    _, _, term, _, info = env.step(0)  # A0 ATTACK enemy 0: crit kill
    assert info["fighter_attack"]["d20_roll"] == 20
    assert info["enemy_hps"] == [0, 18] and not term

    _, _, _, _, info = env.step(5)  # A0 END_TURN hands to A1, round stays 1
    assert info["round"] == 1 and info["active_actor_index"] == 1
    assert info["enemy_attacks"] == [None, None]

    _, _, _, _, info = env.step(2)  # A1 CLEAVE the one living enemy: miss
    assert info["cleave_living_targets"] == 1
    assert info["cleave_attacks"][0] is None
    assert info["cleave_attacks"][1]["d20_roll"] == 2
    env.step(4)  # A1 ACTION_SURGE
    _, _, _, _, info = env.step(1)  # A1 ATTACK enemy 1: 12 for 6
    assert info["fighter_attack"]["damage"] == 6
    assert info["enemy_hps"] == [0, 12]

    _, _, term, trunc, info = env.step(5)  # A1 END_TURN: enemy 1 misses, round 2
    assert info["enemy_attacks"] == [None, info["enemy_attacks"][1]]
    assert info["enemy_attacks"][1]["d20_roll"] == 5
    assert (info["ally_hps"], info["round"], info["active_actor_index"]) == ([20, 20], 2, 0)
    assert not term and not trunc


def test_m3_dead_ally_skipped_and_combat_continues() -> None:
    env = StagedCombatEnv("m3")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.step(5)  # A0 END_TURN, no enemy phase yet
    env.np_random = fixed_rng(15, 3, 1)  # E0 hits A0 for the kill, E1 misses A1

    _, _, term, _, info = env.step(5)  # A1 END_TURN runs the enemy phase
    assert info["enemy_attacks"][0]["damage"] == 1
    assert info["enemy_attacks"][1]["d20_roll"] == 1
    assert info["ally_hps"] == [0, 20] and not term
    assert info["round"] == 2 and info["active_actor_index"] == 1  # A0 skipped
    assert env.action_masks().tolist() == [True, True, True, False, True, True]

    env.np_random = fixed_rng(1, 1)
    _, _, term, _, info = env.step(5)  # A1 END_TURN: both enemies act, still round on
    assert all(attack is not None for attack in info["enemy_attacks"])
    assert info["ally_hps"] == [0, 20] and not term
    assert info["round"] == 3 and info["active_actor_index"] == 1


def test_m3_victory_with_dead_ally_and_defeat_needs_both() -> None:
    env = StagedCombatEnv("m3")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.step(5)
    env.np_random = fixed_rng(15, 3, 1)
    env.step(5)  # A0 dies in round 1; round 2 belongs to A1
    env.enemies[0].hp = 0
    env.enemies[1].hp = 1
    env.np_random = fixed_rng(20, 8, 8)
    _, reward, terminated, _, info = env.step(1)  # A1 kills the last enemy
    assert terminated and reward == 1.0
    assert info["ally_hps"] == [0, 20] and info["enemy_hps"] == [0, 0]
    assert not env.action_masks().any()

    env.reset(seed=2)
    env.allies[0].hp = 1
    env.allies[1].hp = 1
    env.step(5)
    env.np_random = fixed_rng(15, 3, 15, 3)  # each enemy kills one ally
    _, reward, terminated, _, info = env.step(5)
    assert terminated and reward == -1.0
    assert info["ally_hps"] == [0, 0]


def test_m3_damage_reward_rejected() -> None:
    with pytest.raises(ValueError, match="unavailable for m3"):
        StagedCombatEnv("m3", reward_mode="damage")


def test_m3_heuristic_acts_for_the_active_ally() -> None:
    base = [1.0, 20, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1,
            9, 12, 14, 4, 6, 2, 4, 12, 14, 4, 6, 2, 1.0]
    assert len(base) == 26
    obs = np.array(base, dtype=np.float32)
    agent = M3HeuristicAgent()
    assert agent.choose_action(obs, np.array([True, True, True, True, True, True])) == 3
    assert agent.choose_action(obs, np.array([True, True, True, False, True, True])) == 2
    assert agent.choose_action(obs, np.array([True, True, False, False, True, True])) == 1
    obs[0] = 0.0  # same board, healthy ALLY 0 acts: no heal, cleave instead
    assert agent.choose_action(obs, np.array([True, True, True, True, True, True])) == 2
    assert isinstance(make_heuristic_agent("m3"), M3HeuristicAgent)


def test_m3_evaluation_counts_two_ally_outcomes() -> None:
    summary = evaluate(make_heuristic_agent("m3"), [7], stage="m3")
    result = summary.results[0]
    assert result.won and not result.lost  # ALLY 0 dies, ALLY 1 wins the fight
    assert result.fighter_hp == 20 and result.rounds == 5
    for agent in (RandomAgent(seed=1), make_heuristic_agent("m3")):
        summary = evaluate(agent, range(5), stage="m3")
        assert summary.wins + summary.losses + summary.truncations == 5
