"""Contracts for the M4 two-ally, three-enemy stage."""

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from agents.heuristic_agent import M4HeuristicAgent, make_heuristic_agent
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


def test_m4_contracts() -> None:
    env = make_env("m4")
    assert isinstance(env, StagedCombatEnv)
    obs, info = env.reset(seed=2)
    assert env.action_names == ("ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "ATTACK_ENEMY_2",
                                "CLEAVE", "SECOND_WIND", "ACTION_SURGE", "END_TURN")
    assert env.observation_fields == (
        ("active_actor_index",)
        + tuple(f"ally_{slot}_{f}" for slot in range(2) for f in (
            "hp", "action_count", "bonus_action_count",
            "second_wind_count", "action_surge_count", "cleave_count"))
        + tuple(f"enemy_{slot}_{f}" for slot in range(3) for f in (
            "hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus"))
        + ("round_number",)
    )
    assert env.action_space.n == 7 and len(obs) == 32
    assert env.observation_space.contains(obs)
    assert obs[0] == 0 and obs[-1] == 1
    assert info["ally_hps"] == [20, 20] and info["active_actor_index"] == 0
    assert "fighter_hp" not in info
    assert info["turn_order"] == ["ALLY/0", "ALLY/1", "ENEMY/0", "ENEMY/1", "ENEMY/2"]
    assert env._controlled_refs == frozenset({ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1)})
    env.close()


def test_m4_scaled_enemy_generation() -> None:
    m4 = StagedCombatEnv("m4")
    m4.reset(seed=2)
    assert stats(m4) == ((22, 13, 3, 4, 4), (22, 15, 3, 6, 5), (22, 17, 7, 4, 6))
    for hp, ac, attack, die, bonus in stats(m4):
        assert 12 <= hp <= 24 and 12 <= ac <= 18 and 3 <= attack <= 7
        assert die in (4, 6, 8) and 2 <= bonus <= 6
    first = stats(m4)
    m4.reset(seed=2)
    assert stats(m4) == first
    m4.reset(seed=3)
    assert stats(m4) != first
    m1b = StagedCombatEnv("m1b")
    m1b.reset(seed=2)
    assert stats(m1b) != first[:2]  # scaled profile differs from the M1B one


def test_m4_seed2_two_ally_golden() -> None:
    env = StagedCombatEnv("m4")
    env.reset(seed=2)

    _, _, _, _, info = env.step(0)  # A0 ATTACK enemy 0: miss
    assert info["fighter_attack"]["d20_roll"] == 2
    env.step(5)  # A0 ACTION_SURGE
    _, _, _, _, info = env.step(0)  # A0 ATTACK enemy 0: 12 for 6
    assert info["fighter_attack"]["damage"] == 6
    assert info["enemy_hps"] == [16, 22, 22]

    _, _, _, _, info = env.step(6)  # A0 END_TURN hands to A1
    assert info["round"] == 1 and info["active_actor_index"] == 1
    assert info["enemy_attacks"] == [None, None, None]

    _, _, _, _, info = env.step(3)  # A1 CLEAVE all three: miss, 3, 3
    assert info["cleave_living_targets"] == 3
    assert [a["damage"] if a else None for a in info["cleave_attacks"]] == [0, 3, 3]
    assert info["enemy_hps"] == [16, 19, 19]

    _, _, term, trunc, info = env.step(6)  # A1 END_TURN: enemies hit A0 twice
    assert [a["d20_roll"] if a else None for a in info["enemy_attacks"]] == [4, 15, 14]
    assert info["ally_hps"] == [3, 20]
    assert (info["round"], info["active_actor_index"]) == (2, 0)
    assert not term and not trunc


def test_m4_dead_ally_skipped_and_enemies_retarget() -> None:
    env = StagedCombatEnv("m4")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.step(6)  # A0 END_TURN
    env.np_random = fixed_rng(13, 1, 1, 1)  # E0 kills A0, E1/E2 miss A1

    _, _, term, _, info = env.step(6)  # A1 END_TURN runs the enemy phase
    assert info["enemy_attacks"][0]["damage"] == 1
    assert all(attack is not None for attack in info["enemy_attacks"])
    assert info["ally_hps"] == [0, 20] and not term
    assert info["round"] == 2 and info["active_actor_index"] == 1
    assert env.action_masks()[0] and env.action_masks()[6]


def test_m4_victory_with_dead_ally_and_defeat_needs_both() -> None:
    env = StagedCombatEnv("m4")
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.step(6)
    env.np_random = fixed_rng(13, 1, 1, 1)
    env.step(6)  # A0 dies in round 1; round 2 belongs to A1
    env.enemies[0].hp = 0
    env.enemies[1].hp = 0
    env.enemies[2].hp = 1
    env.np_random = fixed_rng(20, 8, 8)
    _, reward, terminated, _, info = env.step(2)  # A1 kills the last enemy
    assert terminated and reward == 1.0
    assert info["ally_hps"] == [0, 20] and info["enemy_hps"] == [0, 0, 0]

    env.reset(seed=2)
    env.allies[0].hp = 1
    env.allies[1].hp = 1
    env.step(6)
    env.np_random = fixed_rng(13, 1, 13, 1)  # E0 kills A0, E1 kills A1, E2 never acts
    _, reward, terminated, _, info = env.step(6)
    assert terminated and reward == -1.0
    assert info["ally_hps"] == [0, 0]
    assert info["enemy_attacks"][2] is None


def test_m4_damage_reward_rejected() -> None:
    with pytest.raises(ValueError, match="unavailable for m4"):
        StagedCombatEnv("m4", reward_mode="damage")


def test_m4_heuristic_acts_for_the_active_ally() -> None:
    base = [1.0, 20, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1,
            9, 22, 14, 4, 6, 2, 4, 22, 14, 4, 6, 2, 0, 22, 14, 4, 6, 2, 1.0]
    assert len(base) == 32
    obs = np.array(base, dtype=np.float32)
    agent = M4HeuristicAgent()
    full = np.array([True] * 7)
    assert agent.choose_action(obs, full) == 4  # hurt active ally heals
    assert agent.choose_action(obs, np.array([True, True, False, True, False, True, True])) == 3
    assert agent.choose_action(obs, np.array([True, True, False, False, False, True, True])) == 1
    single = obs.copy()
    single[19] = 0.0  # only enemy 0 still lives: attack it, don't cleave
    assert agent.choose_action(single, np.array([True, False, False, True, False, True, True])) == 0
    assert isinstance(make_heuristic_agent("m4"), M4HeuristicAgent)


def test_m4_evaluation_completes() -> None:
    for agent in (RandomAgent(seed=1), make_heuristic_agent("m4")):
        summary = evaluate(agent, range(5), stage="m4")
        assert summary.wins + summary.losses + summary.truncations == 5
        assert len(summary.action_counts) == 7
