"""Gymnasium contracts for the one-decision-per-step M0 environment."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from gymnasium import spaces

from combat.actions import Action
from combat.damage import DamageSpec
from combat.env import (
    MAX_ROUNDS,
    M0_ACTIONS,
    OBSERVATION_FIELDS,
    BaldurCombatEnv,
    RewardSnapshot,
)
from combat.resources import Resource


def test_reset_spaces_and_explicit_ordering() -> None:
    env = BaldurCombatEnv()
    observation, info = env.reset(seed=7)

    assert M0_ACTIONS == (Action.ATTACK, Action.SECOND_WIND, Action.ACTION_SURGE, Action.END_TURN)
    assert OBSERVATION_FIELDS == (
        "fighter_hp", "fighter_action_count", "fighter_bonus_action_count",
        "fighter_second_wind_count", "fighter_action_surge_count", "goblin_hp", "round_number",
    )
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 4
    assert isinstance(env.observation_space, spaces.Box)
    assert observation.shape == (7,) and observation.dtype == np.float32
    np.testing.assert_array_equal(observation, [20, 1, 1, 1, 1, 15, 1])
    assert env.observation_space.contains(observation)
    assert info == {"round": 1, "action": None, "fighter_hp": 20, "goblin_hp": 15}


def test_mask_and_each_fighter_decision_is_one_step() -> None:
    env = BaldurCombatEnv()
    env.reset(seed=0)
    np.testing.assert_array_equal(env.action_masks(), [True, False, True, True])

    observation, reward, terminated, truncated, info = env.step(0)
    assert info["action"] == "ATTACK" and "fighter_attack" in info
    assert observation[0] == 20 and observation[5] < 15  # No automatic Goblin turn.
    assert observation[1] == 0 and observation[6] == 1
    assert reward == 0.0 and not terminated and not truncated
    np.testing.assert_array_equal(env.action_masks(), [False, False, True, True])

    observation, _, _, _, _ = env.step(2)
    assert observation[1] == 1 and observation[4] == 0
    np.testing.assert_array_equal(env.action_masks(), [True, False, False, True])

    env.fighter.hp = 10
    observation, _, _, _, info = env.step(1)
    assert info["action"] == "SECOND_WIND" and info["healed"] > 0
    assert observation[0] > 10 and observation[5] < 15 and observation[6] == 1
    assert observation[2] == 0 and observation[3] == 0


def test_end_turn_runs_one_goblin_attack_and_refreshes_per_turn_resources() -> None:
    env = BaldurCombatEnv()
    env.reset(seed=0)
    env.fighter.hp = 10
    env.step(1)  # Spend Second Wind and Bonus Action.
    env.fighter.resources.set(Resource.ACTION, 0)
    before_hp = env.fighter.hp

    observation, reward, terminated, truncated, info = env.step(3)

    assert info["action"] == "END_TURN" and "goblin_attack" in info
    assert before_hp - env.fighter.hp == info["goblin_attack"]["damage"]
    assert env.goblin.resources.get(Resource.ACTION) == 0
    assert (terminated, truncated, reward) == (False, False, 0.0)
    assert info["round"] == env.round_number == 2
    np.testing.assert_array_equal(observation[[1, 2, 3, 4, 6]], [1, 1, 0, 1, 2])


@pytest.mark.parametrize(
    ("action", "expected_reward", "loser"),
    [(0, 1.0, "goblin"), (3, -1.0, "fighter")],
)
def test_death_terminates_immediately_with_terminal_reward(
    action: int, expected_reward: float, loser: str
) -> None:
    env = BaldurCombatEnv()
    env.reset(seed=0)
    if loser == "goblin":
        env.goblin.hp = 1
    else:
        env.fighter.hp = 1

    observation, reward, terminated, truncated, info = env.step(action)

    assert terminated and not truncated and reward == expected_reward
    assert info[f"{loser}_hp"] == 0
    assert env.observation_space.contains(observation)
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(3)


@pytest.mark.parametrize("invalid_action", [-1, 1, 4])
def test_illegal_action_raises_without_transition(invalid_action: int) -> None:
    env = BaldurCombatEnv()
    before, _ = env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(invalid_action)
    np.testing.assert_array_equal(env._get_observation(), before)

    if invalid_action == 1:
        env.step(0)
        with pytest.raises(ValueError, match="ATTACK"):
            env.step(0)


def test_seeded_action_sequence_is_reproducible() -> None:
    first, second = BaldurCombatEnv(), BaldurCombatEnv()
    np.testing.assert_array_equal(first.reset(seed=13)[0], second.reset(seed=13)[0])
    for action in (0, 3, 0):
        first_result, second_result = first.step(action), second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]
        if first_result[2] or first_result[3]:
            break


def test_reward_callable_receives_immutable_before_and_after_snapshots() -> None:
    seen: list[tuple[RewardSnapshot, RewardSnapshot, bool, bool]] = []

    def reward_fn(before: RewardSnapshot, after: RewardSnapshot, terminated: bool, truncated: bool) -> float:
        seen.append((before, after, terminated, truncated))
        return 0.25

    env = BaldurCombatEnv(reward_fn=reward_fn)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert reward == 0.25
    before, after, terminated, truncated = seen[0]
    assert before.goblin_hp == 15 and after.goblin_hp < before.goblin_hp
    assert before.fighter_resources[0] == 1 and after.fighter_resources[0] == 0
    assert not terminated and not truncated
    with pytest.raises(FrozenInstanceError):
        before.goblin_hp = 0


def test_round_fifty_truncates_without_round_fifty_one() -> None:
    env = BaldurCombatEnv()
    env.reset(seed=0)
    env.goblin.damage = DamageSpec(1, 6, -12)  # Even critical hits deal zero.

    for completed_round in range(1, MAX_ROUNDS + 1):
        observation, reward, terminated, truncated, _ = env.step(3)
        assert not terminated and reward == 0.0
        assert truncated is (completed_round == MAX_ROUNDS)
        assert env.round_number == min(completed_round + 1, MAX_ROUNDS)
        assert env.observation_space.contains(observation)

    assert env.round_number == MAX_ROUNDS
    with pytest.raises(RuntimeError, match="reset"):
        env.step(3)
