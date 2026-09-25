"""Random policy and seeded evaluation contracts."""

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from combat.actions import Action
from combat.env import M0_ACTIONS
from evaluation.evaluate import evaluate


def test_random_agent_selects_uniformly_among_legal_indices() -> None:
    agent = RandomAgent(seed=3)
    observation = np.zeros(7, dtype=np.float32)
    mask = np.array([False, True, False, True])
    choices = [agent.choose_action(observation, mask) for _ in range(1_000)]

    assert set(choices) == {1, 3}
    assert 400 < choices.count(1) < 600
    with pytest.raises(ValueError, match="No legal actions"):
        agent.choose_action(observation, np.zeros(4, dtype=bool))


def test_random_agent_repeats_action_sequence_for_same_seed_and_masks() -> None:
    first, second = RandomAgent(seed=17), RandomAgent(seed=17)
    observation = np.zeros(7, dtype=np.float32)
    masks = [
        np.array([True, False, True, True]),
        np.array([False, False, True, True]),
        np.array([True, True, False, True]),
    ] * 20
    assert [first.choose_action(observation, mask) for mask in masks] == [
        second.choose_action(observation, mask) for mask in masks
    ]


def test_evaluation_completes_and_metrics_are_consistent() -> None:
    summary = evaluate(RandomAgent(seed=7), range(20))

    assert summary.episodes == 20
    assert summary.wins + summary.losses + summary.truncations == summary.episodes
    assert 0 <= summary.win_rate <= 1
    assert 1 <= summary.mean_rounds <= 50
    assert 0 <= summary.mean_fighter_hp <= 20
    assert len(summary.action_counts) == len(M0_ACTIONS)
    assert summary.action_counts == tuple(
        sum(result.action_counts[index] for result in summary.results) for index in range(len(M0_ACTIONS))
    )
    assert sum(summary.action_counts) == sum(sum(result.action_counts) for result in summary.results)
    assert summary.second_wind_use_rate == sum(
        result.action_counts[M0_ACTIONS.index(Action.SECOND_WIND)] > 0 for result in summary.results
    ) / summary.episodes
    assert summary.action_surge_use_rate == sum(
        result.action_counts[M0_ACTIONS.index(Action.ACTION_SURGE)] > 0 for result in summary.results
    ) / summary.episodes
    assert all(sum((result.won, result.lost, result.truncated)) == 1 for result in summary.results)


def test_evaluation_repeats_with_same_seeds_and_accepts_other_policies() -> None:
    seeds = [9, 2, 9, 4, 1, 7, 3, 5, 6, 8]
    first = evaluate(RandomAgent(seed=13), seeds)
    second = evaluate(RandomAgent(seed=13), seeds)
    assert first == second
    assert tuple(result.seed for result in first.results) == tuple(seeds)

    class EndTurnPolicy:
        def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
            assert action_mask[3]
            return 3

    fixed = evaluate(EndTurnPolicy(), [0, 1])
    assert fixed.episodes == 2
    assert fixed.action_counts[:3] == (0, 0, 0)
    assert fixed.action_counts[3] > 0

    with pytest.raises(ValueError, match="At least one"):
        evaluate(RandomAgent(seed=0), [])
