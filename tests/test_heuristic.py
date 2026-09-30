"""Heuristic policy behaviour and evaluation-hardening contracts."""

import itertools
import json
from math import sqrt
from statistics import pstdev

import numpy as np
import pytest

from agents.heuristic_agent import HeuristicAgent
from combat.env import M0_ACTIONS
from evaluation.evaluate import EpisodeResult, EvaluationSummary, SB3Policy, evaluate


def _obs(fighter_hp: float = 20.0, fighter_action: float = 1.0) -> np.ndarray:
    return np.array(
        [fighter_hp, fighter_action, 1.0, 1.0, 1.0, 15.0, 1.0],
        dtype=np.float32,
    )


def _mask(flags: tuple[bool, bool, bool, bool]) -> np.ndarray:
    return np.array(flags, dtype=bool)


@pytest.mark.parametrize(
    ("fighter_hp", "expected"),
    [(1.0, 1), (5.0, 1), (10.0, 1)],
)
def test_heuristic_uses_second_wind_when_hurt_and_legal(
    fighter_hp: float, expected: int
) -> None:
    agent = HeuristicAgent()
    action = agent.choose_action(_obs(fighter_hp=fighter_hp), _mask((True, True, True, True)))
    assert action == expected


def test_heuristic_prefers_attack_when_hp_high() -> None:
    agent = HeuristicAgent()
    action = agent.choose_action(_obs(fighter_hp=20.0), _mask((True, True, True, True)))
    assert action == 0


def test_heuristic_skips_second_wind_at_full_hp() -> None:
    agent = HeuristicAgent()
    action = agent.choose_action(_obs(fighter_hp=20.0), _mask((False, True, False, True)))
    assert action == 3


def test_heuristic_second_wind_threshold_boundary() -> None:
    agent = HeuristicAgent(second_wind_threshold=10)
    assert (
        agent.choose_action(_obs(fighter_hp=10.0), _mask((True, True, False, True))) == 1
    )
    assert (
        agent.choose_action(_obs(fighter_hp=11.0), _mask((True, True, False, True))) == 0
    )


def test_heuristic_uses_action_surge_without_actions_when_attack_illegal() -> None:
    agent = HeuristicAgent()
    action = agent.choose_action(
        _obs(fighter_hp=20.0, fighter_action=0.0), _mask((False, False, True, True))
    )
    assert action == 2


def test_heuristic_skips_action_surge_while_actions_remain() -> None:
    agent = HeuristicAgent()
    action = agent.choose_action(
        _obs(fighter_hp=20.0, fighter_action=1.0), _mask((False, False, True, True))
    )
    assert action == 3


def test_heuristic_falls_back_to_end_turn() -> None:
    agent = HeuristicAgent()
    assert (
        agent.choose_action(_obs(fighter_hp=20.0), _mask((False, False, False, True)))
        == 3
    )


def test_heuristic_falls_back_to_first_legal_index() -> None:
    agent = HeuristicAgent()
    # High HP skips Second Wind; Attack/Surge/EndTurn illegal leaves index 1.
    assert (
        agent.choose_action(_obs(fighter_hp=20.0), _mask((False, True, False, False)))
        == 1
    )


def test_heuristic_raises_on_empty_mask() -> None:
    agent = HeuristicAgent()
    with pytest.raises(ValueError, match="No legal actions"):
        agent.choose_action(_obs(), _mask((False, False, False, False)))


def test_heuristic_respects_mask_across_obs_and_mask_sweep() -> None:
    agent = HeuristicAgent()
    hp_values = [0.0, 1.0, 5.0, 10.0, 11.0, 19.0, 20.0]
    action_counts = [0.0, 1.0, 2.0]
    for fighter_hp, fighter_action, bits in itertools.product(
        hp_values, action_counts, range(1, 16)
    ):
        mask = _mask(tuple(bool(bits & (1 << index)) for index in range(4)))
        action = agent.choose_action(
            _obs(fighter_hp=fighter_hp, fighter_action=fighter_action), mask
        )
        assert mask[action], (fighter_hp, fighter_action, mask)


def test_heuristic_is_deterministic_for_same_obs_and_mask() -> None:
    agent = HeuristicAgent()
    observation = _obs(fighter_hp=8.0)
    mask = _mask((True, True, True, True))
    assert agent.choose_action(observation, mask) == agent.choose_action(
        observation, mask
    )


def test_heuristic_completes_episodes_on_fixed_seeds() -> None:
    summary = evaluate(HeuristicAgent(), [0, 1, 2])
    assert summary.episodes == 3
    assert summary.wins + summary.losses + summary.truncations == 3


def _episode(
    seed: int, won: bool, rounds: int, fighter_hp: int
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        won=won,
        lost=not won,
        truncated=False,
        rounds=rounds,
        fighter_hp=fighter_hp,
        action_counts=tuple(0 for _ in M0_ACTIONS),
    )


def test_win_rate_se_matches_binomial_standard_error() -> None:
    summary = EvaluationSummary(
        tuple(
            _episode(seed, won, rounds=3, fighter_hp=10)
            for seed, won in enumerate([True, True, True, False])
        )
    )
    rate = 0.75
    assert summary.win_rate == pytest.approx(rate)
    assert summary.win_rate_se == pytest.approx(sqrt(rate * (1.0 - rate) / 4))


def test_std_metrics_match_pstdev_and_zero_for_single_episode() -> None:
    summary = EvaluationSummary(
        (
            _episode(0, True, rounds=2, fighter_hp=20),
            _episode(1, False, rounds=6, fighter_hp=0),
            _episode(2, True, rounds=4, fighter_hp=12),
        )
    )
    assert summary.mean_rounds_std == pytest.approx(pstdev([2, 6, 4]))
    assert summary.mean_fighter_hp_std == pytest.approx(pstdev([20, 0, 12]))

    single = EvaluationSummary((_episode(0, True, rounds=5, fighter_hp=7),))
    assert single.mean_rounds_std == 0.0
    assert single.mean_fighter_hp_std == 0.0


def test_summary_as_dict_is_json_serializable_with_expected_keys() -> None:
    summary = EvaluationSummary(
        (
            _episode(0, True, rounds=2, fighter_hp=20),
            _episode(1, False, rounds=6, fighter_hp=0),
        )
    )
    payload = summary.as_dict()
    assert set(payload) == {
        "episodes",
        "wins",
        "losses",
        "truncations",
        "win_rate",
        "win_rate_se",
        "mean_rounds",
        "mean_rounds_std",
        "mean_fighter_hp",
        "mean_fighter_hp_std",
        "action_counts",
        "second_wind_use_rate",
        "action_surge_use_rate",
    }
    assert json.loads(json.dumps(payload)) == payload


def test_sb3_policy_returns_model_action_and_passes_mask_through() -> None:
    received: dict[str, object] = {}

    class FakeModel:
        def predict(
            self, obs: np.ndarray, action_masks: np.ndarray, deterministic: bool
        ) -> tuple[np.ndarray, None]:
            received["obs"] = obs
            received["mask"] = np.asarray(action_masks).copy()
            received["deterministic"] = deterministic
            return np.array(2), None

    policy = SB3Policy(FakeModel())
    observation = _obs()
    mask = _mask((False, False, True, True))
    action = policy.choose_action(observation, mask)

    assert action == 2
    assert isinstance(action, int)
    assert np.array_equal(np.asarray(received["mask"]), mask)
    assert received["deterministic"] is True


def test_sb3_policy_accepts_bare_action_result() -> None:
    class BareModel:
        def predict(
            self, obs: np.ndarray, action_masks: np.ndarray, deterministic: bool
        ) -> np.ndarray:
            return np.array([1])

    assert SB3Policy(BareModel()).choose_action(_obs(), _mask((True, True, True, True))) == 1


def test_evaluate_with_heuristic_is_deterministic_and_rejects_empty_seeds() -> None:
    seeds = [0, 1, 2, 3]
    assert evaluate(HeuristicAgent(), seeds) == evaluate(HeuristicAgent(), seeds)
    with pytest.raises(ValueError, match="At least one"):
        evaluate(HeuristicAgent(), [])
