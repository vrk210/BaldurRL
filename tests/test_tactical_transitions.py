"""Exact M5/M6 transitions agree with the real environment and its rule functions."""

from collections import defaultdict

import numpy as np
import pytest

import combat.tactics as tactics
from agents.expectimax_agent import ExpectimaxAgent
from agents.random_agent import RandomAgent
from agents.value_functions import make_layout, value_features
from combat.damage import DamageSpec
from combat.distributions import attack_damage_distribution
from combat.rolls import RollMode, resolve_attack_with_mode, roll_attack
from combat.stages import make_env
from combat.tactical_transitions import (
    TacticalTransitionModel,
    attack_outcome_distribution,
    enumerate_paths,
    roll_hit_distribution,
)
from combat.transitions import Status, transition_model


def _visited_states(stage: str, seed: int, steps: int = 40):
    """(env-free) observations along a seeded random-policy episode."""
    env = make_env(stage)
    obs, _ = env.reset(seed=seed)
    agent = RandomAgent(seed)
    seen = []
    for _ in range(steps):
        seen.append(obs.copy())
        obs, _, terminated, truncated, _ = env.step(agent.choose_action(obs, env.action_masks()))
        if terminated or truncated:
            break
    return seen


def _raw_distribution(model: TacticalTransitionModel, state, action: int, limit: int):
    """Every random path through the unmodified `env.step`, or None past `limit` paths."""
    merged = defaultdict(float)

    def run(rng):
        env = model.load(state)
        env.np_random = rng
        env.step(action)
        return model.state_from_env(env)

    for count, (probability, successor) in enumerate(enumerate_paths(run), start=1):
        if count > limit:
            return None
        merged[successor] += probability
    return merged


def test_normal_attack_distribution_matches_m4_derivation():
    damage = DamageSpec(1, 8, 3)
    for armor_class in (12, 16, 19):
        for divisor in (1, 2):
            collapsed = defaultdict(float)
            for (_, amount), p in attack_outcome_distribution(5, armor_class, damage, RollMode.NORMAL, divisor):
                collapsed[amount] += p
            reference = attack_damage_distribution(5, armor_class, damage, divisor)
            assert set(collapsed) == set(reference)
            for amount, p in reference.items():
                assert collapsed[amount] == pytest.approx(p, abs=1e-15)


@pytest.mark.parametrize("mode", list(RollMode))
def test_moded_distributions_sum_to_one_and_order_modes(mode):
    distribution = attack_outcome_distribution(5, 15, DamageSpec(1, 10, 3), mode, 1)
    assert sum(p for _, p in distribution) == pytest.approx(1.0, abs=1e-12)
    hit = dict(roll_hit_distribution(5, 15, mode))
    assert hit[True] + hit[False] == pytest.approx(1.0, abs=1e-15)
    normal = dict(roll_hit_distribution(5, 15, RollMode.NORMAL))[True]
    if mode is RollMode.ADVANTAGE:
        assert hit[True] > normal
    elif mode is RollMode.DISADVANTAGE:
        assert hit[True] < normal


def test_collapsed_draws_are_restored_after_enumeration():
    env = make_env("m5")
    obs, _ = env.reset(seed=4)
    model = transition_model("m5", obs)
    state = model.state_from_observation(obs)
    model.outcomes(state, model.legal_actions(state)[0])
    assert tactics.resolve_attack_with_mode is resolve_attack_with_mode
    assert tactics.roll_attack is roll_attack


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_observation_round_trip_and_legality(stage):
    for obs in _visited_states(stage, 51300):
        env = make_env(stage)
        model = transition_model(stage, obs)
        state = model.state_from_observation(obs)
        np.testing.assert_array_equal(model.observation(state), obs)
        assert model.same_encounter(obs)
        assert model.status(state) is Status.ONGOING


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_exact_outcomes_equal_raw_path_enumeration(stage):
    """Every cheap (state, action) pair matches the unmodified environment to rounding."""
    checked = 0
    for obs in _visited_states(stage, 51301)[::2]:
        model = transition_model(stage, obs)
        state = model.state_from_observation(obs)
        for action in model.legal_actions(state):
            raw = _raw_distribution(model, state, action, limit=3000)
            if raw is None:
                continue
            expected = {successor: p for p, successor in model.outcomes(state, action)}
            assert set(raw) == set(expected)
            for successor, p in expected.items():
                assert raw[successor] == pytest.approx(p, abs=1e-12)
            checked += 1
    assert checked >= 10


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_enemy_phase_matches_sampling(stage):
    """A full three-enemy END_TURN agrees with 20k real samples (no unseen outcome, small z)."""
    env = make_env(stage)
    obs, _ = env.reset(seed=51302)
    model = transition_model(stage, obs)
    # Ally 0 ends its turn, then ally 1 ends its turn: the second hand-over runs all enemies.
    state = model.state_from_observation(obs)
    (_, state), = model.outcomes(state, model.action_count - 1)
    expected = {successor: p for p, successor in model.outcomes(state, model.action_count - 1)}
    assert sum(expected.values()) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(7)
    counts = defaultdict(int)
    samples = 20_000
    for _ in range(samples):
        live = model.load(state)
        live.np_random = rng
        live.step(model.action_count - 1)
        counts[model.state_from_env(live)] += 1
    assert set(counts) <= set(expected)
    for successor, p in expected.items():
        if samples * p >= 20:
            sd = np.sqrt(samples * p * (1 - p))
            assert abs(counts[successor] - samples * p) < 5.5 * sd


def test_afterstate_features_zero_per_turn_fields():
    layout = make_layout("m6")
    env = make_env("m6")
    obs, _ = env.reset(seed=3)
    features = value_features(layout, obs[None], afterstate=True)[0]
    fields = env.observation_fields
    for name in ("ally_0_action_count", "ally_1_bonus_action_count", "ally_0_movement_count", "ally_1_disengaged"):
        assert features[fields.index(name)] == 0.0
    for name in ("ally_0_hp", "ally_0_trip_count", "ally_0_action_surge_count", "ally_1_cleave_count"):
        assert features[fields.index(name)] > 0.0


def test_one_decision_planner_scores_end_turn_by_afterstate():
    env = make_env("m5")
    obs, _ = env.reset(seed=11)
    leaf = lambda observations: np.full(len(observations), 0.4)  # noqa: E731
    after = lambda observations: np.full(len(observations), 0.9)  # noqa: E731
    agent = ExpectimaxAgent("m5", depth=1, leaf_value=leaf, afterstate_value=after, decision_horizon=1)
    action = agent.choose_action(obs, env.action_masks())
    q = dict(agent.last_diagnostics.q_values)
    assert env.action_names[action] == "END_TURN"
    assert q[action] == pytest.approx(0.9)
    assert set(q) == set(np.flatnonzero(env.action_masks()).tolist())
