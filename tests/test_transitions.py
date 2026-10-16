"""Exact transition enumeration matches the sampling environments."""

import numpy as np
import pytest

from combat.env import MAX_ROUNDS
from combat.distributions import attack_damage_distribution, second_wind_healing_distribution
from combat.mechanics import attack_hit
from combat.damage import DamageSpec
from combat.resources import Resource
from combat.stages import make_env
from combat.transitions import (
    RosterTransitionModel,
    Status,
    register_transition_model,
    transition_model,
)

from transition_harness import exhaustive_distribution, load_state, max_abs_z, model_for, sampled_counts


STAGES = ("m0", "m1a", "m1b", "m2", "m3", "m4")
_RESOURCE_INDEX = {resource: 1 + index for index, resource in enumerate(Resource)}


def edit(model, state, *, actor=None, round_number=None, allies=None, enemies=None):
    values = list(state)
    if actor is not None:
        values[1] = actor
    if round_number is not None:
        values[2] = round_number
    width = 1 + len(Resource)
    for slot, fields in (allies or {}).items():
        base = 3 + width * slot
        for name, value in fields.items():
            values[base + (0 if name == "hp" else _RESOURCE_INDEX[Resource[name]])] = value
    enemy_base = 3 + width * model.n_allies
    for slot, hp in (enemies or {}).items():
        values[enemy_base + slot] = hp
    return tuple(values)


def assert_same_distribution(left, right, tol=1e-12):
    assert set(left) == set(right)
    for key in left:
        assert abs(left[key] - right[key]) <= tol, key


# ---------------------------------------------------------------- shared helpers


def test_attack_distribution_matches_rule_enumeration():
    damage = DamageSpec(1, 8, 3)
    dist = attack_damage_distribution(5, 15, damage)
    assert abs(sum(dist.values()) - 1.0) < 1e-15
    hits = sum(attack_hit(roll, 5, 15)[0] for roll in range(1, 21))
    assert dist[0] == pytest.approx((20 - hits) / 20)
    # A natural 20 against an unreachable AC still hits with doubled dice.
    crit_only = attack_damage_distribution(0, 100, damage)
    assert crit_only[0] == pytest.approx(19 / 20)
    assert min(k for k in crit_only if k) == 2 + 3 and max(crit_only) == 16 + 3
    assert crit_only[19] == pytest.approx(1 / 20 / 64)
    halved = attack_damage_distribution(5, 15, damage, 2)
    assert min(k for k in halved if k) == 2 and max(halved) == 19 // 2
    tiny = attack_damage_distribution(5, 2, DamageSpec(1, 4, 0), 2)
    assert min(k for k in tiny if k) == 1  # halving keeps at least 1 damage
    assert second_wind_healing_distribution() == {heal: pytest.approx(0.1) for heal in range(3, 13)}


# ---------------------------------------------------------------- state round trips


@pytest.mark.parametrize("stage", STAGES)
def test_observation_round_trip_and_legality(stage):
    env, model = model_for(stage, 11)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = model.state_from_env(env)
        obs = env._get_observation() if stage == "m0" else env._observation()
        assert np.array_equal(model.observation(state), obs)
        if model.status(state) is Status.ONGOING:
            assert model.state_from_observation(obs) == state
            assert np.array_equal(model.legal_mask(state), env.action_masks())
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            env.step(action)
        else:
            assert not env.action_masks().any() and model.legal_actions(state) == ()
            env.reset(seed=11)
    with pytest.raises(ValueError):
        other = make_env(stage)
        other_obs, _ = other.reset(seed=12)
        if stage != "m0":
            model.state_from_observation(other_obs)
        else:
            raise ValueError("M0 has no randomized encounter statistics")


def test_registry_rejects_unknown_stage_and_accepts_plugins():
    with pytest.raises(ValueError):
        transition_model("m99", np.zeros(3))
    sentinel = object()
    register_transition_model("plugin_test", lambda stage, obs: sentinel)
    assert transition_model("plugin_test", np.zeros(1)) is sentinel


# ---------------------------------------------------------------- exhaustive checks


def exhaustive_cases():
    """(stage, seed, state edits, actions) with at most two enemy attacks per step."""
    return [
        ("m0", 1, {}, None),
        ("m0", 1, dict(round_number=MAX_ROUNDS, allies={0: dict(hp=4)}, enemies={0: 2}), None),
        ("m1a", 2, dict(allies={0: dict(hp=3, ACTION=2, BONUS_ACTION=0)}, enemies={0: 1}), None),
        ("m1b", 3, {}, None),
        ("m1b", 3, dict(round_number=MAX_ROUNDS, allies={0: dict(hp=6)}), None),
        ("m2", 4, {}, None),
        ("m2", 4, dict(allies={0: dict(hp=2)}, enemies={0: 0, 1: 3}), None),
        ("m3", 5, {}, None),
        ("m3", 5, dict(actor=1, allies={0: dict(hp=0), 1: dict(hp=7)}), None),
        ("m3", 5, dict(actor=0, allies={0: dict(hp=9), 1: dict(hp=0)}, round_number=MAX_ROUNDS), None),
        ("m3", 5, dict(actor=1, allies={0: dict(hp=3), 1: dict(hp=5)}, round_number=MAX_ROUNDS), None),
        ("m4", 6, dict(enemies={1: 0}), None),
        ("m4", 6, dict(actor=1, allies={0: dict(hp=2), 1: dict(hp=4, SECOND_WIND=0)}, enemies={0: 0}), None),
        ("m4", 7, dict(actor=1, enemies={0: 0, 2: 5}, round_number=MAX_ROUNDS), None),
    ]


@pytest.mark.parametrize("case", exhaustive_cases(), ids=lambda case: f"{case[0]}-{case[1]}-{sorted(case[2])}")
def test_enumeration_equals_exhaustive_rng_paths(case):
    stage, seed, edits, _ = case
    env, model = model_for(stage, seed)
    state = edit(model, model.state_from_env(env), **edits)
    load_state(env, state)
    legal = model.legal_actions(state)
    assert np.array_equal(model.legal_mask(state), env.action_masks())
    for action in legal:
        expected = exhaustive_distribution(env, model, state, action)
        enumerated = dict((s, p) for p, s in model.outcomes(state, action))
        assert len(enumerated) == len(model.outcomes(state, action)), "successors must be merged"
        assert abs(sum(enumerated.values()) - 1.0) < 1e-12
        assert_same_distribution(enumerated, expected)


def test_terminal_and_truncation_states():
    env, model = model_for("m3", 5)
    start = model.state_from_env(env)
    # Killing the last enemy wins immediately, with no END_TURN.
    state = edit(model, start, enemies={0: 0, 1: 1})
    win = [s for p, s in model.outcomes(state, 1) if model.status(s) is Status.WIN]
    assert len(win) == 1 and model.legal_actions(win[0]) == ()
    # Last allied END_TURN in round 50 truncates unless the enemies kill everyone.
    last = edit(model, start, actor=1, round_number=MAX_ROUNDS, allies={0: dict(hp=1), 1: dict(hp=1)})
    statuses = {model.status(s) for p, s in model.outcomes(last, 5)}
    assert statuses <= {Status.TRUNCATED, Status.LOSS} and Status.TRUNCATED in statuses
    # The first ally's END_TURN in round 50 hands play to the second ally.
    first = edit(model, start, round_number=MAX_ROUNDS)
    ((p, nxt),) = model.outcomes(first, 5)
    assert p == 1.0 and model.status(nxt) is Status.ONGOING and nxt[1] == 1 and nxt[2] == MAX_ROUNDS
    # A wrapped round refreshes the next living ally's Action and Bonus Action only.
    spent = edit(model, start, actor=1, allies={0: dict(ACTION=0, BONUS_ACTION=0, ACTION_SURGE=0)})
    seen_slots = set()
    for p, s in model.outcomes(spent, 5):
        if model.status(s) is Status.ONGOING:
            slot = 0 if s[3] > 0 else 1  # a dead first ally is skipped
            seen_slots.add(slot)
            assert s[1] == slot and s[2] == 2
            base = 3 + (1 + len(Resource)) * slot
            assert s[base + _RESOURCE_INDEX[Resource.ACTION]] == 1
            assert s[base + _RESOURCE_INDEX[Resource.BONUS_ACTION]] == 1
            assert s[3 + _RESOURCE_INDEX[Resource.ACTION_SURGE]] == 0
    assert 0 in seen_slots
    with pytest.raises(ValueError):
        model.outcomes(win[0], 5)


# ---------------------------------------------------------------- statistical checks


@pytest.mark.parametrize("action_name", ["CLEAVE", "END_TURN"])
def test_three_enemy_steps_match_sampling(action_name):
    env, model = model_for("m4", 8)
    state = edit(model, model.state_from_env(env), actor=1, allies={0: dict(hp=9), 1: dict(hp=14)})
    action = env.action_names.index(action_name)
    expected = dict((s, p) for p, s in model.outcomes(state, action))
    samples = 20_000
    counts = sampled_counts(env, model, state, action, samples, seed=99)
    assert set(counts) <= set(expected)
    assert max_abs_z(counts, expected, samples) < 5.0


def test_automatic_attack_decision_targets_lowest_living_ally():
    from combat.actors import ActorRef, Side
    from combat.legality import automatic_attack_decision

    env = make_env("m4")
    env.reset(seed=2)
    roster = env._roster()
    enemy = ActorRef(Side.ENEMY, 1)
    assert automatic_attack_decision(enemy, roster, env.decisions).target_index == 0
    roster.allies[0].hp = 0
    assert automatic_attack_decision(enemy, roster, env.decisions).target_index == 1
    roster.allies[1].hp = 0
    assert automatic_attack_decision(enemy, roster, env.decisions) is None
