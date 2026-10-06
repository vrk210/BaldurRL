"""Observed decision reconstruction preserves state and owns independent rolls."""

import copy
from unittest.mock import Mock

import numpy as np
import pytest

from combat.actors import ActorRef, Side
from combat.env import MAX_ROUNDS
from combat.resources import Resource
from combat.simulation import simulation_from_observation
from combat.stages import make_env
from combat.turns import TurnManager
from agents.heuristic_agent import make_heuristic_agent


STAGES = ("m0", "m1a", "m1b", "m2", "m3", "m4")


def observation_for(env):
    return env._get_observation() if env.stage == "m0" else env._observation()


@pytest.mark.parametrize("stage", STAGES)
def test_live_state_round_trip_and_mutation_isolation(stage):
    env = make_env(stage)
    env.reset(seed=17)
    roster = env._roster()
    if len(roster.allies) == 2:
        env.step(env.action_names.index("END_TURN"))
        roster.allies[0].hp = 0
    active = roster.get(env._turns().current)
    active.hp = 11
    for resource in Resource:
        active.resources.set(resource, 0)
    # One Action remains after Surge was spent; no other ability may recharge.
    active.resources.set(Resource.ACTION, 1)
    if len(roster.enemies) > 1:
        roster.enemies[0].hp = 0
    roster.enemies[-1].hp = 3
    env.round_number = MAX_ROUNDS
    env.turns = TurnManager(env.turns.order, MAX_ROUNDS, current=env.turns.current)
    obs = observation_for(env)
    mask = env.action_masks()
    live_rng = copy.deepcopy(env.np_random.bit_generator.state)
    sim = simulation_from_observation(stage, obs, seed=123)
    assert np.array_equal(observation_for(sim), obs)
    assert np.array_equal(sim.action_masks(), mask)
    assert sim.turns.current == env.turns.current
    assert sim.turns.round_number == env.turns.round_number == MAX_ROUNDS
    sim_active = sim._roster().get(sim.turns.current)
    for resource in Resource:
        assert sim_active.resources.get(resource) == active.resources.get(resource)
    sim_active.hp = 1
    sim_active.resources.set(Resource.ACTION, 0)
    sim._roster().enemies[-1].hp = 0
    sim.np_random.integers(100)
    assert np.array_equal(observation_for(env), obs)
    assert np.array_equal(env.action_masks(), mask)
    assert env.np_random.bit_generator.state == live_rng


@pytest.mark.parametrize("stage", STAGES)
def test_restored_end_turn_obeys_refresh_and_round_limit(stage):
    env = make_env(stage)
    obs, _ = env.reset(seed=5)
    if stage in ("m3", "m4"):
        obs, *_ = env.step(env.action_names.index("END_TURN"))
    fields = env.observation_fields
    for index, field in enumerate(fields):
        if field.endswith("_count"):
            obs[index] = 0
    sim = simulation_from_observation(stage, obs, seed=3)
    # Every automatic enemy misses, deterministically.
    rng = Mock(spec=np.random.Generator)
    rng.integers.return_value = np.int64(1)
    sim.np_random = rng
    obs, _, terminated, truncated, _ = sim.step(sim.action_names.index("END_TURN"))
    assert not terminated and not truncated and obs[-1] == 2
    first = sim._roster().allies[0]
    assert first.resources.get(Resource.ACTION) == 1
    assert first.resources.get(Resource.BONUS_ACTION) == 1
    for resource in (Resource.SECOND_WIND, Resource.ACTION_SURGE, Resource.CLEAVE):
        assert first.resources.get(resource) == 0
    # Restore the last-round decision for the last ally; truncation must not wrap.
    if stage in ("m3", "m4"):
        obs, *_ = sim.step(sim.action_names.index("END_TURN"))
    obs[-1] = MAX_ROUNDS
    last = simulation_from_observation(stage, obs, seed=3)
    last.np_random = rng
    _, _, terminated, truncated, _ = last.step(last.action_names.index("END_TURN"))
    assert not terminated and truncated
    assert last.round_number == last.turns.round_number == MAX_ROUNDS


def test_invalid_observations_and_dead_active_ally_rejected():
    env = make_env("m4")
    obs, _ = env.reset(seed=2)
    for index, value in ((0, 2), (1, -1), (2, 0.5), (31, 0), (1, np.nan), (17, 5), (13, 24), (14, 11), (15, 11), (16, 2), (18, 1)):
        invalid = obs.copy()
        invalid[index] = value
        with pytest.raises(ValueError):
            simulation_from_observation("m4", invalid, seed=0)
    obs[0] = 1
    obs[7] = 0
    with pytest.raises(ValueError, match="living ally"):
        simulation_from_observation("m4", obs, seed=0)
    with pytest.raises(ValueError):
        simulation_from_observation("m4", obs[:-1], seed=0)


def test_turn_manager_restores_current_without_advancing():
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1), ActorRef(Side.ENEMY, 0))
    turns = TurnManager(order, 49, current=order[1])
    assert turns.current == order[1] and turns.round_number == 49
    assert turns.peek_next(lambda ref: True) == (order[2], False)
    with pytest.raises(ValueError, match="belong"):
        TurnManager(order, current=ActorRef(Side.ENEMY, 1))


@pytest.mark.parametrize("stage, dead_ally_zero", [(stage, False) for stage in STAGES] + [("m3", True), ("m4", True)])
def test_reconstructed_partial_turn_matches_reseeded_live_transitions(stage, dead_ally_zero):
    env = make_env(stage)
    env.reset(seed=17)
    roster = env._roster()
    if len(roster.allies) == 2:
        env.step(env.action_names.index("END_TURN"))
        if dead_ally_zero:
            roster.allies[0].hp = 0
    active = roster.get(env.turns.current)
    active.hp = 14
    for resource in Resource:
        active.resources.set(resource, 0)
    active.resources.set(Resource.ACTION, 2)  # Surge spent; its Actions still remain.
    if len(roster.enemies) > 1:
        roster.enemies[0].hp = 0
    env.round_number = 49
    env.turns = TurnManager(env.turns.order, 49, current=env.turns.current)
    obs = observation_for(env)
    sim = simulation_from_observation(stage, obs, seed=701)
    env.np_random = np.random.default_rng(701)
    policy = make_heuristic_agent(stage)
    decisions = 0
    while True:
        mask = env.action_masks()
        assert np.array_equal(sim.action_masks(), mask)
        assert sim.turns.current == env.turns.current
        assert sim.turns.round_number == env.turns.round_number
        action = policy.choose_action(obs, mask)
        actual = env.step(action)
        reconstructed = sim.step(action)
        assert np.array_equal(actual[0], reconstructed[0])
        assert actual[1:] == reconstructed[1:]
        decisions += 1
        obs = actual[0]
        if actual[2] or actual[3]:
            break
        assert decisions < 30
    assert decisions >= 3  # Spent-turn attacks followed by a genuine turn transition.
