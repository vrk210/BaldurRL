"""M5/M6 reconstruction from observations: round trips, live equivalence, validation."""

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from combat.resources import Resource
from combat.simulation import simulation_from_observation
from combat.stages import make_env


def _advance_to_rich_state(stage: str, seed: int):
    """Play a few random decisions so conditions, charges, and positions vary."""
    env = make_env(stage)
    obs, _ = env.reset(seed=seed)
    agent = RandomAgent(seed)
    for _ in range(9):
        obs, _, terminated, truncated, _ = env.step(agent.choose_action(obs, env.action_masks()))
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)
    return env, obs


@pytest.mark.parametrize("stage", ["m5", "m6"])
@pytest.mark.parametrize("seed", [3, 8, 21])
def test_round_trip_and_reseeded_transitions_match(stage, seed) -> None:
    env, obs = _advance_to_rich_state(stage, seed)
    mask = env.action_masks()
    sim = simulation_from_observation(stage, obs, seed=99)
    assert np.array_equal(sim._observation(), obs)
    assert np.array_equal(sim.action_masks(), mask)
    assert sim.turns.current == env.turns.current and sim.round_number == env.round_number
    for live, restored in zip(env.allies, sim.allies):
        assert (live.hp, live.dodging, live.deep, live.disengaged) == (
            restored.hp, restored.dodging, restored.deep, restored.disengaged)
        for resource in Resource:
            assert live.resources.get(resource) == restored.resources.get(resource)
    for live, restored in zip(env.enemies, sim.enemies):
        assert (live.hp, live.max_hp, live.role, live.targeting, live.rank, live.prone, live.last_attacker) == (
            restored.hp, restored.max_hp, restored.role, restored.targeting, restored.rank, restored.prone,
            restored.last_attacker)
        for resource in (Resource.HEAL, Resource.REACTION):
            assert live.resources.get(resource) == restored.resources.get(resource)
    env.np_random = np.random.default_rng(99)
    policy = RandomAgent(7)
    for _ in range(40):
        action = policy.choose_action(obs, env.action_masks())
        actual, restored = env.step(action), sim.step(action)
        assert np.array_equal(actual[0], restored[0])
        assert actual[1:4] == restored[1:4]
        obs = actual[0]
        if actual[2] or actual[3]:
            break


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_restoration_isolates_live_state(stage) -> None:
    env, obs = _advance_to_rich_state(stage, 5)
    state = env.np_random.bit_generator.state
    sim = simulation_from_observation(stage, obs, seed=1)
    sim.allies[0].hp = 0
    sim.enemies[0].prone = not sim.enemies[0].prone
    assert np.array_equal(env._observation(), obs)
    assert env.np_random.bit_generator.state == state


def test_invalid_tactical_observations_rejected() -> None:
    env = make_env("m6")
    obs, _ = env.reset(seed=4)  # BRUTE, ARCHER, HEALER
    index = {name: position for position, name in enumerate(env.observation_fields)}
    corruptions = {
        "two roles": {"enemy_0_is_archer": 1},
        "no targeting": {"enemy_1_targets_retaliate": 0, "enemy_1_targets_weakest": 0, "enemy_1_targets_random": 0},
        "duplicate roles": {"enemy_1_is_archer": 0, "enemy_1_is_brute": 1},
        "hp above max": {"enemy_0_hp": 26, "enemy_0_max_hp": 20},
        "wrong die": {"enemy_1_damage_die_size": 6},
        "stat out of profile": {"enemy_2_attack_bonus": 8},
        "heal on non-healer": {"enemy_0_heal_count": 1},
        "wrong rank": {"enemy_0_back_rank": 1},
        "dead active ally": {"ally_0_hp": 0},
        "all enemies dead": {"enemy_0_hp": 0, "enemy_1_hp": 0, "enemy_2_hp": 0},
        "fractional": {"ally_0_hp": 10.5},
    }
    for label, changes in corruptions.items():
        bad = obs.copy()
        for name, value in changes.items():
            bad[index[name]] = value
        with pytest.raises(ValueError):
            simulation_from_observation("m6", bad, seed=0)
            pytest.fail(label)
    with pytest.raises(ValueError):
        simulation_from_observation("m5", obs, seed=0)  # wrong stage length


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_rollout_planner_runs_on_tactical_stages(stage) -> None:
    env = make_env(stage)
    obs, _ = env.reset(seed=12)
    planner = RolloutAgent(stage, rollouts_per_action=2, seed=0)
    action = planner.choose_action(obs, env.action_masks())
    assert env.action_masks()[action]
    assert planner.last_diagnostics.simulated_episodes == 2 * int(env.action_masks().sum())
