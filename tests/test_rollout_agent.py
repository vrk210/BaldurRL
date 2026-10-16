"""Deterministic planner budgets, scoring, policy isolation, and evaluation wiring."""

import copy
import json
import sys

import numpy as np
import pytest

import agents.rollout_agent as rollout_module
from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from combat.stages import make_env
from evaluation.evaluate import main


class ConstantPolicy:
    def __init__(self, action):
        self.action = action

    def choose_action(self, observation, action_mask):
        return self.action


class OutcomeSimulator:
    def __init__(self, wins, calls, seed, *, continuation=False):
        self.wins = wins
        self.calls = calls
        self.seed = seed
        self.continuation = continuation
        self.first = None

    def step(self, action):
        self.calls.append((self.seed, action))
        if self.first is None:
            self.first = action
            if self.continuation:
                return np.zeros(7), 0.0, False, False, {}
        reward = 1.0 if self.first in self.wins else -1.0
        return np.zeros(7), reward, True, False, {}

    def action_masks(self):
        return np.array([True, False, False, True])

    def close(self):
        pass


def controlled_simulator(monkeypatch, wins, calls, **kwargs):
    monkeypatch.setattr(rollout_module, "simulation_from_observation", lambda stage, obs, seed: OutcomeSimulator(wins, calls, seed, **kwargs))


def test_known_outcomes_maximize_wins_and_budget(monkeypatch):
    calls = []
    controlled_simulator(monkeypatch, {2}, calls)
    planner = RolloutAgent(rollouts_per_action=3, seed=8)
    mask = np.array([True, False, True, True])
    obs = np.array([20, 1, 1, 1, 1, 15, 1], dtype=np.float32)
    before = obs.copy(), mask.copy()
    assert planner.choose_action(obs, mask) == 2
    diagnostics = planner.last_diagnostics
    assert diagnostics.scores == ((0, 0.0), (2, 1.0), (3, 0.0))
    assert diagnostics.simulated_episodes == diagnostics.simulated_decisions == 9
    assert diagnostics.planning_seconds >= 0
    assert [action for seed, action in calls] == [0] * 3 + [2] * 3 + [3] * 3
    assert [seed for seed, action in calls[:3]] == [seed for seed, action in calls[3:6]]
    assert np.array_equal(obs, before[0]) and np.array_equal(mask, before[1])


@pytest.mark.parametrize("base_choice, expected", [(2, 2), (3, 0)])
def test_exact_ties_prefer_base_then_stable_index(monkeypatch, base_choice, expected):
    controlled_simulator(monkeypatch, {0, 2}, [])
    planner = RolloutAgent(continuation_factory=lambda seed: ConstantPolicy(base_choice), rollouts_per_action=2, seed=1)
    assert planner.choose_action(np.zeros(7), np.array([True, False, True, True])) == expected


def test_single_action_skips_simulator_factory_and_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Single action must require no planning")
    monkeypatch.setattr(rollout_module, "simulation_from_observation", forbidden)
    planner = RolloutAgent(continuation_factory=forbidden, seed=2)
    state = copy.deepcopy(planner.rng.bit_generator.state)
    assert planner.choose_action(np.zeros(7), np.array([False, False, False, True])) == 3
    assert planner.rng.bit_generator.state == state
    assert planner.last_diagnostics.simulated_episodes == 0
    assert planner.last_diagnostics.simulated_decisions == 0


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError, match="positive integer"):
        RolloutAgent(rollouts_per_action=budget)


def test_invalid_factory_mask_and_shapes():
    with pytest.raises(TypeError, match="callable"):
        RolloutAgent(continuation_factory=0)
    planner = RolloutAgent()
    with pytest.raises(ValueError, match="No legal"):
        planner.choose_action(np.zeros(7), np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        planner.choose_action(np.zeros(6), np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        planner.choose_action(np.zeros(7), np.ones(5, dtype=bool))


@pytest.mark.parametrize("choice", [-1, 1, 4, 0.5, True])
def test_invalid_continuation_choices_fail(monkeypatch, choice):
    controlled_simulator(monkeypatch, {0}, [], continuation=True)
    planner = RolloutAgent(continuation_factory=lambda seed: ConstantPolicy(choice), rollouts_per_action=1)
    with pytest.raises(ValueError, match="Continuation policy"):
        planner.choose_action(np.zeros(7), np.array([True, False, False, True]))


def test_policy_factory_isolates_stochastic_state(monkeypatch):
    calls = []
    policies = []
    seeds = []
    def factory(seed):
        policy = RandomAgent(seed)
        seeds.append(seed)
        policies.append(policy)
        return policy
    controlled_simulator(monkeypatch, {0}, calls, continuation=True)
    planner = RolloutAgent(continuation_factory=factory, rollouts_per_action=2, seed=3)
    planner.choose_action(np.zeros(7), np.array([True, False, False, True]))
    assert len(policies) == 4 and len({id(policy) for policy in policies}) == 4
    assert seeds[:2] == seeds[2:]
    assert [a for _, a in calls[1:4:2]] == [a for _, a in calls[5:8:2]]


@pytest.mark.parametrize("stage", ["m0", "m1a", "m1b", "m2", "m3", "m4"])
def test_planning_reproducible_and_ignores_live_rng(stage):
    env = make_env(stage)
    obs, _ = env.reset(seed=2)
    mask = env.action_masks()
    state = copy.deepcopy(env.np_random.bit_generator.state)
    first = RolloutAgent(stage, rollouts_per_action=1, seed=91)
    action = first.choose_action(obs, mask)
    assert mask[action]
    assert env.np_random.bit_generator.state == state
    # Change actual future rolls without changing any observable state.
    env.np_random.integers(0, 100, size=100)
    changed_state = copy.deepcopy(env.np_random.bit_generator.state)
    second = RolloutAgent(stage, rollouts_per_action=1, seed=91)
    assert second.choose_action(obs, mask) == action
    assert second.last_diagnostics.scores == first.last_diagnostics.scores
    assert second.last_diagnostics.simulated_decisions == first.last_diagnostics.simulated_decisions
    assert env.np_random.bit_generator.state == changed_state
    assert np.array_equal(mask, env.action_masks())
    third = RolloutAgent(stage, continuation_factory=lambda seed: RandomAgent(seed), rollouts_per_action=1, seed=18)
    fourth = RolloutAgent(stage, continuation_factory=lambda seed: RandomAgent(seed), rollouts_per_action=1, seed=18)
    assert third.choose_action(obs, mask) == fourth.choose_action(obs, mask)
    assert third.last_diagnostics.scores == fourth.last_diagnostics.scores


def test_cli_rollout_run_card(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--stage", "m4", "--agent", "rollout", "--rollouts-per-action", "1", "--episodes", "1", "--agent-seed", "12", "--trace-dir", str(tmp_path), "--format", "json"])
    main()
    assert json.loads(capsys.readouterr().out)["episodes"] == 1
    card = json.loads((tmp_path / "run_card.json").read_text())
    assert card["policy"] == "rollout" and card["agent_seed"] == 12
    assert card["rollout_settings"]["continuation_policy"] == "heuristic"
    assert card["rollout_settings"]["rollouts_per_action"] == 1
    assert card["rollout_settings"]["planner_seed"] == 12


def test_cli_ppo_continuation_reuses_model_with_fresh_adapters(monkeypatch, capsys):
    MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO
    from evaluation.evaluate import SB3Policy
    model = object()
    loaded = []
    adapters = []
    monkeypatch.setattr(MaskablePPO, "load", lambda path: loaded.append(path) or model)
    def choose(adapter, observation, mask):
        adapters.append(adapter)
        assert adapter.model is model and adapter.deterministic
        return int(np.flatnonzero(mask)[0])
    monkeypatch.setattr(SB3Policy, "choose_action", choose)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--agent", "rollout", "--rollout-policy", "ppo", "--model", "fixed.zip", "--rollouts-per-action", "1", "--episodes", "1", "--format", "json"])
    main()
    assert loaded == ["fixed.zip"]
    assert len({id(adapter) for adapter in adapters}) > 1
    assert json.loads(capsys.readouterr().out)["episodes"] == 1
