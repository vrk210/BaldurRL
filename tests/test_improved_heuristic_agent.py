"""Paired gate, common seeds, deterministic choices and live-state isolation."""
import copy
import numpy as np
import pytest
from agents.heuristic_agent import make_heuristic_agent
from agents.improved_heuristic_agent import ImprovedHeuristicAgent, paired_pvalue
from combat.stages import make_env


@pytest.mark.parametrize("stage, variant", [("m5", "tuned"), ("m6", "adaptive")])
def test_infinite_strictness_matches_every_decision(stage, variant):
    env = make_env(stage)
    base, agent = make_heuristic_agent(stage, variant), ImprovedHeuristicAgent(stage, alpha=0)
    for seed in (220000, 220001):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = env.action_masks()
            action = agent.choose_action(obs, mask)
            assert action == base.choose_action(obs, mask)
            assert agent.last_diagnostics.simulated_episodes == 0
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()


def test_bonferroni_rejects_noisy_maximum(monkeypatch):
    env = make_env("m5")
    obs, _ = env.reset(seed=220002)
    mask = env.action_masks()
    base = make_heuristic_agent("m5", "tuned").choose_action(obs, mask)
    alternative = next(int(a) for a in np.flatnonzero(mask) if a != base)
    agent = ImprovedHeuristicAgent("m5", looks=(4,), alpha=0.05)
    monkeypatch.setattr(agent, "_outcome", lambda obs, a, seed: int(a == alternative))
    assert agent.choose_action(obs, mask) == base
    agent = ImprovedHeuristicAgent("m5", looks=(16,), alpha=0.05)
    monkeypatch.setattr(agent, "_outcome", lambda obs, a, seed: int(a == alternative))
    assert agent.choose_action(obs, mask) == alternative
    assert paired_pvalue(16, 0) == 2**-16
    assert paired_pvalue(0, 0) == paired_pvalue(1, 1) == 1
    env.close()


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_common_seeds_reproducible_and_independent_of_combat(monkeypatch, stage):
    env = make_env(stage)
    obs, _ = env.reset(seed=220003)
    mask = env.action_masks()
    first, second = [ImprovedHeuristicAgent(stage, seed=72, looks=(3,)) for _ in range(2)]
    calls = []
    monkeypatch.setattr(first, "_outcome", lambda obs, a, seed: calls.append((a, seed)) or seed % 2)
    action = first.choose_action(obs, mask)
    seeds = [seed for _, seed in calls[:3]]
    assert all([seed for _, seed in calls[i:i + 3]] == seeds for i in range(0, len(calls), 3))
    env.np_random.integers(0, 100, size=100)
    state = copy.deepcopy(env.np_random.bit_generator.state)
    again = []
    monkeypatch.setattr(second, "_outcome", lambda obs, a, seed: again.append((a, seed)) or seed % 2)
    assert second.choose_action(obs, mask) == action
    assert calls == again
    assert first.last_diagnostics.scores == second.last_diagnostics.scores
    assert env.np_random.bit_generator.state == state
    obs2, _ = env.reset(seed=220004)
    third = ImprovedHeuristicAgent(stage, seed=72, looks=(3,))
    others = []
    monkeypatch.setattr(third, "_outcome", lambda obs, a, seed: others.append(seed) or 0)
    third.choose_action(obs2, env.action_masks())
    assert others[:3] == seeds
    env.close()


@pytest.mark.parametrize("kwargs", [{"alpha": -1}, {"looks": ()}, {"looks": (2, 1)},
                                    {"looks": (True,)}, {"futility_margin": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ImprovedHeuristicAgent("m5", **kwargs)


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_real_rollouts_are_deterministic_and_preserve_live_state(stage):
    env = make_env(stage)
    obs, _ = env.reset(seed=220005)
    mask, original = env.action_masks(), obs.copy()
    state = copy.deepcopy(env.np_random.bit_generator.state)
    agents = [ImprovedHeuristicAgent(stage, seed=99, looks=(2,)) for _ in range(2)]
    actions = [agent.choose_action(obs, mask) for agent in agents]
    assert actions[0] == actions[1]
    assert agents[0].last_diagnostics.scores == agents[1].last_diagnostics.scores
    assert agents[0].last_diagnostics.paired_counts == agents[1].last_diagnostics.paired_counts
    assert np.array_equal(obs, original)
    assert np.array_equal(mask, env.action_masks())
    assert env.np_random.bit_generator.state == state
    env.close()


def test_candidate_futility_uses_matching_prefixes_and_keeps_family_size(monkeypatch):
    env = make_env('m5')
    obs, _ = env.reset(seed=220007)
    mask = env.action_masks()
    agent = ImprovedHeuristicAgent('m5', looks=(4, 16), alpha=.05, prune_nonpositive=True)
    anchor = agent.heuristic.choose_action(obs, mask)
    alternate = next(int(a) for a in np.flatnonzero(mask) if a != anchor)
    calls = []
    monkeypatch.setattr(agent, '_outcome', lambda obs, a, seed: calls.append((a, seed)) or int(a == alternate))
    assert agent.choose_action(obs, mask) == alternate
    counts = dict(agent.last_diagnostics.sample_counts)
    assert counts[anchor] == counts[alternate] == 16
    assert all(n == 4 for a, n in counts.items() if a not in (anchor, alternate))
    assert agent.last_diagnostics.simulated_episodes == sum(counts.values())
    # Every continuation candidate receives the anchor's exact seed prefix.
    streams = {a: [s for called, s in calls if called == a] for a in counts}
    assert all(seeds == streams[anchor][:len(seeds)] for seeds in streams.values())
    env.close()
