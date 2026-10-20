from types import SimpleNamespace

import numpy as np
import pytest

from agents.guided_planner import GuidedPlannerAgent
from agents.heuristic_agent import make_heuristic_agent
from agents.policy_prior import HeuristicPrior, MLPPolicyPrior, masked_softmax
from agents.value_functions import make_layout
from combat.stages import make_env
from combat.transitions import Status
from expert_iteration import check_seed_split, collect_expert, fit_prior
from value_training import EpsilonPolicy, save_value, targets_for


class ToyModel:
    stage = 'm5'
    action_count = 2

    def same_encounter(self, obs): return True
    def state_from_observation(self, obs): return (int(obs[0]),)
    def status(self, state):
        return Status.WIN if state == (3,) else Status.LOSS if state == (4,) else Status.ONGOING
    def legal_actions(self, state): return (0, 1) if state[0] < 3 else ()
    def observations(self, states): return np.asarray(states, dtype=float).reshape(-1, 1)
    def outcomes(self, state, action):
        if state == (0,):
            return ((0.5, (3,)), (0.5, (4,))) if action == 0 else ((0.8, (1,)), (0.2, (4,)))
        if state == (1,):
            return ((0.6, (3,)), (0.4, (4,))) if action == 0 else ((0.9, (2,)), (0.1, (4,)))
        return ((1., (3 if action == 0 else 4,)),)


class Anchor:
    def choose_action(self, obs, mask): return 0


def test_guided_exact_chance_depth_budget_and_conservative_margin():
    leaf = lambda obs: np.where(obs[:, 0] == 1, 0.8, 0.9)
    agent = GuidedPlannerAgent('m5', leaf, Anchor(), horizon=1, deviation_margin=0.2)
    agent._model = ToyModel()
    assert agent.choose_action(np.array([0]), np.ones(2, bool)) == 0
    assert dict(agent.last_diagnostics.q_values) == pytest.approx({0: 0.5, 1: 0.64})
    assert agent.last_diagnostics.proposed_action == 1
    assert agent.last_diagnostics.selected_value == 0.5
    agent.deviation_margin = 0.03
    assert agent.choose_action(np.array([0]), np.ones(2, bool)) == 1
    agent.horizon = 3
    assert agent.choose_action(np.array([0]), np.ones(2, bool)) == 1
    assert dict(agent.last_diagnostics.q_values)[1] == pytest.approx(0.72)
    # One expansion per candidate cuts off the next decision; chance weights
    # remain exact and the leaf changes the backed-up value in the expected way.
    agent.node_budget = 1
    agent.choose_action(np.array([0]), np.ones(2, bool))
    assert dict(agent.last_diagnostics.q_values)[1] == pytest.approx(0.648)
    assert agent.last_diagnostics.budget_cutoffs == 1
    assert agent.last_diagnostics.expanded_nodes <= 3
    # A prior prefers action 1; top-k still includes the heuristic anchor 0.
    agent.root_top_k = 1
    agent.prior = SimpleNamespace(probabilities=lambda obs, mask: np.array([0., 1.]))
    agent.choose_action(np.array([0]), np.ones(2, bool))
    assert set(dict(agent.last_diagnostics.q_values)) == {0, 1}


def test_prior_masks_and_round_trip(tmp_path):
    probs = masked_softmax(np.array([[1000, 0, 1001]]), np.array([[1, 0, 1]], bool))
    assert probs[0, 1] == 0
    assert probs.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        masked_softmax(np.ones((1, 3)), np.zeros((1, 3), bool))
    env = make_env('m6')
    obs, _ = env.reset(seed=180000)
    mask = env.action_masks()
    layout = make_layout('m6')
    weights = [(np.zeros((len(layout.high), len(mask))), np.arange(len(mask), dtype=float))]
    path = tmp_path / 'prior.npz'
    save_value(str(path), weights, dict(stage='m6', layers=1, kind='policy_prior'))
    prior = MLPPolicyPrior.load(path)
    p = prior.probabilities(obs, mask)
    assert p[~mask].sum() == 0
    assert prior.choose_action(obs, mask) in np.flatnonzero(mask)
    anchor = make_heuristic_agent('m6', 'adaptive')
    assert np.all(HeuristicPrior(anchor).probabilities(obs, mask)[mask] > 0)


def test_teacher_labels_survive_full_exploration_and_chunking():
    anchor = make_heuristic_agent('m5', 'tuned')
    leaf = lambda obs: np.full(len(obs), 0.4)
    data = collect_expert('m5', anchor, range(180000, 180002), epsilon=1, leaf=leaf)
    pieces = [collect_expert('m5', anchor, range(s, s + 1), epsilon=1, leaf=leaf) for s in (180000, 180001)]
    for key in data:
        np.testing.assert_array_equal(data[key], np.concatenate([p[key] for p in pieces]))
    assert np.all(data['mask'][np.arange(len(data['win'])), data['expert_action']])
    assert np.any(data['action'] != data['expert_action'])
    assert np.all(data['next_value'][data['done']] == data['win'][data['done']])


def test_td_lambda_uses_successors_and_stops_at_episode_boundaries():
    data = dict(win=np.array([1, 1, 0, 0]), next_value=np.array([0.2, 1., .8, 0.]),
                done=np.array([0, 1, 0, 1], bool), seed=np.array([1, 1, 2, 2]), step=np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(targets_for(data, 'td:0'), [.2, 1, .8, 0])
    np.testing.assert_allclose(targets_for(data, 'td:1'), [1, 1, 0, 0])
    np.testing.assert_allclose(targets_for(data, 'td:0.5'), [.6, 1, .4, 0])
    with pytest.raises(ValueError): targets_for(data, 'td:1.1')
    data['done'][-1] = False
    with pytest.raises(ValueError): targets_for(data, 'td:0.5')
    with pytest.raises(ValueError): check_seed_split(dict(seed=[1]), dict(seed=[1]))
    with pytest.raises(ValueError): check_seed_split(dict(seed=[130000]), dict(seed=[1]))


def test_epsilon_policy_does_not_report_stale_search_values():
    class Base:
        last_diagnostics = SimpleNamespace(value=0.7)
        def choose_action(self, obs, mask): return 0
    policy = EpsilonPolicy(Base(), 0, 0)
    policy.choose_action(np.zeros(1), np.ones(2, bool))
    assert policy.last_diagnostics.value == .7
    policy.epsilon = 1
    policy.choose_action(np.zeros(1), np.ones(2, bool))
    assert policy.last_diagnostics is None


def test_prior_fitting_reduces_validation_loss_and_keeps_masks(tmp_path):
    env = make_env('m5')
    obs, _ = env.reset(seed=180000)
    mask = env.action_masks()
    data = dict(observations=np.tile(obs, (16, 1)), mask=np.tile(mask, (16, 1)), expert_action=np.full(16, 3))
    weights, info = fit_prior('m5', data, data, hidden=[8], epochs=8, batch_size=4, lr=.05, threads=1)
    assert info['history'][-1]['val_cross_entropy'] < info['history'][0]['val_cross_entropy']
    assert MLPPolicyPrior(weights, 'm5').choose_action(obs, mask) == 3
