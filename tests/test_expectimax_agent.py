"""Expectimax search: exactness against an independent M2 optimum, caching, and API."""

import numpy as np
import pytest

import agents.expectimax_agent as expectimax_module
from agents.expectimax_agent import ExpectimaxAgent
from combat.stages import make_env
from combat.transitions import Status
from evaluation.evaluate import evaluate

import m2_oracle


def m2_encounter(seed):
    env = make_env("m2")
    obs, _ = env.reset(seed=seed)
    enemies = [dict(hp=e.max_hp, ac=e.armor_class, atk=e.attack_bonus, die=e.damage.die_size, bonus=e.damage.bonus)
               for e in env.enemies]
    return env, obs, enemies


@pytest.fixture(scope="module")
def m2_solution():
    env, obs, enemies = m2_encounter(9003)
    V, Q = m2_oracle.solve(enemies)
    return obs, V, Q


def m2_obs(base, f, e0, e1, sw, su, cl, a, b, round_number=1):
    obs = base.copy()
    obs[[0, 1, 2, 3, 4, 5, 6, 12, 18]] = (f, a, b, sw, su, cl, e0, e1, round_number)
    return obs


def oracle_index(obs):
    o = obs.astype(int)
    return (o[0], o[6], o[12], o[3], o[4], o[5], o[1], o[2])


@pytest.mark.parametrize("state", [
    (10, 3, 3, 0, 0, 0, 0, 0),
    (1, 5, 1, 1, 0, 0, 1, 0),
    (3, 3, 4, 0, 0, 1, 2, 1),
    (9, 2, 2, 1, 0, 0, 2, 1),
    (5, 0, 1, 0, 0, 1, 1, 1),
])
def test_unbounded_search_reproduces_independent_m2_optimum(m2_solution, state):
    base, V, _ = m2_solution
    f, e0, e1, sw, su, cl, a, b = state
    obs = m2_obs(base, f, e0, e1, sw, su, cl, a, b)
    agent = ExpectimaxAgent("m2", depth=None)
    model = agent.model_for(obs)
    assert agent.state_value(model.state_from_observation(obs)) == pytest.approx(V[oracle_index(obs)], abs=1e-12)


def test_exact_leaf_gives_exact_root_q_values(m2_solution):
    base, V, Q = m2_solution
    leaf = lambda observations: np.array([V[oracle_index(o)] for o in observations])
    agent = ExpectimaxAgent("m2", depth=1, leaf_value=leaf)
    env = make_env("m2")
    obs, _ = env.reset(seed=9003)
    steps = 0
    done = False
    while not done:
        mask = env.action_masks()
        action = agent.choose_action(obs, mask)
        q_star = Q[(slice(None),) + oracle_index(obs)]
        if agent.last_diagnostics.q_values:
            for legal, q in agent.last_diagnostics.q_values:
                assert q == pytest.approx(q_star[legal], abs=1e-9)
            assert q_star[action] == pytest.approx(q_star[mask].max(), abs=1e-9)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 3


def test_depth_and_leaf_validation():
    with pytest.raises(ValueError):
        ExpectimaxAgent("m4", depth=0, leaf_value=lambda o: np.zeros(len(o)))
    with pytest.raises(ValueError):
        ExpectimaxAgent("m4", depth=1)
    agent = ExpectimaxAgent("m4", depth=1, leaf_value=lambda o: np.zeros(len(o) + 1))
    env = make_env("m4")
    obs, _ = env.reset(seed=3)
    with pytest.raises(ValueError, match="one value"):
        agent.choose_action(obs, env.action_masks())
    with pytest.raises(ValueError):
        agent.choose_action(obs[:-1], env.action_masks())
    wrong = env.action_masks().copy()
    wrong[-1] = False
    with pytest.raises(ValueError, match="disagrees"):
        agent.choose_action(obs, wrong)


def test_single_action_and_batched_leaves_and_turn_cache():
    calls = []

    def leaf(observations):
        calls.append(len(observations))
        return np.full(len(observations), 0.5)

    agent = ExpectimaxAgent("m4", depth=1, leaf_value=leaf)
    env = make_env("m4")
    obs, _ = env.reset(seed=4)
    end_turn = env.action_names.index("END_TURN")
    single = np.zeros(7, dtype=bool)
    single[end_turn] = True
    # A one-action mask is answered without search (the model still validates it).
    obs_after = obs.copy()
    obs_after[[2, 3, 4, 5, 6]] = 0
    assert agent.choose_action(obs_after, single) == end_turn and calls == []
    action = agent.choose_action(obs, env.action_masks())
    assert len(calls) == 1 and calls[0] == agent.last_diagnostics.leaf_evaluations > 0
    expanded = agent.last_diagnostics.expanded_nodes
    assert expanded > 0
    obs, *_ = env.step(action)
    agent.choose_action(obs, env.action_masks())
    # The follow-up decision of the same turn reuses the transposition table.
    assert agent.last_diagnostics.expanded_nodes == 0 and len(calls) == 1


def test_terminal_values_are_exact():
    agent = ExpectimaxAgent("m3", depth=1, leaf_value=lambda o: np.full(len(o), 0.25))
    env = make_env("m3")
    obs, _ = env.reset(seed=5)
    obs[13], obs[19] = 0, 1  # one enemy left at 1 HP: any hit wins
    model = agent.model_for(obs)
    agent.choose_action(obs, model.legal_mask(model.state_from_observation(obs)))
    q = dict(agent.last_diagnostics.q_values)
    hit_probability = sum(p for p, s in model.outcomes(model.state_from_observation(obs), 1)
                          if model.status(s) is Status.WIN)
    # Hit and win (exact 1), or Surge and attack again, else hand over to a 0.25 leaf.
    h = hit_probability
    assert 0 < h < 1
    assert q[1] == pytest.approx(h + (1 - h) * (h + (1 - h) * 0.25), abs=1e-12)
    assert q[5] == pytest.approx(0.25)  # END_TURN hands ally 1 a leaf-valued turn start
    assert agent.last_diagnostics.value == max(q.values())


def test_planner_plays_full_episodes_through_evaluation():
    agent = ExpectimaxAgent("m4", depth=1, leaf_value=lambda o: np.full(len(o), 0.5))
    summary = evaluate(agent, range(18000, 18003), stage="m4")
    assert summary.episodes == 3
    assert expectimax_module.ExpectimaxDiagnostics is not None


def test_node_budget_falls_back_to_shallower_depth():
    leaf = lambda o: np.full(len(o), 0.5)
    env = make_env("m4")
    obs, _ = env.reset(seed=6)
    mask = env.action_masks()
    tiny = ExpectimaxAgent("m4", depth=2, leaf_value=leaf, node_budget=5)
    shallow = ExpectimaxAgent("m4", depth=1, leaf_value=leaf)
    assert tiny.choose_action(obs, mask) == shallow.choose_action(obs, mask)
    assert tiny.last_diagnostics.depth_used == 1
    assert tiny.last_diagnostics.q_values == shallow.last_diagnostics.q_values
    # Near the end of a fight a depth-2 search fits a modest budget.
    late = obs.copy()
    late[[13, 19, 25]] = (0, 0, 2)
    roomy = ExpectimaxAgent("m4", depth=2, leaf_value=leaf, node_budget=50_000)
    model = roomy.model_for(late)
    roomy.choose_action(late, model.legal_mask(model.state_from_observation(late)))
    assert roomy.last_diagnostics.depth_used == 2
    with pytest.raises(ValueError):
        ExpectimaxAgent("m4", depth=None, node_budget=10)
