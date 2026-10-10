"""Leaf value functions, value-training utilities, threat-aware agent, planning evaluation."""

from collections import defaultdict
from functools import lru_cache

import numpy as np
import pytest

from agents.heuristic_agent import M4ThreatAwareAgent
from agents.value_functions import MLPValue, RaceValue, make_layout, value_features
from combat.stages import make_env
from evaluation.evaluate import evaluate
from evaluation.planning_eval import paired_difference, run_chunk, summarize
from value_training import collect, lambda_returns, save_value, targets_for


def m4_obs(seed=0):
    env = make_env("m4")
    obs, _ = env.reset(seed=seed)
    return env, obs


def test_race_value_is_a_monotone_probability():
    _, obs = m4_obs(1)
    race = RaceValue("m4", scale=3.0, bias=0.0)
    weaker_enemy, hurt_ally = obs.copy(), obs.copy()
    weaker_enemy[13] = 1
    hurt_ally[1] = 3
    values = race(np.stack([obs, weaker_enemy, hurt_ally]))
    assert np.all((values > 0) & (values < 1))
    assert values[1] > values[0] > values[2]
    for stage in ("m0", "m1a", "m1b", "m2", "m3"):
        env = make_env(stage)
        stage_obs, _ = env.reset(seed=2)
        assert 0 < RaceValue(stage)(stage_obs[None])[0] < 1


def test_mlp_value_round_trip_and_canonical_inputs(tmp_path):
    rng = np.random.default_rng(0)
    layout = make_layout("m4")
    weights = [(rng.normal(size=(32, 8)), rng.normal(size=8)), (rng.normal(size=(8, 1)), rng.normal(size=1))]
    path = tmp_path / "v.npz"
    save_value(str(path), weights, dict(stage="m4", layers=2, canonicalize=True))
    value = MLPValue.load(path)
    _, obs = m4_obs(3)
    expected_hidden = np.maximum(value_features(layout, obs[None]) @ weights[0][0] + weights[0][1], 0)
    expected = 1 / (1 + np.exp(-(expected_hidden @ weights[1][0] + weights[1][1])[:, 0]))
    assert value(obs[None]) == pytest.approx(expected)
    # Inactive ally per-turn counts are refreshed (set) before that ally acts again.
    changed = obs.copy()
    changed[8], changed[9] = 0, 0  # ally 1 action / bonus action while ally 0 is active
    assert value(changed[None]) == pytest.approx(value(obs[None]))
    active_changed = obs.copy()
    active_changed[2] = 0
    assert not np.allclose(value_features(layout, active_changed[None]), value_features(layout, obs[None]))
    assert np.all(value_features(layout, obs[None]) <= 1.0)


def test_lambda_returns_and_target_modes():
    search = np.array([0.5, 0.6, np.nan, 0.2, 0.3], dtype=np.float32)
    outcome = np.array([1, 1, 1, 0, 0], dtype=np.int8)
    seeds = np.array([7, 7, 7, 8, 8])
    lam = 0.5
    # Episode 7: G2 = 0.5*1 + 0.5*1 (missing search -> outcome) = 1; G1 = 0.5*0.6 + 0.5*1; G0 = 0.5*0.5 + 0.5*G1.
    g1 = 0.5 * 0.6 + 0.5 * 1.0
    expected = [0.5 * 0.5 + 0.5 * g1, g1, 1.0, 0.5 * 0.2 + 0.5 * 0.15, 0.15]
    assert lambda_returns(search.astype(np.float64), outcome.astype(np.float64), seeds, lam) == pytest.approx(expected)
    data = dict(search_value=search, win=outcome, seed=seeds)
    assert targets_for(data, "outcome") == pytest.approx(outcome)
    assert targets_for(data, "search") == pytest.approx([0.5, 0.6, 1.0, 0.2, 0.3])
    assert targets_for(data, "lambda:1") == pytest.approx(outcome)
    with pytest.raises(ValueError):
        targets_for(data, "bogus")


def test_collect_labels_every_decision_with_the_episode_outcome():
    data = collect("m4", M4ThreatAwareAgent(), range(500, 503))
    summary = evaluate(M4ThreatAwareAgent(), range(500, 503), stage="m4")
    assert len(data["win"]) == sum(sum(r.action_counts) for r in summary.results)
    for result in summary.results:
        labels = set(data["win"][data["seed"] == result.seed])
        assert labels == {int(result.won)}
    assert np.all(np.isnan(data["search_value"]))


def test_planning_eval_matches_evaluate_and_paired_difference():
    rows = run_chunk("m4", dict(name="threat"), [900, 901, 902, 903])
    summary = evaluate(M4ThreatAwareAgent(), range(900, 904), stage="m4")
    assert [row["won"] for row in rows] == [result.won for result in summary.results]
    assert summarize(rows)["wins"] == summary.wins
    a = [dict(seed=s, won=w) for s, w in zip(range(4), (1, 1, 0, 1))]
    b = [dict(seed=s, won=w) for s, w in zip(range(4), (1, 0, 0, 0))]
    diff = paired_difference(a, b)
    assert diff["delta_pp"] == pytest.approx(50.0)
    assert diff["paired_se_pp"] == pytest.approx(100 * np.std([0, 1, 0, 1], ddof=1) / 2)


# Independent restatement of the M4 policy-probe scoring (closed forms), as an oracle.
def _probe_attacks_to_kill(hp, ac):
    @lru_cache(None)
    def go(h):
        if h <= 0:
            return 0.0
        probs = defaultdict(float)
        for roll in range(1, 21):
            if roll == 20:
                for d1 in range(1, 9):
                    for d2 in range(1, 9):
                        probs[d1 + d2 + 3] += 1 / 20 / 64
            elif roll != 1 and roll + 5 >= ac:
                for d in range(1, 9):
                    probs[d + 3] += 1 / 20 / 8
            else:
                probs[0] += 1 / 20
        return (1 + sum(p * go(h - d) for d, p in probs.items() if d > 0)) / (1 - probs[0])
    return go(hp)


def _probe_score(obs, i):
    hp, _, ac, attack, die, bonus = (int(v) for v in obs[13 + 6 * i:19 + 6 * i])
    threat = np.clip((21 + attack - 16) / 20, .05, .95) * ((die + 1) / 2 + bonus) + .05 * (die + 1) / 2
    return _probe_attacks_to_kill(hp, ac) / threat


def test_threat_agent_targets_like_the_probe_formula():
    agent = M4ThreatAwareAgent()
    rng = np.random.default_rng(1)
    for seed in range(40):
        env, obs = m4_obs(seed)
        obs[[13, 19, 25]] = rng.integers(1, 13, 3)
        mask = np.array([True, True, True, False, False, False, True])
        expected = min(range(3), key=lambda i: (_probe_score(obs, i), i))
        assert agent.choose_action(obs, mask) == expected
    env, obs = m4_obs(5)
    assert agent.choose_action(obs, env.action_masks()) == 3  # Cleave with three living enemies
    obs[1] = 9
    mask = env.action_masks().copy()
    mask[4] = True
    assert agent.choose_action(obs, mask) == 4  # Second Wind at <= 10 HP


def test_ensemble_averages_member_logits(tmp_path):
    from agents.value_functions import EnsembleValue
    from evaluation.planning_eval import make_leaf

    rng = np.random.default_rng(4)
    paths = []
    for index in range(2):
        weights = [(rng.normal(size=(32, 4)), rng.normal(size=4)), (rng.normal(size=(4, 1)), rng.normal(size=1))]
        paths.append(tmp_path / f"v{index}.npz")
        save_value(str(paths[-1]), weights, dict(stage="m4", layers=2, canonicalize=True))
    members = [MLPValue.load(path) for path in paths]
    _, obs = m4_obs(2)
    ensemble = make_leaf("m4", "ens:" + ",".join(str(path) for path in paths))
    expected = 1 / (1 + np.exp(-(members[0].logits(obs[None]) + members[1].logits(obs[None])) / 2))
    assert ensemble(obs[None]) == pytest.approx(expected)
    with pytest.raises(ValueError):
        EnsembleValue([])
