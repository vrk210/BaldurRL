"""Stage-aware policy and evaluation contracts."""

import json

import numpy as np
import pytest

from agents.heuristic_agent import M1AHeuristicAgent, M1BHeuristicAgent, M2HeuristicAgent, make_heuristic_agent
from agents.random_agent import RandomAgent
from combat.stages import make_env
from evaluation.evaluate import evaluate, write_run_card


@pytest.mark.parametrize(
    ("stage", "action_count", "observation_count"),
    [("m0", 4, 7), ("m1a", 4, 12), ("m1b", 5, 18), ("m2", 6, 19), ("m3", 6, 26), ("m4", 7, 32),
     ("m5", 11, 63), ("m6", 13, 75)],
)
def test_stage_metadata_and_policies_match_spaces(stage: str, action_count: int, observation_count: int) -> None:
    env = make_env(stage)
    observation, _ = env.reset(seed=17)
    names = tuple(env.action_names)
    fields = tuple(env.observation_fields)
    assert len(names) == env.action_space.n == action_count
    assert len(fields) == len(observation) == observation_count
    assert len(env.decisions) == action_count
    for agent in (RandomAgent(seed=1), make_heuristic_agent(stage)):
        for _ in range(20):
            mask = env.action_masks()
            action = agent.choose_action(observation, mask)
            assert mask[action]
            observation, _, ended, truncated, _ = env.step(action)
            if ended or truncated:
                observation, _ = env.reset(seed=17)
    env.close()


def test_stage_heuristics_choose_documented_priorities() -> None:
    assert M1AHeuristicAgent().choose_action(
        np.array([10, 1, 1, 1, 1, 15, 15, 15, 4, 6, 2, 1], dtype=np.float32),
        np.array([True, True, True, True]),
    ) == 1
    m1b_obs = np.array([20, 1, 1, 1, 1, 9, 12, 14, 4, 6, 2, 4, 12, 14, 4, 6, 2, 1], dtype=np.float32)
    assert M1BHeuristicAgent().choose_action(m1b_obs, np.array([True, True, False, True, True])) == 1
    m2_obs = np.insert(m1b_obs, 5, 1)
    assert M2HeuristicAgent().choose_action(m2_obs, np.array([True, True, True, False, True, True])) == 2
    assert M2HeuristicAgent().choose_action(m2_obs, np.array([True, True, False, False, True, True])) == 1


@pytest.mark.parametrize("stage", ["m1a", "m1b", "m2", "m3", "m4", "m5", "m6"])
def test_evaluation_reproducible_and_stage_traces(stage: str, tmp_path) -> None:
    seeds = [31, 32, 33]
    first = evaluate(make_heuristic_agent(stage), seeds, trace_path=tmp_path / "episodes.jsonl", stage=stage)
    second = evaluate(make_heuristic_agent(stage), seeds, stage=stage)
    assert first == second
    assert first.wins + first.losses + first.truncations == len(seeds)
    assert len(first.action_counts) == make_env(stage).action_space.n
    write_run_card(tmp_path / "run_card.json", policy="heuristic", summary=first, trace_file="episodes.jsonl", stage=stage)
    card = json.loads((tmp_path / "run_card.json").read_text())
    assert card["stage"] == stage
    traces = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    for trace in traces:
        for step in trace["steps"]:
            assert step["action_mask"][step["action_index"]]
            assert step["action"] == card["action_names"][step["action_index"]]
            assert "semantic_action" in step and "target_index" in step
    if stage in ("m1b", "m2", "m3", "m4", "m5", "m6"):
        assert "target_attack_counts" in first.as_dict()
        assert "first_kill_rates" in first.as_dict()
    if stage in ("m2", "m3", "m4", "m5", "m6"):
        assert "cleave_use_rate" in first.as_dict()
        assert "mean_living_enemies_at_cleave" in first.as_dict()
    # Role-aware kill order is reported only for stages whose env reports roles.
    assert ("first_kill_role_rates" in first.as_dict()) == (stage in ("m5", "m6"))
    for result in first.results:
        if stage in ("m5", "m6"):
            assert set(result.kill_order_roles) <= {"BRUTE", "ARCHER", "HEALER"}
            assert len(result.kill_order_roles) == 3 if result.won else len(result.kill_order_roles) < 3
        else:
            assert result.kill_order_roles is None and "kill_order_roles" not in result.as_dict()
