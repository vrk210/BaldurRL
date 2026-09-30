"""Masked-env wrapper and tiny MaskablePPO training smoke tests."""

import json

import numpy as np
import pytest

sb3_contrib = pytest.importorskip("sb3_contrib")
stable_baselines3 = pytest.importorskip("stable_baselines3")

from train import M0EvalCallback, make_masked_env, make_vec_env, parse_args, train  # noqa: E402


def test_masked_env_reports_matching_masks() -> None:
    env = make_masked_env(seed=0)
    assert env.action_space == env.unwrapped.action_space
    assert env.observation_space == env.unwrapped.observation_space
    env.reset(seed=0)

    steps = 0
    while steps < 500:
        mask = env.get_wrapper_attr("action_masks")()
        assert mask.tolist() == env.unwrapped.action_masks().tolist()
        assert mask.any()
        _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[0]))
        steps += 1
        if terminated or truncated:
            env.reset()
    env.close()


def test_vec_env_masks_match_maskableppo_path() -> None:
    from sb3_contrib.common.maskable.utils import get_action_masks

    vec_env = make_vec_env(n_envs=2, seed=11, vec="dummy")
    vec_env.reset()
    masks = get_action_masks(vec_env)
    assert masks.shape == (2, 4)
    expected = [
        env.get_wrapper_attr("action_masks")().tolist() for env in vec_env.envs
    ]
    assert masks.tolist() == expected
    vec_env.close()


def test_eval_callback_saves_best_model(tmp_path) -> None:
    from sb3_contrib import MaskablePPO

    best_path = tmp_path / "best_model.zip"
    vec_env = make_vec_env(n_envs=1, seed=3, vec="dummy")
    model = MaskablePPO(
        "MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, seed=3
    )
    callback = M0EvalCallback(
        eval_seeds=[0, 1], eval_freq=64, best_model_path=best_path
    )
    model.learn(total_timesteps=128, callback=callback)
    vec_env.close()
    assert best_path.exists()
    assert 0.0 <= callback.last_win_rate <= 1.0


def test_train_smoke_writes_artifacts(tmp_path) -> None:
    args = parse_args([
        "--timesteps", "128",
        "--seed", "5",
        "--n-envs", "1",
        "--n-steps", "64",
        "--batch-size", "32",
        "--n-epochs", "1",
        "--eval-episodes", "4",
        "--eval-seed-start", "1000",
        "--eval-freq", "64",
        "--checkpoint-freq", "100000",
        "--save-dir", str(tmp_path),
    ])
    payload = train(args)
    assert (tmp_path / "final_model.zip").exists()
    assert (tmp_path / "final_eval.json").exists()
    assert (tmp_path / "final_eval_traces.jsonl").exists()
    assert (tmp_path / "final_run_card.json").exists()
    assert (tmp_path / "best_model.zip").exists()
    assert payload["episodes"] == 4
    assert 0.0 <= payload["win_rate"] <= 1.0
    assert json.loads((tmp_path / "final_eval.json").read_text()) == payload
    assert json.loads((tmp_path / "final_run_card.json").read_text())["summary"] == payload
