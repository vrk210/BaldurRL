"""Train MaskablePPO on a selected Fighter combat stage.

Train/eval seed-split convention (see ``evaluation/evaluate.py``): training
uses seeds derived from ``--seed`` (one per sub-env); final and periodic
evaluation runs on the fixed held-out list
``range(--eval-seed-start, --eval-seed-start + --eval-episodes)`` so runs are
comparable against the random/heuristic baselines.
"""

import json
import warnings
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from combat.stages import make_env
from evaluation.evaluate import SB3Policy, evaluate, write_run_card


def make_masked_env(
    seed: int, stage: str = "m0", reward_mode: str = "terminal"
) -> gym.Env:
    """Build one monitored env with native masks (top-level: Subproc-safe).

    ``MaskablePPO`` queries ``action_masks()`` on each sub-env every rollout
    step via the VecEnv mask path (no ``info`` key in this sb3-contrib).
    """
    env = make_env(stage, reward_mode=reward_mode)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env(
    n_envs: int, seed: int, vec: str, stage: str = "m0", reward_mode: str = "terminal"
) -> VecEnv:
    """Build vectorized environments with per-worker seeds and native masks."""
    thunks: list[Callable[[], gym.Env]] = [
        partial(make_masked_env, seed + rank, stage, reward_mode) for rank in range(n_envs)
    ]
    if vec == "subproc":
        return SubprocVecEnv(thunks)
    return DummyVecEnv(thunks)


class M0EvalCallback(BaseCallback):
    """Periodically score the policy with the stage's eval harness.

    Uses :func:`evaluation.evaluate.evaluate` with :class:`SB3Policy`, so
    masks are handled exactly as in baseline evaluation. Saves the model
    whenever held-out win rate strictly improves.
    """

    def __init__(
        self,
        eval_seeds: list[int],
        eval_freq: int,
        best_model_path: Path,
        verbose: int = 0,
        stage: str = "m0",
    ) -> None:
        super().__init__(verbose)
        self.eval_seeds = eval_seeds
        self.eval_freq = eval_freq
        self.best_model_path = best_model_path
        self.stage = stage
        self.best_win_rate = -1.0
        self.last_win_rate = 0.0
        self.next_eval_timestep = eval_freq if eval_freq > 0 else None

    def _on_step(self) -> bool:
        if self.next_eval_timestep is None or self.num_timesteps < self.next_eval_timestep:
            return True
        # A vectorized step may cross a boundary. Evaluate once and advance
        # past all crossed thresholds to avoid duplicate evaluations.
        self.next_eval_timestep = (
            self.num_timesteps // self.eval_freq + 1
        ) * self.eval_freq
        summary = evaluate(
            SB3Policy(self.model, deterministic=True), self.eval_seeds, stage=self.stage
        )
        self.last_win_rate = summary.win_rate
        self.logger.record(f"{self.stage}_eval/win_rate", summary.win_rate)
        self.logger.record(f"{self.stage}_eval/mean_rounds", summary.mean_rounds)
        self.logger.record(f"{self.stage}_eval/mean_fighter_hp", summary.mean_fighter_hp)
        if summary.win_rate > self.best_win_rate:
            self.best_win_rate = summary.win_rate
            self.model.save(str(self.best_model_path))
            if self.verbose:
                print(
                    f"Step {self.num_timesteps}: new best win rate "
                    f"{summary.win_rate:.2%} -> {self.best_model_path}"
                )
        return True


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Train MaskablePPO on a combat stage")
    parser.add_argument("--stage", choices=["m0", "m1a", "m1b", "m2", "m3", "m4"], default="m0")
    parser.add_argument("--reward", choices=["terminal", "damage"], default="terminal")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)
    if args.reward == "damage" and args.stage != "m2":
        parser.error("--reward damage is available only for --stage m2")
    if args.save_dir is None:
        suffix = "_damage" if args.reward == "damage" else ""
        args.save_dir = (
            "runs/m0_ppo" if args.stage == "m0"
            else f"runs/{args.stage}_ppo{suffix}_s{args.seed}"
        )
    return args


def train(args: Namespace) -> dict[str, Any]:
    """Run MaskablePPO training; returns the final eval summary dict."""
    save_dir = Path(args.save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(
        range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes)
    )

    tensorboard_log = args.tensorboard_log
    if tensorboard_log is not None:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            warnings.warn(
                "tensorboard package not installed; continuing without "
                "TensorBoard logging."
            )
            tensorboard_log = None

    vec_env = make_vec_env(args.n_envs, args.seed, args.vec, args.stage, args.reward)
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        tensorboard_log=tensorboard_log,
        device=args.device,
        verbose=args.verbose,
    )
    eval_callback = M0EvalCallback(
        eval_seeds,
        args.eval_freq,
        save_dir / "best_model.zip",
        verbose=args.verbose,
        stage=args.stage,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
        save_path=str(checkpoints_dir),
        name_prefix=f"ppo_{args.stage}",
    )
    started_at = perf_counter()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    elapsed_seconds = perf_counter() - started_at
    (save_dir / "training_metadata.json").write_text(json.dumps({
        "stage": args.stage,
        "reward_mode": args.reward,
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "actual_timesteps": model.num_timesteps,
        "training_seconds": elapsed_seconds,
    }, indent=2) + "\n", encoding="utf-8")
    model.save(str(save_dir / "final_model.zip"))

    trace_path = save_dir / "final_eval_traces.jsonl"
    summary = evaluate(
        SB3Policy(model, deterministic=True), eval_seeds,
        trace_path=trace_path, stage=args.stage,
    )
    payload = summary.as_dict()
    with open(save_dir / "final_eval.json", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    write_run_card(
        save_dir / "final_run_card.json",
        policy="ppo",
        summary=summary,
        trace_file=trace_path.name,
        model_path=str(save_dir / "final_model.zip"),
        stage=args.stage,
    )
    vec_env.close()
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = train(args)
    print(f"Episodes:              {payload['episodes']}")
    print(f"Wins:                  {payload['wins']}")
    print(f"Win rate:              {payload['win_rate']:.2%} (SE {payload['win_rate_se']:.2%})")
    print(f"Mean rounds:           {payload['mean_rounds']:.2f}")
    print(f"Mean final Fighter HP: {payload['mean_fighter_hp']:.2f}")
    print(f"Artifacts:             {args.save_dir}")


if __name__ == "__main__":
    main()
