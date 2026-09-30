"""Train MaskablePPO on the M0 Fighter-versus-Goblin encounter.

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
from typing import Any, Callable

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from combat.env import BaldurCombatEnv
from evaluation.evaluate import SB3Policy, evaluate, write_run_card


def mask_fn(env: gym.Env) -> Any:
    """Return the M0 legality mask for the env wrapped by ``ActionMasker``."""
    raw: Any = env.unwrapped
    return raw.action_masks()


def make_masked_env(seed: int) -> gym.Env:
    """Build one monitored, action-masked M0 env (top-level: Subproc-safe).

    ``MaskablePPO`` queries ``action_masks()`` on each sub-env every rollout
    step via the VecEnv mask path (no ``info`` key in this sb3-contrib).
    """
    env = BaldurCombatEnv()
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env(n_envs: int, seed: int, vec: str) -> VecEnv:
    """Build a vectorized masked M0 env with per-worker seeds."""
    thunks: list[Callable[[], gym.Env]] = [
        partial(make_masked_env, seed + rank) for rank in range(n_envs)
    ]
    if vec == "subproc":
        return SubprocVecEnv(thunks)
    return DummyVecEnv(thunks)


class M0EvalCallback(BaseCallback):
    """Periodically score the policy with the M0 eval harness.

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
    ) -> None:
        super().__init__(verbose)
        self.eval_seeds = eval_seeds
        self.eval_freq = eval_freq
        self.best_model_path = best_model_path
        self.best_win_rate = -1.0
        self.last_win_rate = 0.0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        summary = evaluate(
            SB3Policy(self.model, deterministic=True), self.eval_seeds
        )
        self.last_win_rate = summary.win_rate
        self.logger.record("m0_eval/win_rate", summary.win_rate)
        self.logger.record("m0_eval/mean_rounds", summary.mean_rounds)
        self.logger.record("m0_eval/mean_fighter_hp", summary.mean_fighter_hp)
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
    parser = ArgumentParser(description="Train MaskablePPO on the M0 encounter")
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
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--save-dir", type=str, default="runs/m0_ppo")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args(argv)


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

    vec_env = make_vec_env(args.n_envs, args.seed, args.vec)
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
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
        save_path=str(checkpoints_dir),
        name_prefix="ppo_m0",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    model.save(str(save_dir / "final_model.zip"))

    trace_path = save_dir / "final_eval_traces.jsonl"
    summary = evaluate(SB3Policy(model, deterministic=True), eval_seeds, trace_path=trace_path)
    payload = summary.as_dict()
    with open(save_dir / "final_eval.json", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    write_run_card(
        save_dir / "final_run_card.json",
        policy="ppo",
        summary=summary,
        trace_file=trace_path.name,
        model_path=str(save_dir / "final_model.zip"),
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
