"""Evaluate policies on fixed M0 environment seeds.

Train/eval seed-split convention: train on random or otherwise non-overlapping
seeds; evaluate on a fixed held-out seed list (e.g. ``range(1000, 2000)``) so
runs are comparable. The agent/policy RNG (e.g. ``RandomAgent(seed=...)``) is
independent of the environment episode seeds passed to :func:`evaluate`.
"""

import json
from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Protocol

import numpy as np

from combat.actions import Action
from combat.env import M0_ACTIONS, OBSERVATION_FIELDS, BaldurCombatEnv


class Policy(Protocol):
    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int: ...


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    won: bool
    lost: bool
    truncated: bool
    rounds: int
    fighter_hp: int
    action_counts: tuple[int, ...]  # M0_ACTIONS order

    @property
    def used_second_wind(self) -> bool:
        return self.action_counts[M0_ACTIONS.index(Action.SECOND_WIND)] > 0

    @property
    def used_action_surge(self) -> bool:
        return self.action_counts[M0_ACTIONS.index(Action.ACTION_SURGE)] > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "won": self.won,
            "lost": self.lost,
            "truncated": self.truncated,
            "rounds": self.rounds,
            "fighter_hp": self.fighter_hp,
            "action_counts": list(self.action_counts),
        }


@dataclass(frozen=True)
class EvaluationSummary:
    results: tuple[EpisodeResult, ...]

    @property
    def episodes(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(result.won for result in self.results)

    @property
    def losses(self) -> int:
        return sum(result.lost for result in self.results)

    @property
    def truncations(self) -> int:
        return sum(result.truncated for result in self.results)

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes

    @property
    def win_rate_se(self) -> float:
        rate = self.win_rate
        return sqrt(rate * (1.0 - rate) / self.episodes)

    @property
    def mean_rounds(self) -> float:
        return mean(result.rounds for result in self.results)

    @property
    def mean_rounds_std(self) -> float:
        if self.episodes < 2:
            return 0.0
        return pstdev(result.rounds for result in self.results)

    @property
    def mean_fighter_hp(self) -> float:
        return mean(result.fighter_hp for result in self.results)

    @property
    def mean_fighter_hp_std(self) -> float:
        if self.episodes < 2:
            return 0.0
        return pstdev(result.fighter_hp for result in self.results)

    @property
    def action_counts(self) -> tuple[int, ...]:
        return tuple(sum(result.action_counts[index] for result in self.results) for index in range(len(M0_ACTIONS)))

    @property
    def second_wind_use_rate(self) -> float:
        return sum(result.used_second_wind for result in self.results) / self.episodes

    @property
    def action_surge_use_rate(self) -> float:
        return sum(result.used_action_surge for result in self.results) / self.episodes

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "truncations": self.truncations,
            "win_rate": self.win_rate,
            "win_rate_se": self.win_rate_se,
            "mean_rounds": self.mean_rounds,
            "mean_rounds_std": self.mean_rounds_std,
            "mean_fighter_hp": self.mean_fighter_hp,
            "mean_fighter_hp_std": self.mean_fighter_hp_std,
            "action_counts": list(self.action_counts),
            "second_wind_use_rate": self.second_wind_use_rate,
            "action_surge_use_rate": self.action_surge_use_rate,
        }


class SB3Policy:
    """Adapter for a MaskablePPO-like model with an ``action_masks`` predict kwarg.

    ``model`` must expose ``predict(obs, action_masks=mask, deterministic=...)``
    returning either ``(action, state)`` or a bare action. No stable-baselines3
    import is required; duck-typing keeps this module importable without SB3.
    """

    def __init__(self, model: Any, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        result = self.model.predict(
            observation, action_masks=action_mask, deterministic=self.deterministic
        )
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
        return int(np.asarray(action).item() if np.asarray(action).size == 1 else np.asarray(action).ravel()[0])


def evaluate(
    agent: Policy, episode_seeds: Iterable[int], trace_path: str | Path | None = None
) -> EvaluationSummary:
    """Run seeded episodes, optionally recording one JSON line per episode."""
    env = BaldurCombatEnv()
    results: list[EpisodeResult] = []
    output_path = Path(trace_path) if trace_path is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_context = output_path.open("w", encoding="utf-8") if output_path else nullcontext(None)
    with trace_context as trace_file:
        for seed in episode_seeds:
            observation, _ = env.reset(seed=seed)
            action_counts = [0] * len(M0_ACTIONS)
            steps: list[dict[str, Any]] = []
            terminated = truncated = False
            while not (terminated or truncated):
                action_mask = env.action_masks()
                action = agent.choose_action(observation, action_mask)
                next_observation, reward, terminated, truncated, info = env.step(action)
                action_counts[action] += 1
                if trace_file is not None:
                    steps.append({
                        "observation": observation.tolist(),
                        "action_mask": action_mask.tolist(),
                        "action_index": action,
                        "action": M0_ACTIONS[action].name,
                        "reward": reward,
                        "next_observation": next_observation.tolist(),
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    })
                observation = next_observation

            result = EpisodeResult(
                seed=seed,
                won=terminated and info["goblin_hp"] == 0,
                lost=terminated and info["fighter_hp"] == 0,
                truncated=truncated,
                rounds=info["round"],
                fighter_hp=info["fighter_hp"],
                action_counts=tuple(action_counts),
            )
            results.append(result)
            if trace_file is not None:
                trace_file.write(json.dumps({"result": result.as_dict(), "steps": steps}) + "\n")

    if not results:
        raise ValueError("At least one episode seed is required")
    return EvaluationSummary(tuple(results))


def write_run_card(
    path: str | Path,
    *,
    policy: str,
    summary: EvaluationSummary,
    trace_file: str,
    model_path: str | None = None,
    agent_seed: int | None = None,
) -> None:
    """Save enough metadata to identify and inspect an evaluation run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card = {
        "schema_version": 1,
        "policy": policy,
        "model_path": model_path,
        "agent_seed": agent_seed,
        "episode_seeds": [result.seed for result in summary.results],
        "observation_fields": list(OBSERVATION_FIELDS),
        "action_names": [action.name for action in M0_ACTIONS],
        "trace_file": trace_file,
        "summary": summary.as_dict(),
        "loss_seeds": [result.seed for result in summary.results if result.lost],
        "truncation_seeds": [result.seed for result in summary.results if result.truncated],
    }
    output_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent

    parser = ArgumentParser(description="Evaluate an M0 baseline agent")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--agent", choices=["random", "heuristic", "ppo"], default="random")
    parser.add_argument("--model", type=str, default=None, help="MaskablePPO checkpoint for --agent ppo")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    args = parser.parse_args()

    agent: Policy
    label: str
    if args.agent == "ppo":
        if args.model is None:
            parser.error("--agent ppo requires --model")
        from sb3_contrib import MaskablePPO

        agent = SB3Policy(MaskablePPO.load(args.model), deterministic=True)
        label = "MaskablePPO"
    elif args.agent == "heuristic":
        agent = HeuristicAgent()
        label = "HeuristicAgent baseline"
    else:
        agent = RandomAgent(seed=args.agent_seed)
        label = "RandomAgent baseline"

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    trace_path = args.trace_dir / "episodes.jsonl" if args.trace_dir is not None else None
    summary = evaluate(agent, seeds, trace_path=trace_path)
    if args.trace_dir is not None:
        write_run_card(
            args.trace_dir / "run_card.json",
            policy=args.agent,
            summary=summary,
            trace_file="episodes.jsonl",
            model_path=args.model,
            agent_seed=args.agent_seed if args.agent == "random" else None,
        )

    if args.format == "json":
        payload = json.dumps(summary.as_dict(), indent=2)
        if args.output is not None:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        else:
            print(payload)
        return

    lines = [
        label,
        f"Episodes:                 {summary.episodes}",
        f"Wins:                     {summary.wins}",
        f"Losses:                   {summary.losses}",
        f"Truncations:              {summary.truncations}",
        f"Win rate:                 {summary.win_rate:.2%} (SE {summary.win_rate_se:.2%})",
        f"Mean rounds:              {summary.mean_rounds:.2f} (SD {summary.mean_rounds_std:.2f})",
        f"Mean final Fighter HP:    {summary.mean_fighter_hp:.2f} (SD {summary.mean_fighter_hp_std:.2f})",
        f"Second Wind used:         {summary.second_wind_use_rate:.2%}",
        f"Action Surge used:        {summary.action_surge_use_rate:.2%}",
        "",
        "Action selections:",
    ]
    for action, count in zip(M0_ACTIONS, summary.action_counts):
        lines.append(f"{action.name + ':':<25}{count}")
    text = "\n".join(lines)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
