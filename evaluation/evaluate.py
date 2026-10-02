"""Seeded, policy-agnostic evaluation of the staged combat environments."""

import json
from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Protocol

import numpy as np

from combat.stages import make_env


class Policy(Protocol):
    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int: ...


_M0_NAMES = ("ATTACK", "SECOND_WIND", "ACTION_SURGE", "END_TURN")


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    won: bool
    lost: bool
    truncated: bool
    rounds: int
    fighter_hp: int
    action_counts: tuple[int, ...]
    action_names: tuple[str, ...] = _M0_NAMES
    first_kill_slot: int | None = None
    cleave_living_counts: tuple[int, ...] = ()

    def _used(self, name: str) -> bool:
        return name in self.action_names and self.action_counts[self.action_names.index(name)] > 0

    @property
    def used_second_wind(self) -> bool:
        return self._used("SECOND_WIND")

    @property
    def used_action_surge(self) -> bool:
        return self._used("ACTION_SURGE")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": self.seed,
            "won": self.won,
            "lost": self.lost,
            "truncated": self.truncated,
            "rounds": self.rounds,
            "fighter_hp": self.fighter_hp,
            "action_counts": list(self.action_counts),
        }
        if "ATTACK_ENEMY_0" in self.action_names:
            payload["first_kill_slot"] = self.first_kill_slot
        if "CLEAVE" in self.action_names:
            payload["cleave_living_counts"] = list(self.cleave_living_counts)
        return payload


@dataclass(frozen=True)
class EvaluationSummary:
    results: tuple[EpisodeResult, ...]

    @property
    def episodes(self) -> int:
        return len(self.results)

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.results[0].action_names if self.results else _M0_NAMES

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
        return pstdev(result.rounds for result in self.results) if self.episodes > 1 else 0.0

    @property
    def mean_fighter_hp(self) -> float:
        return mean(result.fighter_hp for result in self.results)

    @property
    def mean_fighter_hp_std(self) -> float:
        return pstdev(result.fighter_hp for result in self.results) if self.episodes > 1 else 0.0

    @property
    def action_counts(self) -> tuple[int, ...]:
        return tuple(sum(result.action_counts[index] for result in self.results) for index in range(len(self.action_names)))

    @property
    def second_wind_use_rate(self) -> float:
        return sum(result.used_second_wind for result in self.results) / self.episodes

    @property
    def action_surge_use_rate(self) -> float:
        return sum(result.used_action_surge for result in self.results) / self.episodes

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if "ATTACK_ENEMY_0" in self.action_names:
            counts = self.action_counts
            payload["target_attack_counts"] = [counts[0], counts[1]]
            first_kills = [sum(result.first_kill_slot == index for result in self.results) for index in (0, 1)]
            payload["first_kill_counts"] = first_kills
            payload["first_kill_rates"] = [count / self.episodes for count in first_kills]
        if "CLEAVE" in self.action_names:
            payload["cleave_use_rate"] = sum(result._used("CLEAVE") for result in self.results) / self.episodes
            living = [count for result in self.results for count in result.cleave_living_counts]
            payload["mean_living_enemies_at_cleave"] = mean(living) if living else 0.0
        if self.action_names != _M0_NAMES:
            payload["action_use_rates"] = {
                name: sum(result._used(name) for result in self.results) / self.episodes
                for name in self.action_names
            }
        return payload


class SB3Policy:
    """Adapter for a MaskablePPO-like model using the same policy interface."""

    def __init__(self, model: Any, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        result = self.model.predict(
            observation, action_masks=action_mask, deterministic=self.deterministic
        )
        action = result[0] if isinstance(result, tuple) else result
        array = np.asarray(action)
        return int(array.item() if array.size == 1 else array.ravel()[0])


def _metadata(env: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(env.action_names), tuple(env.observation_fields)


def _decision(env: Any, index: int) -> tuple[str, int | None]:
    decision = env.decisions[index]
    return decision.action.name, decision.target_index


def _enemy_hps(info: dict[str, Any]) -> tuple[int, ...]:
    if "enemy_hps" in info:
        return tuple(int(hp) for hp in info["enemy_hps"])
    return (int(info["goblin_hp"]),)


def evaluate(
    agent: Policy,
    episode_seeds: Iterable[int],
    trace_path: str | Path | None = None,
    *,
    stage: str = "m0",
) -> EvaluationSummary:
    """Run seeded episodes; traces store every Fighter decision and diagnostic info."""
    env = make_env(stage)
    action_names, observation_fields = _metadata(env)
    if len(action_names) != env.action_space.n or len(observation_fields) != env.observation_space.shape[0]:
        raise ValueError(f"Inconsistent action or observation metadata for {stage}")
    results: list[EpisodeResult] = []
    output_path = Path(trace_path) if trace_path is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_context = output_path.open("w", encoding="utf-8") if output_path else nullcontext(None)
    try:
        with trace_context as trace_file:
            for seed in episode_seeds:
                observation, reset_info = env.reset(seed=seed)
                previous_enemy_hps = _enemy_hps(reset_info)
                first_kill_slot: int | None = None
                cleave_living_counts: list[int] = []
                action_counts = [0] * len(action_names)
                steps: list[dict[str, Any]] = []
                terminated = truncated = False
                info = reset_info
                while not (terminated or truncated):
                    action_mask = env.action_masks()
                    action = agent.choose_action(observation, action_mask)
                    if not isinstance(action, (int, np.integer)) or not (0 <= action < len(action_names)) or not action_mask[action]:
                        raise ValueError(f"Policy selected masked or invalid action {action!r}")
                    semantic_action, target_index = _decision(env, int(action))
                    if semantic_action == "CLEAVE":
                        cleave_living_counts.append(sum(hp > 0 for hp in previous_enemy_hps))
                    next_observation, reward, terminated, truncated, info = env.step(int(action))
                    current_enemy_hps = _enemy_hps(info)
                    if first_kill_slot is None and len(current_enemy_hps) > 1:
                        first_kill_slot = next((index for index, (before, after) in enumerate(zip(previous_enemy_hps, current_enemy_hps)) if before > 0 and after == 0), None)
                    previous_enemy_hps = current_enemy_hps
                    action_counts[int(action)] += 1
                    if trace_file is not None:
                        steps.append({
                            "observation": observation.tolist(),
                            "action_mask": action_mask.tolist(),
                            "action_index": int(action),
                            "action": action_names[int(action)],
                            "semantic_action": semantic_action,
                            "target_index": target_index,
                            "reward": reward,
                            "next_observation": next_observation.tolist(),
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        })
                    observation = next_observation
                result = EpisodeResult(
                    seed=int(seed),
                    won=terminated and all(hp == 0 for hp in previous_enemy_hps),
                    lost=terminated and info["fighter_hp"] == 0,
                    truncated=truncated,
                    rounds=info["round"],
                    fighter_hp=info["fighter_hp"],
                    action_counts=tuple(action_counts),
                    action_names=action_names,
                    first_kill_slot=first_kill_slot,
                    cleave_living_counts=tuple(cleave_living_counts),
                )
                results.append(result)
                if trace_file is not None:
                    trace_file.write(json.dumps({"result": result.as_dict(), "steps": steps}) + "\n")
    finally:
        env.close()
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
    stage: str = "m0",
) -> None:
    """Save metadata needed to inspect and compare a traced evaluation run."""
    env = make_env(stage)
    action_names, observation_fields = _metadata(env)
    env.close()
    if action_names != summary.action_names:
        raise ValueError("Summary action names do not match the requested stage")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card = {
        "schema_version": 2,
        "stage": stage,
        "policy": policy,
        "model_path": model_path,
        "agent_seed": agent_seed,
        "episode_seeds": [result.seed for result in summary.results],
        "observation_fields": list(observation_fields),
        "action_names": list(action_names),
        "trace_file": trace_file,
        "summary": summary.as_dict(),
        "loss_seeds": [result.seed for result in summary.results if result.lost],
        "truncation_seeds": [result.seed for result in summary.results if result.truncated],
    }
    output_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from agents.heuristic_agent import make_heuristic_agent
    from agents.random_agent import RandomAgent

    parser = ArgumentParser(description="Evaluate a combat policy")
    parser.add_argument("--stage", choices=["m0", "m1a", "m1b", "m2"], default="m0")
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
    if args.agent == "ppo":
        if args.model is None:
            parser.error("--agent ppo requires --model")
        from sb3_contrib import MaskablePPO
        agent = SB3Policy(MaskablePPO.load(args.model), deterministic=True)
    elif args.agent == "heuristic":
        agent = make_heuristic_agent(args.stage)
    else:
        agent = RandomAgent(seed=args.agent_seed)

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    trace_path = args.trace_dir / "episodes.jsonl" if args.trace_dir is not None else None
    summary = evaluate(agent, seeds, trace_path=trace_path, stage=args.stage)
    if args.trace_dir is not None:
        write_run_card(
            args.trace_dir / "run_card.json",
            policy=args.agent,
            summary=summary,
            trace_file="episodes.jsonl",
            model_path=args.model,
            agent_seed=args.agent_seed if args.agent == "random" else None,
            stage=args.stage,
        )
    if args.format == "json":
        output = json.dumps(summary.as_dict(), indent=2)
    else:
        payload = summary.as_dict()
        lines = [
            f"{args.stage.upper()} {args.agent}",
            f"Episodes:                 {summary.episodes}",
            f"Wins:                     {summary.wins}",
            f"Losses:                   {summary.losses}",
            f"Truncations:              {summary.truncations}",
            f"Win rate:                 {summary.win_rate:.2%} (SE {summary.win_rate_se:.2%})",
            f"Mean rounds:              {summary.mean_rounds:.2f} (SD {summary.mean_rounds_std:.2f})",
            f"Mean final Fighter HP:    {summary.mean_fighter_hp:.2f} (SD {summary.mean_fighter_hp_std:.2f})",
            f"Second Wind used:         {summary.second_wind_use_rate:.2%}",
            f"Action Surge used:        {summary.action_surge_use_rate:.2%}",
        ]
        for key in ("target_attack_counts", "first_kill_counts", "cleave_use_rate", "mean_living_enemies_at_cleave"):
            if key in payload:
                lines.append(f"{key}: {payload[key]}")
        lines.extend(["", "Action selections:"])
        lines.extend(f"{name + ':':<25}{count}" for name, count in zip(summary.action_names, summary.action_counts))
        output = "\n".join(lines)
    if args.output is not None:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
