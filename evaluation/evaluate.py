"""Evaluate policies on fixed M0 environment seeds."""

from argparse import ArgumentParser
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Protocol

import numpy as np

from combat.actions import Action
from combat.env import M0_ACTIONS, BaldurCombatEnv


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
    def mean_rounds(self) -> float:
        return mean(result.rounds for result in self.results)

    @property
    def mean_fighter_hp(self) -> float:
        return mean(result.fighter_hp for result in self.results)

    @property
    def action_counts(self) -> tuple[int, ...]:
        return tuple(sum(result.action_counts[index] for result in self.results) for index in range(len(M0_ACTIONS)))

    @property
    def second_wind_use_rate(self) -> float:
        return sum(result.used_second_wind for result in self.results) / self.episodes

    @property
    def action_surge_use_rate(self) -> float:
        return sum(result.used_action_surge for result in self.results) / self.episodes


def evaluate(agent: Policy, episode_seeds: Iterable[int]) -> EvaluationSummary:
    """Run one episode per seed using the agent's independent action RNG."""
    env = BaldurCombatEnv()
    results: list[EpisodeResult] = []
    for seed in episode_seeds:
        observation, _ = env.reset(seed=seed)
        action_counts = [0] * len(M0_ACTIONS)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.choose_action(observation, env.action_masks())
            observation, _, terminated, truncated, info = env.step(action)
            action_counts[action] += 1

        results.append(EpisodeResult(
            seed=seed,
            won=terminated and info["goblin_hp"] == 0,
            lost=terminated and info["fighter_hp"] == 0,
            truncated=truncated,
            rounds=info["round"],
            fighter_hp=info["fighter_hp"],
            action_counts=tuple(action_counts),
        ))

    if not results:
        raise ValueError("At least one episode seed is required")
    return EvaluationSummary(tuple(results))


def main() -> None:
    from agents.random_agent import RandomAgent

    parser = ArgumentParser(description="Evaluate the M0 RandomAgent baseline")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--agent-seed", type=int, default=0)
    args = parser.parse_args()

    summary = evaluate(RandomAgent(seed=args.agent_seed), range(args.episodes))
    print("RandomAgent baseline")
    print(f"Episodes:                 {summary.episodes}")
    print(f"Wins:                     {summary.wins}")
    print(f"Losses:                   {summary.losses}")
    print(f"Truncations:              {summary.truncations}")
    print(f"Win rate:                 {summary.win_rate:.2%}")
    print(f"Mean rounds:              {summary.mean_rounds:.2f}")
    print(f"Mean final Fighter HP:    {summary.mean_fighter_hp:.2f}")
    print(f"Second Wind used:         {summary.second_wind_use_rate:.2%}")
    print(f"Action Surge used:        {summary.action_surge_use_rate:.2%}")
    print("\nAction selections:")
    for action, count in zip(M0_ACTIONS, summary.action_counts):
        print(f"{action.name + ':':<25}{count}")


if __name__ == "__main__":
    main()
