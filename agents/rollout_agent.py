"""One-step lookahead scored by complete independent continuation episodes."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from combat.simulation import simulation_from_observation
from combat.stages import make_env

from .heuristic_agent import make_heuristic_agent
from .policy import Policy


PolicyFactory = Callable[[int], Policy]


@dataclass(frozen=True)
class RolloutDiagnostics:
    """Scores contain legal action/probability pairs; decisions include first actions."""

    scores: tuple[tuple[int, float], ...]
    simulated_episodes: int
    simulated_decisions: int
    planning_seconds: float


def _choose(policy: Policy, observation: np.ndarray, mask: np.ndarray) -> int:
    # Keep potentially stateful/custom policies from modifying simulator inputs.
    action = policy.choose_action(observation.copy(), mask.copy())
    if (
        isinstance(action, (bool, np.bool_))
        or not isinstance(action, (int, np.integer))
        or not 0 <= action < len(mask)
        or not mask[action]
    ):
        raise ValueError(f"Continuation policy selected masked or invalid action {action!r}")
    return int(action)


class RolloutAgent:
    """Estimate wins for every legal first action; fixed policies do no training.

    Factories must return fresh policy state for each seed, independently of any
    live policy. Stateless policies may share immutable weights. This contract
    does not restore live recurrent history; such history requires a factory
    that explicitly provides an isolated, correctly initialized continuation.
    """

    def __init__(
        self,
        stage: str = "m0",
        *,
        continuation_factory: PolicyFactory | None = None,
        rollouts_per_action: int = 32,
        seed: int | None = None,
    ) -> None:
        if isinstance(rollouts_per_action, bool) or not isinstance(rollouts_per_action, (int, np.integer)) or rollouts_per_action < 1:
            raise ValueError("rollouts_per_action must be a positive integer")
        env = make_env(stage)
        self.action_count = env.action_space.n
        self.observation_shape = env.observation_space.shape
        env.close()
        self.stage = stage
        self.continuation_factory = (
            (lambda seed: make_heuristic_agent(stage))
            if continuation_factory is None else continuation_factory
        )
        if not callable(self.continuation_factory):
            raise TypeError("continuation_factory must be callable")
        self.rollouts_per_action = int(rollouts_per_action)
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics = RolloutDiagnostics((), 0, 0, 0.0)

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        started = perf_counter()
        observation = np.asarray(observation).copy()
        mask = np.asarray(action_mask, dtype=np.bool_).copy()
        if observation.shape != self.observation_shape or mask.shape != (self.action_count,):
            raise ValueError(f"Observation/mask shape does not match {self.stage}")
        legal = np.flatnonzero(mask)
        if not legal.size:
            raise ValueError("No legal actions available")
        if legal.size == 1:
            self.last_diagnostics = RolloutDiagnostics((), 0, 0, perf_counter() - started)
            return int(legal[0])

        # Common random numbers across candidates; never derive seeds from combat.
        seeds = self.rng.integers(0, 2**63 - 1, size=(self.rollouts_per_action, 2))
        wins: dict[int, int] = {}
        decisions = 0
        for candidate in legal:
            action = int(candidate)
            wins[action] = 0
            for simulation_seed, policy_seed in seeds:
                env = simulation_from_observation(self.stage, observation, seed=int(simulation_seed))
                try:
                    policy = self.continuation_factory(int(policy_seed))
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    decisions += 1
                    while not (terminated or truncated):
                        continuation = _choose(policy, next_obs, env.action_masks())
                        next_obs, reward, terminated, truncated, _ = env.step(continuation)
                        decisions += 1
                    wins[action] += int(terminated and reward == 1.0)
                finally:
                    env.close()
        best = max(wins.values())
        tied = [action for action, count in wins.items() if count == best]
        choice = tied[0]
        if len(tied) > 1:
            base_seed = int(self.rng.integers(0, 2**63 - 1))
            base_choice = _choose(self.continuation_factory(base_seed), observation, mask)
            if base_choice in tied:
                choice = base_choice
        self.last_diagnostics = RolloutDiagnostics(
            tuple((action, count / self.rollouts_per_action) for action, count in wins.items()),
            len(legal) * self.rollouts_per_action,
            decisions,
            perf_counter() - started,
        )
        return choice
