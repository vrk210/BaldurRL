"""Test helpers: drive the real environments to check exact transition enumeration."""

from collections import defaultdict
from math import prod

import numpy as np

from combat.actors import ActorRef, Side
from combat.resources import Resource
from combat.stages import make_env
from combat.transitions import RosterTransitionModel, State
from combat.turns import TurnManager


def model_for(stage: str, seed: int) -> tuple:
    """A live environment plus a model bound to a separate copy of its encounter."""
    envs = []
    for _ in range(2):
        env = make_env(stage)
        env.reset(seed=seed)
        envs.append(env)
    return envs[0], RosterTransitionModel(envs[1])


def load_state(env, state: State) -> None:
    """Write an ongoing compact state into a live environment of the same encounter."""
    roster = env._roster()
    width = 1 + len(Resource)
    for slot, ally in enumerate(roster.allies):
        base = 3 + width * slot
        ally.hp = state[base]
        for index, resource in enumerate(Resource):
            ally.resources.set(resource, state[base + 1 + index])
    enemy_base = 3 + width * len(roster.allies)
    for slot, enemy in enumerate(roster.enemies):
        enemy.hp = state[enemy_base + slot]
    env.round_number = state[2]
    env.turns = TurnManager(env._turns().order, state[2], current=ActorRef(Side.ALLY, state[1]))
    env._terminated = False
    env._truncated = False


class ScriptedRng:
    """Returns a prescribed prefix of rolls, then each draw's lowest value."""

    def __init__(self, prefix: list[int]) -> None:
        self.prefix = prefix
        self.draws: list[tuple[int, int, int]] = []

    def integers(self, low: int, high: int) -> np.int64:
        index = len(self.draws)
        value = self.prefix[index] if index < len(self.prefix) else low
        assert low <= value < high
        self.draws.append((value, low, high))
        return np.int64(value)


def exhaustive_distribution(env, model: RosterTransitionModel, state: State, action: int) -> dict[State, float]:
    """Enumerate every RNG path through the real `env.step` (odometer order)."""
    results: dict[State, float] = defaultdict(float)
    prefix: list[int] = []
    while True:
        load_state(env, state)
        rng = ScriptedRng(prefix)
        env.np_random = rng
        env.step(action)
        results[model.state_from_env(env)] += prod(1.0 / (high - low) for _, low, high in rng.draws)
        draws = rng.draws
        position = len(draws) - 1
        while position >= 0 and draws[position][0] == draws[position][2] - 1:
            position -= 1
        if position < 0:
            return dict(results)
        prefix = [value for value, _, _ in draws[:position]] + [draws[position][0] + 1]


def sampled_counts(env, model: RosterTransitionModel, state: State, action: int, samples: int, seed: int) -> dict[State, int]:
    counts: dict[State, int] = defaultdict(int)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        load_state(env, state)
        env.np_random = rng
        env.step(action)
        counts[model.state_from_env(env)] += 1
    return dict(counts)


def max_abs_z(counts: dict[State, int], expected: dict[State, float], samples: int) -> float:
    """Largest per-outcome deviation in binomial standard errors (with a +3 count slack)."""
    worst = 0.0
    for key in set(counts) | set(expected):
        p = expected.get(key, 0.0)
        deviation = abs(counts.get(key, 0) - samples * p)
        sd = np.sqrt(samples * p * (1.0 - p))
        worst = max(worst, max(0.0, deviation - 3.0) / max(sd, 1e-12) if deviation > 3.0 else 0.0)
    return worst
