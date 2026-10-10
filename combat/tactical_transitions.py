"""Exact chance-node transitions for the M5/M6 tactical stages.

The M0–M4 `RosterTransitionModel` enumerates outcomes from distribution tables.
M5/M6 rules (roles, Trip/Prone, Dodge, healing, targeting policies, ranks and
opportunity attacks) are richer, so instead of re-deriving them this model
drives the real environment code with an enumerating random source:

* `_PathRng` replays every sequence of random draws (odometer order) and
  records each path's probability, so outcomes come from the same functions
  `TacticalCombatEnv.step` uses.
* Attack resolution is collapsed to one draw per attack. While enumerating,
  `combat.tactics.resolve_attack_with_mode` and `combat.tactics.roll_attack`
  are swapped for versions that draw once from the attack's exact
  distribution. Those distributions are themselves computed by enumerating
  every die path through the original `combat.rolls` functions, so no rule is
  restated here. Callers in `combat.tactics` use only the damage dealt (and,
  for Trip, whether it hit); the other result fields are informational.
* The enemy phase after `END_TURN` runs one actor at a time through
  `TacticalCombatEnv._pass_step`, merging identical intermediate states.

States are compact hashable tuples (see `_layout` below). Every M5/M6 state at
an allied decision is fully observable, so observations convert both ways.
"""

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache

import numpy as np

from . import tactics
from .actions import Action
from .actors import ActorRef, Side
from .characters import Character, Goblin
from .damage import DamageSpec
from .resources import Resource
from .rolls import AttackRoll, ModedAttackResult, RollMode, resolve_attack_with_mode, roll_attack
from .simulation import simulation_from_observation
from .tactical_env import TACTICAL_STAGES, TacticalCombatEnv
from .transitions import Outcome, State, Status, register_transition_model
from .turns import TurnManager

_STATUSES = tuple(Status)
_STATIC_SUFFIXES = (
    "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus",
    "is_brute", "is_archer", "is_healer",
    "targets_weakest", "targets_retaliate", "targets_random", "back_rank",
)
_HUGE_HP = 1_000_000


# --------------------------------------------------------------------------- random paths
class _PathRng:
    """Replays a prefix of branch choices, then takes the first option of each new draw."""

    def __init__(self, prefix: list[int], exact: bool = False) -> None:
        self.prefix = prefix
        self.choices: list[tuple[int, int]] = []
        self.exact = exact
        self.probability: float | Fraction = Fraction(1) if exact else 1.0

    def _pick(self, options: int) -> int:
        index = len(self.choices)
        choice = self.prefix[index] if index < len(self.prefix) else 0
        self.choices.append((choice, options))
        return choice

    def integers(self, low: int, high: int) -> np.int64:
        """Uniform integer in [low, high), matching `numpy.random.Generator.integers`."""
        options = int(high) - int(low)
        if options < 1:
            raise ValueError("Empty integer range")
        self.probability = self.probability * Fraction(1, options) if self.exact else self.probability / options
        return np.int64(int(low) + self._pick(options))

    def outcome(self, distribution: tuple[tuple[object, float], ...]) -> object:
        """One weighted draw from an exact (value, probability) distribution."""
        if self.exact:
            raise TypeError("Weighted draws are not available in exact-fraction enumeration")
        value, probability = distribution[self._pick(len(distribution))]
        self.probability *= probability
        return value


def enumerate_paths(
    run: Callable[[_PathRng], object], *, exact: bool = False
) -> Iterator[tuple[float | Fraction, object]]:
    """Yield (probability, result) for every branch sequence `run` can take.

    With `exact=True`, probabilities are `Fraction`s (uniform draws only).
    """
    prefix: list[int] = []
    while True:
        rng = _PathRng(prefix, exact)
        result = run(rng)
        yield rng.probability, result
        choices = rng.choices
        position = len(choices) - 1
        while position >= 0 and choices[position][0] == choices[position][1] - 1:
            position -= 1
        if position < 0:
            return
        prefix = [choice for choice, _ in choices[:position]] + [choices[position][0] + 1]


# --------------------------------------------------------------------------- exact attack draws
def _dummy(attack_bonus: int, armor_class: int, damage: DamageSpec, hp: int) -> Character:
    return Goblin("enumeration", hp, hp, armor_class, attack_bonus, damage)


@lru_cache(maxsize=None)
def attack_outcome_distribution(
    attack_bonus: int, armor_class: int, damage: DamageSpec, mode: RollMode, divisor: int
) -> tuple[tuple[tuple[bool, int], float], ...]:
    """Exact (hit, damage before the HP cap) distribution of `rolls.resolve_attack_with_mode`."""
    merged: dict[tuple[bool, int], Fraction] = defaultdict(Fraction)

    def run(rng: _PathRng) -> tuple[bool, int]:
        attacker = _dummy(attack_bonus, 10, damage, 1)
        defender = _dummy(0, armor_class, DamageSpec(1, 4, 0), _HUGE_HP)
        result = resolve_attack_with_mode(attacker, defender, rng, mode, damage_divisor=divisor)
        return result.hit, _HUGE_HP - defender.hp

    for probability, key in enumerate_paths(run, exact=True):
        merged[key] += probability
    return tuple((key, float(p)) for key, p in sorted(merged.items()))


@lru_cache(maxsize=None)
def roll_hit_distribution(attack_bonus: int, armor_class: int, mode: RollMode) -> tuple[tuple[bool, float], ...]:
    """Exact hit/miss distribution of `rolls.roll_attack`."""
    merged: dict[bool, Fraction] = defaultdict(Fraction)
    paths = enumerate_paths(lambda rng: roll_attack(attack_bonus, armor_class, rng, mode).hit, exact=True)
    for probability, hit in paths:
        merged[hit] += probability
    return tuple((hit, float(p)) for hit, p in sorted(merged.items()))


def _exact_resolve_attack(
    attacker: Character, defender: Character, rng: _PathRng, mode: RollMode, *, damage_divisor: int = 1
) -> ModedAttackResult:
    distribution = attack_outcome_distribution(
        attacker.attack_bonus, defender.armor_class, attacker.damage, mode, damage_divisor
    )
    hit, amount = rng.outcome(distribution)
    damage = min(defender.hp, amount)
    defender.hp -= damage
    return ModedAttackResult(mode.value, (), 0, hit, False, damage, 0)


def _exact_roll_attack(attack_bonus: int, armor_class: int, rng: _PathRng, mode: RollMode = RollMode.NORMAL) -> AttackRoll:
    hit = rng.outcome(roll_hit_distribution(attack_bonus, armor_class, mode))
    return AttackRoll((), 0, 0, hit, False)


@contextmanager
def _collapsed_attack_draws() -> Iterator[None]:
    saved = (tactics.resolve_attack_with_mode, tactics.roll_attack)
    tactics.resolve_attack_with_mode = _exact_resolve_attack
    tactics.roll_attack = _exact_roll_attack
    try:
        yield
    finally:
        tactics.resolve_attack_with_mode, tactics.roll_attack = saved


# --------------------------------------------------------------------------- model
class TacticalTransitionModel:
    """Exact one-decision dynamics for one M5/M6 encounter.

    State layout: (status, current side (0 ally / 1 enemy), current slot, round,
    then per ally (hp, dodging, deep, disengaged, *resource counts), then per
    enemy (hp, prone, last attacker + 1, *resource counts except Action)).
    Enemy Action is omitted because every enemy turn refreshes it before use.
    """

    def __init__(self, env: TacticalCombatEnv) -> None:
        if env.stage not in TACTICAL_STAGES:
            raise ValueError(f"Not a tactical stage: {env.stage}")
        self.stage = env.stage
        self.env = env
        self.decisions = tuple(env.decisions)
        self.action_count = len(self.decisions)
        self._end_turn = next(i for i, d in enumerate(self.decisions) if d.action is Action.END_TURN)
        self._order = env._turns().order
        self._ally_resources = [tuple(sorted(a.resources._amounts, key=lambda r: r.value)) for a in env.allies]
        self._enemy_resources = [
            tuple(r for r in sorted(e.resources._amounts, key=lambda r: r.value) if r is not Resource.ACTION)
            for e in env.enemies
        ]
        self._enemy_has_action = [Resource.ACTION in e.resources._amounts for e in env.enemies]
        self._ally_width = [4 + len(r) for r in self._ally_resources]
        self._enemy_width = [3 + len(r) for r in self._enemy_resources]
        self._enemy_base = 4 + sum(self._ally_width)
        fields = env.observation_fields
        self._template = env._observation().copy()
        self._static_cols = np.array(
            [i for i, name in enumerate(fields) if name.endswith(_STATIC_SUFFIXES)], dtype=np.intp
        )
        self._legal_cache: dict[State, tuple[int, ...]] = {}
        self._outcome_cache: dict[tuple[State, int], tuple[Outcome, ...]] = {}
        self._step_cache: dict[State, tuple[tuple[float, bool, State], ...]] = {}

    @classmethod
    def from_observation(cls, stage: str, observation: np.ndarray) -> "TacticalTransitionModel":
        return cls(simulation_from_observation(stage, observation, seed=0))

    # ------------------------------------------------------------------ encoding
    def state_from_env(self, env: TacticalCombatEnv) -> State:
        roster = env._roster()
        if roster.side_defeated(Side.ALLY):
            status = Status.LOSS
        elif roster.side_defeated(Side.ENEMY):
            status = Status.WIN
        elif env._truncated:
            status = Status.TRUNCATED
        else:
            status = Status.ONGOING
        current = env._turns().current
        values = [int(status), int(current.side is Side.ENEMY), current.slot, env.round_number]
        if status is not Status.ONGOING:
            values[1:3] = [0, 0]
        for ally, resources in zip(env.allies, self._ally_resources):
            values.extend((ally.hp, int(ally.dodging), int(ally.deep), int(ally.disengaged)))
            values.extend(ally.resources.get(r) for r in resources)
        for enemy, resources in zip(env.enemies, self._enemy_resources):
            values.extend((enemy.hp, int(enemy.prone), 0 if enemy.last_attacker is None else enemy.last_attacker + 1))
            values.extend(enemy.resources.get(r) for r in resources)
        return tuple(values)

    def load(self, state: State) -> TacticalCombatEnv:
        """Write a compact state into the bound scratch environment."""
        env = self.env
        index = 4
        for ally, resources in zip(env.allies, self._ally_resources):
            ally.hp, dodging, deep, disengaged = state[index:index + 4]
            ally.dodging, ally.deep, ally.disengaged = bool(dodging), bool(deep), bool(disengaged)
            for offset, resource in enumerate(resources):
                ally.resources.set(resource, state[index + 4 + offset])
            index += 4 + len(resources)
        for enemy, resources, has_action in zip(env.enemies, self._enemy_resources, self._enemy_has_action):
            enemy.hp, prone, last = state[index:index + 3]
            enemy.prone = bool(prone)
            enemy.last_attacker = None if last == 0 else last - 1
            for offset, resource in enumerate(resources):
                enemy.resources.set(resource, state[index + 3 + offset])
            if has_action:
                enemy.resources.set(Resource.ACTION, 1)
            index += 3 + len(resources)
        side = Side.ENEMY if state[1] else Side.ALLY
        env.turns = TurnManager(self._order, state[3], current=ActorRef(side, state[2]))
        env.round_number = state[3]
        env._terminated = state[0] in (Status.WIN, Status.LOSS)
        env._truncated = state[0] == Status.TRUNCATED
        return env

    def state_from_observation(self, observation: np.ndarray) -> State:
        if not self.same_encounter(observation):
            raise ValueError("Observation does not belong to this encounter")
        return self.state_from_env(simulation_from_observation(self.stage, np.asarray(observation), seed=0))

    def same_encounter(self, observation: np.ndarray) -> bool:
        values = np.asarray(observation)
        return values.shape == self._template.shape and bool(
            np.array_equal(values[self._static_cols], self._template[self._static_cols])
        )

    def observations(self, states: Sequence[State]) -> np.ndarray:
        return np.stack([self.load(state)._observation() for state in states]) if states else np.zeros(
            (0,) + self._template.shape, dtype=np.float32
        )

    def observation(self, state: State) -> np.ndarray:
        return self.observations([state])[0]

    # ------------------------------------------------------------------ queries
    def status(self, state: State) -> Status:
        return _STATUSES[state[0]]

    def ends_turn(self, action: int) -> bool:
        return action == self._end_turn

    def legal_actions(self, state: State) -> tuple[int, ...]:
        if state[0] != Status.ONGOING or state[1]:
            return ()
        legal = self._legal_cache.get(state)
        if legal is None:
            legal = tuple(int(i) for i in np.flatnonzero(self.load(state).action_masks()))
            self._legal_cache[state] = legal
        return legal

    def legal_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.bool_)
        mask[list(self.legal_actions(state))] = True
        return mask

    # ------------------------------------------------------------------ transitions
    def outcomes(self, state: State, action: int) -> tuple[Outcome, ...]:
        key = (state, action)
        cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached
        if action not in self.legal_actions(state):
            raise ValueError(f"Decision {action} is not legal in this state")
        with _collapsed_attack_draws():
            if action == self._end_turn:
                result = self._end_turn_outcomes(state)
            else:
                result = self._run_paths(state, lambda env: env.step(action))
        outcomes = tuple((p, s) for s, p in result.items())
        self._outcome_cache[key] = outcomes
        return outcomes

    def _run_paths(self, state: State, act: Callable[[TacticalCombatEnv], object]) -> dict[State, float]:
        env = self.env
        merged: dict[State, float] = defaultdict(float)

        def run(rng: _PathRng) -> State:
            self.load(state)
            env.np_random = rng
            act(env)
            return self.state_from_env(env)

        for probability, successor in enumerate_paths(run):
            merged[successor] += probability
        return merged

    def _end_turn_outcomes(self, state: State) -> dict[State, float]:
        """`END_TURN`: end the ally's turn, then one actor at a time, merging states."""
        env = self.load(state)
        env._end_current_turn()
        frontier: dict[State, float] = {self.state_from_env(env): 1.0}
        finished: dict[State, float] = defaultdict(float)
        while frontier:
            following: dict[State, float] = defaultdict(float)
            for current, p in frontier.items():
                for q, done, successor in self._pass_step(current):
                    (finished if done else following)[successor] += p * q
            frontier = following
        return finished

    def _pass_step(self, state: State) -> tuple[tuple[float, bool, State], ...]:
        cached = self._step_cache.get(state)
        if cached is not None:
            return cached
        env = self.env
        merged: dict[tuple[bool, State], float] = defaultdict(float)

        def run(rng: _PathRng) -> tuple[bool, State]:
            self.load(state)
            env.np_random = rng
            done, _, _ = env._pass_step()
            return done, self.state_from_env(env)

        for probability, key in enumerate_paths(run):
            merged[key] += probability
        result = tuple((p, done, s) for (done, s), p in merged.items())
        self._step_cache[state] = result
        return result


for _stage in TACTICAL_STAGES:
    register_transition_model(_stage, TacticalTransitionModel.from_observation)
