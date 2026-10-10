"""Exact chance-node transitions for planning over the staged combat simulators.

A `TransitionModel` is bound to one encounter's static statistics. For a compact,
hashable state and a legal decision it returns the exact distribution of
successor states, including the complete automatic phase after `END_TURN`.
Identical successors are merged, so HP outcomes collapse into few states.

Rules are not re-derived here: dice, hit/critical, damage halving, and healing
come from the distribution helpers in `mechanics.py`; costs from the ability
catalog; legality and the automatic enemy policy from `legality.py`; turn order
and refresh from the environment's `TurnManager` order and `turn_refresh`.

New stages plug in by registering a factory (`register_transition_model`) that
returns any object satisfying `TransitionModel`, i.e. their own outcome
enumeration over their own state encoding. Planners depend only on the protocol.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Protocol

import numpy as np

from .abilities import M2_ABILITIES
from .actions import Action
from .actors import ActorRef, Side
from .characters import refresh_turn_resources
from .env import MAX_ROUNDS, BaldurCombatEnv
from .legality import automatic_attack_decision, legal_mask_for
from .distributions import attack_damage_distribution, second_wind_healing_distribution
from .resources import Resource
from .simulation import simulation_from_observation
from .stages import StagedCombatEnv


class Status(IntEnum):
    ONGOING = 0
    WIN = 1
    LOSS = 2
    TRUNCATED = 3


State = tuple[int, ...]
Outcome = tuple[float, State]
_STATUSES = tuple(Status)


class TransitionModel(Protocol):
    """Exact one-decision dynamics for one encounter, as used by planners."""

    stage: str
    action_count: int

    def state_from_observation(self, observation: np.ndarray) -> State: ...

    def same_encounter(self, observation: np.ndarray) -> bool: ...

    def status(self, state: State) -> Status: ...

    def legal_actions(self, state: State) -> tuple[int, ...]: ...

    def outcomes(self, state: State, action: int) -> tuple[Outcome, ...]: ...

    def ends_turn(self, action: int) -> bool: ...

    def observations(self, states: Sequence[State]) -> np.ndarray: ...


# Compact state layout for the M0–M4 roster model:
#   (status, active ally slot, round, *ally blocks, *enemy HP)
# Each ally block is (hp, ACTION, BONUS_ACTION, SECOND_WIND, ACTION_SURGE, CLEAVE).
# The active slot is meaningful only for ONGOING states and is 0 otherwise.
# Enemy per-turn resources are omitted: the environment refreshes them at the
# start of every enemy turn before use, and they are never observed.
_STATUS, _ACTOR, _ROUND, _HEADER = 0, 1, 2, 3
_RESOURCES = tuple(Resource)
_ALLY_WIDTH = 1 + len(_RESOURCES)
_COUNT_FIELDS = {
    "hp": 0,
    "action_count": 1 + _RESOURCES.index(Resource.ACTION),
    "bonus_action_count": 1 + _RESOURCES.index(Resource.BONUS_ACTION),
    "second_wind_count": 1 + _RESOURCES.index(Resource.SECOND_WIND),
    "action_surge_count": 1 + _RESOURCES.index(Resource.ACTION_SURGE),
    "cleave_count": 1 + _RESOURCES.index(Resource.CLEAVE),
}
_STATIC_ENEMY_FIELDS = ("max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus")


def _transition_table(distribution: dict[int, float], hp_values: range, apply: Callable[[int, int], int]) -> list[tuple[tuple[int, float], ...]]:
    """Per current HP, merged (new HP, probability) pairs for one chance event."""
    table = []
    for hp in hp_values:
        merged: dict[int, float] = defaultdict(float)
        for amount, probability in distribution.items():
            merged[apply(hp, amount)] += probability
        table.append(tuple(sorted(merged.items())))
    return table


class RosterTransitionModel:
    """Exact transitions for the M0–M4 environments (fixed-policy enemies)."""

    def __init__(self, env: BaldurCombatEnv | StagedCombatEnv) -> None:
        self.stage = env.stage
        self.decisions = tuple(env.decisions)
        self.action_count = len(self.decisions)
        self.max_rounds = MAX_ROUNDS
        # A private roster used only as scratch input to the shared legality rules.
        self._env = env
        self._roster = env._roster()
        self._allies = self._roster.allies
        self._enemies = self._roster.enemies
        self._order = env._turns().order
        self._controlled = frozenset(env._controlled_refs)
        if any(ref.side is not Side.ALLY for ref in self._controlled):
            raise ValueError("Only allied actors may be policy-controlled")
        self.n_allies = len(self._allies)
        self.n_enemies = len(self._enemies)
        self._enemy_base = _HEADER + _ALLY_WIDTH * self.n_allies
        self._costs = {
            action: tuple(
                (1 + _RESOURCES.index(resource), amount) for resource, amount in spec.costs.items()
            )
            for action, spec in M2_ABILITIES.items()
        }
        self._refresh = tuple(
            tuple((1 + _RESOURCES.index(resource), amount) for resource, amount in ally.turn_refresh.items())
            for ally in self._allies
        )
        self._legal_cache: dict[tuple, tuple[int, ...]] = {}
        self._target_cache: dict[tuple, int | None] = {}
        self._phase_cache: dict[tuple, tuple] = {}
        self._attack_tables: dict[tuple[Side, int, int, int], list] = {}
        heal = second_wind_healing_distribution()
        self._heal_tables = [
            _transition_table(heal, range(ally.max_hp + 1), lambda hp, h, top=ally.max_hp: hp + min(top - hp, h))
            for ally in self._allies
        ]
        self._build_observation_map(env)

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_observation(cls, stage: str, observation: np.ndarray) -> "RosterTransitionModel":
        return cls(simulation_from_observation(stage, observation, seed=0))

    def state_from_env(self, env: BaldurCombatEnv | StagedCombatEnv) -> State:
        """Compact state of a live environment with the same encounter statistics."""
        roster = env._roster()
        allies_down = roster.side_defeated(Side.ALLY)
        enemies_down = roster.side_defeated(Side.ENEMY)
        if allies_down:
            status = Status.LOSS
        elif enemies_down:
            status = Status.WIN
        elif env._truncated:
            status = Status.TRUNCATED
        else:
            status = Status.ONGOING
        current = env._turns().current
        actor = current.slot if status is Status.ONGOING else 0
        if status is Status.ONGOING and current not in self._controlled:
            raise ValueError("Ongoing states must be at a controlled decision")
        values = [int(status), actor, env.round_number]
        for ally in roster.allies:
            values.append(ally.hp)
            values.extend(ally.resources.get(resource) for resource in _RESOURCES)
        values.extend(enemy.hp for enemy in roster.enemies)
        return tuple(values)

    def state_from_observation(self, observation: np.ndarray) -> State:
        """Compact ongoing state for a live allied decision observation."""
        values = np.asarray(observation)
        if values.shape != self._template.shape or not self.same_encounter(values):
            raise ValueError("Observation does not belong to this encounter")
        ints = [int(value) for value in values]
        state = [int(Status.ONGOING), ints[self._actor_col] if self._actor_col is not None else 0, 0]
        state.extend([0] * (_ALLY_WIDTH * self.n_allies + self.n_enemies))
        for column, index in zip(self._dynamic_cols, self._dynamic_src):
            state[index] = ints[column]
        return tuple(state)

    def same_encounter(self, observation: np.ndarray) -> bool:
        values = np.asarray(observation)
        return values.shape == self._template.shape and bool(
            np.array_equal(values[self._static_cols], self._template[self._static_cols])
        )

    # ------------------------------------------------------------------ queries
    def status(self, state: State) -> Status:
        return _STATUSES[state[_STATUS]]

    def ends_turn(self, action: int) -> bool:
        return self.decisions[action].action is Action.END_TURN

    def legal_actions(self, state: State) -> tuple[int, ...]:
        if state[_STATUS] != Status.ONGOING:
            return ()
        slot = state[_ACTOR]
        base = _HEADER + _ALLY_WIDTH * slot
        key = (slot, state[base:base + _ALLY_WIDTH], tuple(hp > 0 for hp in state[self._enemy_base:]))
        legal = self._legal_cache.get(key)
        if legal is None:
            self._load(state)
            mask = legal_mask_for(ActorRef(Side.ALLY, slot), self._roster, self.decisions)
            legal = tuple(index for index, allowed in enumerate(mask) if allowed)
            self._legal_cache[key] = legal
        return legal

    def legal_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.bool_)
        mask[list(self.legal_actions(state))] = True
        return mask

    # ------------------------------------------------------------------ transitions
    def outcomes(self, state: State, action: int) -> tuple[Outcome, ...]:
        """Exact merged successor distribution of one legal controlled decision."""
        if action not in self.legal_actions(state):
            raise ValueError(f"Decision {action} is not legal in this state")
        decision = self.decisions[action]
        slot = state[_ACTOR]
        kind = decision.action
        if kind is Action.END_TURN:
            return tuple((p, s) for s, p in self._end_turn(state).items())
        spent = list(state)
        for index, amount in self._costs[kind]:
            spent[_HEADER + _ALLY_WIDTH * slot + index] -= amount
        if kind is Action.ACTION_SURGE:
            spent[_HEADER + _ALLY_WIDTH * slot + 1 + _RESOURCES.index(Resource.ACTION)] += 1
            return ((1.0, tuple(spent)),)
        if kind is Action.SECOND_WIND:
            hp_index = _HEADER + _ALLY_WIDTH * slot
            results = []
            for hp, p in self._heal_tables[slot][spent[hp_index]]:
                spent[hp_index] = hp
                results.append((p, tuple(spent)))
            return tuple(results)
        if kind is Action.ATTACK:
            target = 0 if decision.target_index is None else decision.target_index
            targets = (target,)
            divisor = 1
        elif kind is Action.CLEAVE:
            targets = tuple(t for t in range(self.n_enemies) if state[self._enemy_base + t] > 0)
            divisor = 2
        else:
            raise ValueError(f"Unsupported decision: {decision.label}")
        branches: list[tuple[float, list[int]]] = [(1.0, spent)]
        for target in targets:
            index = self._enemy_base + target
            table = self._attack_table(Side.ALLY, slot, target, divisor)
            expanded = []
            for p, values in branches:
                for hp, q in table[values[index]]:
                    child = values.copy()
                    child[index] = hp
                    expanded.append((p * q, child))
            branches = expanded
        merged: dict[State, float] = defaultdict(float)
        for p, values in branches:
            merged[self._settle(values)] += p
        return tuple((p, s) for s, p in merged.items())

    def _settle(self, values: list[int]) -> State:
        """Apply the step-end defeat check to a post-decision ally state."""
        allies_down = not any(values[_HEADER + _ALLY_WIDTH * slot] > 0 for slot in range(self.n_allies))
        enemies_down = not any(hp > 0 for hp in values[self._enemy_base:])
        if allies_down or enemies_down:
            values[_STATUS] = int(Status.LOSS if allies_down else Status.WIN)
            values[_ACTOR] = 0
        return tuple(values)

    def _end_turn(self, state: State) -> dict[State, float]:
        """Hand play to the next living actor and resolve every automatic turn."""
        start = self._order.index(ActorRef(Side.ALLY, state[_ACTOR]))
        ally_hps = tuple(state[_HEADER + _ALLY_WIDTH * slot] for slot in range(self.n_allies))
        enemy_alive = tuple(hp > 0 for hp in state[self._enemy_base:])
        key = (start, ally_hps, enemy_alive, state[_ROUND] == self.max_rounds)
        phase = self._phase_cache.get(key)
        if phase is None:
            phase = self._automatic_phase(start, ally_hps, enemy_alive, state[_ROUND] == self.max_rounds)
            self._phase_cache[key] = phase
        results: dict[State, float] = defaultdict(float)
        for p, hps, status, next_slot, wrapped in phase:
            values = list(state)
            for slot, hp in enumerate(hps):
                values[_HEADER + _ALLY_WIDTH * slot] = hp
            values[_STATUS] = int(status)
            if status is Status.ONGOING:
                values[_ACTOR] = next_slot
                values[_ROUND] += int(wrapped)
                base = _HEADER + _ALLY_WIDTH * next_slot
                for index, amount in self._refresh[next_slot]:
                    values[base + index] = amount
            else:
                values[_ACTOR] = 0
            results[tuple(values)] += p
        return results

    def _automatic_phase(
        self, start: int, ally_hps: tuple[int, ...], enemy_alive: tuple[bool, ...], last_round: bool
    ) -> tuple:
        """Distribution of (ally HPs, status, next controlled slot, wrapped).

        Mirrors the environment's END_TURN loop: advance and refresh, run each
        automatic turn, stop on defeat, truncate when the last round would wrap.
        Enemies cannot lose HP here, so their liveness is fixed for the phase.
        """
        order = self._order
        count = len(order)

        def alive(ref: ActorRef, hps: tuple[int, ...]) -> bool:
            return hps[ref.slot] > 0 if ref.side is Side.ALLY else enemy_alive[ref.slot]

        def next_living(position: int, hps: tuple[int, ...]) -> tuple[int, bool]:
            for step in range(1, count + 1):
                candidate = (position + step) % count
                if alive(order[candidate], hps):
                    return candidate, candidate <= position
            raise RuntimeError("No living actor")

        position, wrapped = next_living(start, ally_hps)
        if wrapped:
            raise NotImplementedError("Ending a controlled turn cannot wrap the round in M0–M4")
        finished: dict[tuple, float] = defaultdict(float)
        frontier: dict[tuple[tuple[int, ...], int, bool], float] = {(ally_hps, position, False): 1.0}
        while frontier:
            following: dict[tuple[tuple[int, ...], int, bool], float] = defaultdict(float)
            for (hps, position, has_wrapped), p in frontier.items():
                ref = order[position]
                if ref in self._controlled:
                    finished[(hps, Status.ONGOING, ref.slot, has_wrapped)] += p
                    continue
                for q, after in self._automatic_turn(ref, hps, enemy_alive):
                    if not any(hp > 0 for hp in after):
                        finished[(after, Status.LOSS, 0, False)] += p * q
                        continue
                    following_position, would_wrap = next_living(position, after)
                    if would_wrap and has_wrapped:
                        raise NotImplementedError("Automatic phases spanning two rounds are unsupported")
                    if would_wrap and last_round:
                        finished[(after, Status.TRUNCATED, 0, False)] += p * q
                        continue
                    following[(after, following_position, has_wrapped or would_wrap)] += p * q
            frontier = following
        return tuple((p, hps, status, slot, wrapped) for (hps, status, slot, wrapped), p in finished.items())

    def _automatic_turn(
        self, ref: ActorRef, ally_hps: tuple[int, ...], enemy_alive: tuple[bool, ...]
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        if ref.side is not Side.ENEMY:
            raise NotImplementedError("Automatic allied actors are unsupported")
        key = (ref.slot, tuple(hp > 0 for hp in ally_hps), enemy_alive)
        if key not in self._target_cache:
            for slot, ally in enumerate(self._allies):
                ally.hp = ally_hps[slot]
            for slot, enemy in enumerate(self._enemies):
                enemy.hp = enemy.max_hp if enemy_alive[slot] else 0
            enemy = self._roster.get(ref)
            refresh_turn_resources(enemy)
            decision = automatic_attack_decision(ref, self._roster, self.decisions)
            self._target_cache[key] = None if decision is None else (
                0 if decision.target_index is None else decision.target_index
            )
        target = self._target_cache[key]
        if target is None:
            return ((1.0, ally_hps),)
        table = self._attack_table(Side.ENEMY, ref.slot, target, 1)
        results = []
        for hp, q in table[ally_hps[target]]:
            after = list(ally_hps)
            after[target] = hp
            results.append((q, tuple(after)))
        return tuple(results)

    def _attack_table(self, side: Side, slot: int, target: int, divisor: int) -> list:
        key = (side, slot, target, divisor)
        table = self._attack_tables.get(key)
        if table is None:
            attacker = self._allies[slot] if side is Side.ALLY else self._enemies[slot]
            defender = self._enemies[target] if side is Side.ALLY else self._allies[target]
            distribution = attack_damage_distribution(
                attacker.attack_bonus, defender.armor_class, attacker.damage, divisor
            )
            table = _transition_table(distribution, range(defender.max_hp + 1), lambda hp, d: hp - min(hp, d))
            self._attack_tables[key] = table
        return table

    def _load(self, state: State) -> None:
        """Write a compact state into the scratch roster for legality queries."""
        for slot, ally in enumerate(self._allies):
            base = _HEADER + _ALLY_WIDTH * slot
            ally.hp = state[base]
            for index, resource in enumerate(_RESOURCES):
                ally.resources.set(resource, state[base + 1 + index])
        for slot, enemy in enumerate(self._enemies):
            enemy.hp = state[self._enemy_base + slot]

    # ------------------------------------------------------------------ observations
    def _build_observation_map(self, env: BaldurCombatEnv | StagedCombatEnv) -> None:
        fields = tuple(env.observation_fields)
        template = (env._get_observation() if isinstance(env, BaldurCombatEnv) else env._observation())
        self._template = np.asarray(template, dtype=np.float32).copy()
        dynamic_cols: list[int] = []
        dynamic_src: list[int] = []
        static_cols: list[int] = []
        self._actor_col: int | None = None
        for column, name in enumerate(fields):
            source = self._field_source(name)
            if source == "actor":
                self._actor_col = column
            elif source is None:
                static_cols.append(column)
            else:
                dynamic_cols.append(column)
                dynamic_src.append(source)
        self._dynamic_cols = np.array(dynamic_cols, dtype=np.intp)
        self._dynamic_src = np.array(dynamic_src, dtype=np.intp)
        self._static_cols = np.array(static_cols, dtype=np.intp)

    def _field_source(self, name: str) -> int | str | None:
        if name == "round_number":
            return _ROUND
        if name == "active_actor_index":
            return "actor"
        if name == "goblin_hp" or name == "enemy_hp":
            return self._enemy_base
        if name.startswith("fighter_") and name[len("fighter_"):] in _COUNT_FIELDS:
            return _HEADER + _COUNT_FIELDS[name[len("fighter_"):]]
        parts = name.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            side, slot, rest = parts[0], int(parts[1]), parts[2]
            if side == "ally" and rest in _COUNT_FIELDS:
                return _HEADER + _ALLY_WIDTH * slot + _COUNT_FIELDS[rest]
            if side == "enemy" and rest == "hp":
                return self._enemy_base + slot
            if side == "enemy" and rest in _STATIC_ENEMY_FIELDS:
                return None
        if name.startswith("enemy_") and name[len("enemy_"):] in _STATIC_ENEMY_FIELDS:
            return None
        raise ValueError(f"Unsupported observation field for exact transitions: {name}")

    def observations(self, states: Sequence[State]) -> np.ndarray:
        """Batch of environment observation vectors for compact states."""
        array = np.asarray(states, dtype=np.int64).reshape(len(states), -1)
        out = np.repeat(self._template[None, :], len(array), axis=0)
        out[:, self._dynamic_cols] = array[:, self._dynamic_src]
        if self._actor_col is not None:
            actor = array[:, _ACTOR].copy()
            ended = array[:, _STATUS] != Status.ONGOING
            if np.any(ended):
                hps = array[:, [_HEADER + _ALLY_WIDTH * slot for slot in range(self.n_allies)]]
                living = hps > 0
                first = np.where(living.any(axis=1), living.argmax(axis=1), 0)
                actor[ended] = first[ended]
            out[:, self._actor_col] = actor
        return out

    def observation(self, state: State) -> np.ndarray:
        return self.observations([state])[0]


TransitionModelFactory = Callable[[str, np.ndarray], TransitionModel]
_MODEL_FACTORIES: dict[str, TransitionModelFactory] = {
    stage: RosterTransitionModel.from_observation for stage in ("m0", "m1a", "m1b", "m2", "m3", "m4")
}


def register_transition_model(stage: str, factory: TransitionModelFactory) -> None:
    """Register a stage's own exact outcome enumeration for planners."""
    _MODEL_FACTORIES[stage] = factory


def transition_model(stage: str, observation: np.ndarray) -> TransitionModel:
    """Exact transition model for the encounter described by a live observation."""
    try:
        factory = _MODEL_FACTORIES[stage]
    except KeyError as exc:
        raise ValueError(f"No exact transition model registered for {stage}") from exc
    return factory(stage, observation)
