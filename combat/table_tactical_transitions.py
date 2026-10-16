"""Table-driven exact M5/M6 dynamics; the environment enumerator remains an oracle.

Why this exists: the replay oracle (`tactical_transitions.TacticalTransitionModel`)
runs real environment code for every random path. A three-target Cleave or a
three-enemy END_TURN took seconds, which made full-turn and enemy-phase search
impractical. This model computes the same distributions from cached HP tables;
measured speed and audit results are in docs/RESULTS.md.

HP tables merge capped damage/healing before expanding successor states. Costs,
reach, roll modes, target selection, and lifecycle updates come from the shared
rules. Only turn coordination and compact-state writes live here.

Correctness contract: it must equal the oracle exactly. This is checked by
`tests/test_table_tactical_transitions.py`, by
`python -m evaluation.tactical_transition_benchmark`, and by
`python -m evaluation.tactical_transition_audit`. The audit includes constructed edge states. Rerun it after any change to M5/M6 rules. Both checks compare against the
environment's own code, so neither can catch a rule that is wrong in `tactics.py`.
"""
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from functools import lru_cache

import numpy as np

from . import tactics
from .actions import Action
from .actors import ActorRef, Side
from .damage import DamageSpec
from .distributions import dice_sum_distribution, second_wind_healing_distribution
from .env import MAX_ROUNDS
from .mechanics import damage_dice_count, damage_from_dice, scale_damage, use_action_surge
from .resources import Resource, ResourcePool
from .rolls import RollMode, roll_attack
from .tactical_env import TacticalCombatEnv
from .tactical_transitions import TacticalTransitionModel, enumerate_paths, roll_hit_distribution
from .transitions import Outcome, State, Status
from .turns import TurnManager


@lru_cache(maxsize=None)
def _hp_outcomes(hp: int, maximum: int, distribution: tuple[tuple[int, float], ...], healing: bool = False) -> tuple[tuple[int, float], ...]:
    merged: dict[int, float] = defaultdict(float)
    for amount, probability in distribution:
        following = min(maximum, hp + amount) if healing else max(0, hp - amount)
        merged[following] += probability
    return tuple(sorted(merged.items()))


@lru_cache(maxsize=None)
def _damage(attack: int, ac: int, spec: DamageSpec, mode: RollMode, divisor: int) -> tuple[tuple[int, float], ...]:
    merged: dict[int, float] = defaultdict(float)
    # Enumerate only d20 paths, then convolve damage dice through the mechanics
    # helpers. The oracle independently enumerates all d20 AND damage paths.
    rolls = defaultdict(float)
    for p, result in enumerate_paths(lambda rng: roll_attack(attack, ac, rng, mode), exact=True):
        rolls[result.hit, result.critical] += float(p)
    for (hit, critical), probability in rolls.items():
        if not hit:
            merged[0] += probability
            continue
        for total, p in dice_sum_distribution(damage_dice_count(spec, critical), spec.die_size).items():
            merged[scale_damage(damage_from_dice(total, spec), divisor)] += probability * float(p)
    return tuple(sorted(merged.items()))


class TableTacticalTransitionModel(TacticalTransitionModel):
    """Same state encoding and interface as the independent replay oracle.

    Inherit its validated observation decoder: live input is decoded once per
    decision (about 0.2 ms), while search uses compact states. A direct decoder
    here was 20–40x faster but skipped `simulation_from_observation`'s checks,
    so a reused model planned from 2.5 HP as 2 HP (PR #11 review, 2026-10-15).
    """

    def __init__(self, env: TacticalCombatEnv) -> None:
        super().__init__(env)
        self._ally_base = tuple(4 + sum(self._ally_width[:i]) for i in range(len(env.allies)))
        self._enemy_bases = tuple(self._enemy_base + sum(self._enemy_width[:i]) for i in range(len(env.enemies)))
        self._ally_res = tuple({r: b + 4 + i for i, r in enumerate(rs)} for b, rs in zip(self._ally_base, self._ally_resources))
        self._enemy_res = tuple({r: b + 3 + i for i, r in enumerate(rs)} for b, rs in zip(self._enemy_bases, self._enemy_resources))
        self._starts, self._ends = {}, {}
        self._schedule_cache = {}
        # Obtain condition resets by applying the actual lifecycle functions to
        # scratch actors with every condition enabled. Refresh comes from actors.
        for ref in self._order:
            ally = ref.side is Side.ALLY
            actor = env._roster().get(ref)
            base = (self._ally_base if ally else self._enemy_bases)[ref.slot]
            res = (self._ally_res if ally else self._enemy_res)[ref.slot]
            names = ('dodging', 'deep', 'disengaged') if ally else ('prone',)
            for start in (True, False):
                scratch = copy(actor)
                scratch.resources = ResourcePool(dict(actor.resources._amounts))
                for name in names:
                    setattr(scratch, name, True)
                (tactics.begin_tactical_turn if start else tactics.end_tactical_turn)(scratch)
                updates = {base + 1 + i: 0 for i, name in enumerate(names) if not getattr(scratch, name)}
                if start:
                    updates.update({res[r]: amount for r, amount in actor.turn_refresh.items() if r in res})
                (self._starts if start else self._ends)[ref] = updates
        fields = list(env.observation_fields)
        self._state_cols: dict[int, int] = {}
        for side, bases, resources in (('ally', self._ally_base, self._ally_res), ('enemy', self._enemy_bases, self._enemy_res)):
            names = ('hp', 'dodging', 'deep', 'disengaged') if side == 'ally' else ('hp', 'prone', 'last_attacker')
            for slot, (base, res) in enumerate(zip(bases, resources)):
                for i, name in enumerate(names):
                    field = f'{side}_{slot}_{name}'
                    if field in fields:
                        self._state_cols[base + i] = fields.index(field)
                for resource, index in res.items():
                    self._state_cols[index] = fields.index(f'{side}_{slot}_{resource.name.lower()}_count')
        self._state_indices = np.array(list(self._state_cols))
        self._obs_indices = np.array(list(self._state_cols.values()))

    def observations(self, states: Sequence[State]) -> np.ndarray:
        result = np.tile(self._template, (len(states), 1))
        if not states:
            return result
        values = np.asarray(states)
        result[:, self._obs_indices] = values[:, self._state_indices]
        result[:, -1] = values[:, 3]
        # Terminal observations report the first living ally, as does the env.
        result[:, 0] = [s[2] if s[0] == Status.ONGOING and not s[1] else next((i for i, b in enumerate(self._ally_base) if s[b] > 0), 0) for s in states]
        return result

    def _finish(self, values: list[int]) -> State:
        if not any(values[b] for b in self._ally_base):
            values[0] = int(Status.LOSS)
        elif not any(values[b] for b in self._enemy_bases):
            values[0] = int(Status.WIN)
        if values[0] != Status.ONGOING:
            values[1:3] = [0, 0]
        return tuple(values)

    def _hp_branch(self, frontier: dict[State, float], index: int, maximum: int, distribution: tuple[tuple[int, float], ...], *, healing: bool = False) -> dict[State, float]:
        merged: dict[State, float] = defaultdict(float)
        for state, p in frontier.items():
            for hp, q in _hp_outcomes(state[index], maximum, distribution, healing):
                values = list(state)
                values[index] = hp
                merged[tuple(values)] += p * q
        return merged

    def outcomes(self, state: State, action: int) -> tuple[Outcome, ...]:
        key = (state, action)
        if key in self._outcome_cache:
            return self._outcome_cache[key]
        if action not in self.legal_actions(state):
            raise ValueError(f'Decision {action} is not legal in this state')
        if self.ends_turn(action):
            merged = self._end_turn_outcomes(state)
        else:
            merged = self._ally_outcomes(state, action)
        result = tuple((p, s) for s, p in merged.items())
        self._outcome_cache[key] = result
        return result

    def _ally_outcomes(self, state: State, action: int) -> dict[State, float]:
        env = self.load(state)
        slot = state[2]
        ally = env.allies[slot]
        decision = self.decisions[action]
        kind = decision.action
        if kind is Action.ACTION_SURGE:
            use_action_surge(ally)
            return {self.state_from_env(env): 1.0}
        if kind in (Action.DODGE, Action.DISENGAGE):
            if kind is Action.DODGE:
                tactics.use_dodge(ally)
            else:
                tactics.use_disengage(ally, env.enemies)
            return {self.state_from_env(env): 1.0}
        tactics.spend_tactical(ally, kind)
        values = list(self.state_from_env(env))
        if kind is Action.SECOND_WIND:
            frontier = self._hp_branch({tuple(values): 1.0}, self._ally_base[slot], ally.max_hp, tuple(second_wind_healing_distribution().items()), healing=True)
        elif kind is Action.TRIP:
            target = decision.target_index
            base = self._enemy_bases[target]
            values[base + 2] = slot + 1
            frontier = {}
            for hit, p in roll_hit_distribution(ally.attack_bonus, env.enemies[target].armor_class, RollMode.NORMAL):
                following = values.copy()
                following[base + 1] = int(hit)
                frontier[tuple(following)] = p
        elif kind in (Action.ATTACK, Action.CLEAVE):
            targets = (decision.target_index,) if kind is Action.ATTACK else tuple(i for i, yes in enumerate(tactics.cleave_targets(ally, env.enemies)) if yes)
            # Capture the complete target set before any damage (M6 rank wall).
            for target in targets:
                values[self._enemy_bases[target] + 2] = slot + 1
            frontier = {tuple(values): 1.0}
            for target in targets:
                enemy = env.enemies[target]
                distribution = _damage(ally.attack_bonus, enemy.armor_class, ally.damage, tactics.ally_attack_mode(enemy), 2 if kind is Action.CLEAVE else 1)
                frontier = self._hp_branch(frontier, self._enemy_bases[target], enemy.max_hp, distribution)
        elif kind is Action.ADVANCE:
            frontier = {tuple(values): 1.0}
            if not ally.disengaged:
                engaged = tactics.archer_engaged(env.allies, env.enemies)
                for target in tactics.opportunity_attackers(env.enemies):
                    enemy = env.enemies[target]
                    living, dead = {}, {}
                    for current, p in frontier.items():
                        if current[self._ally_base[slot]] == 0:
                            dead[current] = p
                        else:
                            following = list(current)
                            following[self._enemy_res[target][Resource.REACTION]] -= 1
                            living[tuple(following)] = p
                    distribution = _damage(enemy.attack_bonus, ally.armor_class, enemy.damage, tactics.enemy_attack_mode(enemy, ally, engaged_archer=engaged), 1)
                    frontier = self._hp_branch(living, self._ally_base[slot], ally.max_hp, distribution)
                    for current, p in dead.items():
                        frontier[current] = frontier.get(current, 0.0) + p
            following: dict[State, float] = defaultdict(float)
            for current, p in frontier.items():
                vals = list(current)
                if vals[self._ally_base[slot]]:
                    vals[self._ally_base[slot] + 2] = 1
                current = self._finish(vals)
                if current[0] == Status.ONGOING and not current[self._ally_base[slot]]:
                    for successor, q in self._end_turn_outcomes(current).items():
                        following[successor] += p * q
                else:
                    following[current] += p
            return following
        else:
            raise ValueError(f'Unsupported tactical action: {kind}')
        merged: dict[State, float] = defaultdict(float)
        for current, p in frontier.items():
            merged[self._finish(list(current))] += p
        return merged

    def _end_turn_outcomes(self, state: State) -> dict[State, float]:
        values = list(state)
        ref = ActorRef(Side.ENEMY if state[1] else Side.ALLY, state[2])
        for index, value in self._ends[ref].items():
            values[index] = value
        frontier = {tuple(values): 1.0}
        finished: dict[State, float] = defaultdict(float)
        while frontier:
            following: dict[State, float] = defaultdict(float)
            for current, p in frontier.items():
                for q, done, successor in self._pass_step(current):
                    (finished if done else following)[successor] += p * q
            frontier = following
        return finished

    def _next(self, state: State) -> tuple[ActorRef, bool]:
        living = tuple(state[b] > 0 for b in (*self._ally_base, *self._enemy_bases))
        key = (state[1], state[2], living)
        if key not in self._schedule_cache:
            turns = TurnManager(self._order, current=ActorRef(Side.ENEMY if state[1] else Side.ALLY, state[2]))
            self._schedule_cache[key] = turns.peek_next(lambda ref: living[self._order.index(ref)])
        return self._schedule_cache[key]

    def _pass_step(self, state: State) -> tuple[tuple[float, bool, State], ...]:
        if state in self._step_cache:
            return self._step_cache[state]
        ref, wrap = self._next(state)
        values = list(state)
        if wrap and state[3] == MAX_ROUNDS:
            values[0] = int(Status.TRUNCATED)
            values[1:3] = [0, 0]
            result = ((1.0, True, tuple(values)),)
        else:
            values[1:4] = [int(ref.side is Side.ENEMY), ref.slot, state[3] + int(wrap)]
            for index, value in self._starts[ref].items():
                values[index] = value
            if ref.side is Side.ALLY:
                result = ((1.0, True, tuple(values)),)
            else:
                result = self._enemy_outcomes(tuple(values), ref)
        self._step_cache[state] = result
        return result

    def _enemy_outcomes(self, state: State, ref: ActorRef) -> tuple[tuple[float, bool, State], ...]:
        env = self.load(state)
        enemy = env.enemies[ref.slot]
        target = tactics.enemy_heal_target(enemy, env.enemies)
        values = list(state)
        if target is not None:
            tactics.spend_tactical(enemy, Action.HEAL)
            values = list(self.state_from_env(env))
            spec = env.rules.heal
            distribution = tuple((total + spec.bonus, float(p)) for total, p in dice_sum_distribution(spec.dice_count, spec.die_size).items())
            frontier = self._hp_branch({tuple(values): 1.0}, self._enemy_bases[target], env.enemies[target].max_hp, distribution, healing=True)
        else:
            tactics.spend_tactical(enemy, Action.ATTACK)
            engaged = env.ranks and tactics.archer_engaged(env.allies, env.enemies)
            frontier: dict[State, float] = defaultdict(float)
            targets = enumerate_paths(lambda rng: tactics.choose_enemy_target(enemy, env.allies, rng))
            for probability, target in targets:
                ally = env.allies[target]
                distribution = _damage(enemy.attack_bonus, ally.armor_class, enemy.damage, tactics.enemy_attack_mode(enemy, ally, engaged_archer=engaged), 1)
                for successor, p in self._hp_branch({tuple(values): 1.0}, self._ally_base[target], ally.max_hp, distribution).items():
                    frontier[successor] += float(probability) * p
        merged: dict[State, float] = defaultdict(float)
        for current, p in frontier.items():
            vals = list(current)
            for index, value in self._ends[ref].items():
                vals[index] = value
            merged[self._finish(vals)] += p
        return tuple((p, s[0] != Status.ONGOING, s) for s, p in merged.items())
