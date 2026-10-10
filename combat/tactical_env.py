"""Gymnasium environments for M5 (roles and conditions) and M6 (two ranks).

The environment coordinates sampling, turns, legality, observations, rewards,
and termination. Combat rules live in ``combat.tactics``, ``combat.rolls``, and
``combat.mechanics``. Every M5/M6 rule is a SIMULATOR CHOICE specified in
docs/STAGES_SPEC.md.
"""

from dataclasses import asdict
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import Action
from .actors import ActorRef, CombatRoster, Side
from .characters import Rank, Role, TacticalEnemy, TacticalFighter, Targeting
from .damage import DamageSpec
from .env import MAX_ROUNDS, _new_fighter
from .legality import DecisionSpec, tactical_legal_mask
from .mechanics import use_action_surge, use_second_wind
from .resources import Resource, ResourcePool
from .tactics import (
    EnemyTurnResult,
    RoleProfile,
    TacticalRules,
    archer_engaged,
    begin_tactical_turn,
    end_tactical_turn,
    run_enemy_turn,
    use_advance,
    use_disengage,
    use_dodge,
    use_tactical_attack,
    use_tactical_cleave,
    use_trip,
)
from .turns import TurnManager


TACTICAL_STAGES = ("m5", "m6")
N_ALLIES = 2
N_ENEMIES = 3
ROLE_ORDER = (Role.BRUTE, Role.ARCHER, Role.HEALER)
TARGETING_ORDER = (Targeting.WEAKEST, Targeting.RETALIATE, Targeting.RANDOM)

# Per-stage tuning data; see the STAGES_SPEC M5/M6 tuning history. V1 is kept
# only to document and reproduce the first measurement; stages use V2.
V1_RULES = TacticalRules(
    profiles={
        Role.BRUTE: RoleProfile(hp=(24, 32), ac=(14, 16), attack=(4, 6), die=10, bonus=(2, 4), rank=Rank.FRONT),
        Role.ARCHER: RoleProfile(hp=(10, 16), ac=(12, 14), attack=(5, 7), die=8, bonus=(2, 4), rank=Rank.BACK),
        Role.HEALER: RoleProfile(hp=(12, 18), ac=(12, 14), attack=(2, 4), die=6, bonus=(0, 2), rank=Rank.BACK),
    },
    heal=DamageSpec(1, 8, 2),
    heal_charges=3,
    trip_charges=99,  # v1 Trip was limited only by the Bonus Action.
)
_V2_ARCHER = RoleProfile(hp=(10, 16), ac=(12, 14), attack=(6, 8), die=8, bonus=(3, 5), rank=Rank.BACK)
_V2_HEALER = RoleProfile(hp=(12, 18), ac=(12, 14), attack=(2, 4), die=6, bonus=(0, 2), rank=Rank.BACK)
M5_RULES = TacticalRules(
    profiles={
        Role.BRUTE: RoleProfile(hp=(24, 32), ac=(14, 16), attack=(5, 7), die=10, bonus=(3, 5), rank=Rank.FRONT),
        Role.ARCHER: _V2_ARCHER,
        Role.HEALER: _V2_HEALER,
    },
    heal=DamageSpec(2, 6, 2),
    heal_charges=3,
    trip_charges=2,
)
# M6 offsets the rank wall: a less durable Brute and one fewer heal charge.
M6_RULES = TacticalRules(
    profiles={
        Role.BRUTE: RoleProfile(hp=(20, 26), ac=(14, 16), attack=(5, 7), die=10, bonus=(3, 5), rank=Rank.FRONT),
        Role.ARCHER: _V2_ARCHER,
        Role.HEALER: _V2_HEALER,
    },
    heal=DamageSpec(2, 6, 2),
    heal_charges=2,
    trip_charges=2,
)
STAGE_RULES: dict[str, TacticalRules] = {"m5": M5_RULES, "m6": M6_RULES}

_ALLY_FIELDS = (
    "hp", "action_count", "bonus_action_count", "second_wind_count",
    "action_surge_count", "cleave_count", "trip_count", "dodging",
)
_ALLY_RANK_FIELDS = ("movement_count", "deep", "disengaged")
_ENEMY_FIELDS = (
    "hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus",
    "is_brute", "is_archer", "is_healer",
    "targets_weakest", "targets_retaliate", "targets_random",
    "prone", "heal_count", "last_attacker",
)
_ENEMY_RANK_FIELDS = ("back_rank", "reaction_count")


def has_ranks(stage: str) -> bool:
    return stage == "m6"


def _check_stage(stage: str) -> None:
    if stage not in TACTICAL_STAGES:
        raise ValueError(f"Unknown tactical stage: {stage}")


def tactical_decisions(stage: str) -> tuple[DecisionSpec, ...]:
    """Flat decision list: attacks, trips, self actions, M6 movement, END_TURN last."""
    _check_stage(stage)
    attacks = tuple(DecisionSpec(Action.ATTACK, slot) for slot in range(N_ENEMIES))
    trips = tuple(DecisionSpec(Action.TRIP, slot) for slot in range(N_ENEMIES))
    core = tuple(DecisionSpec(action) for action in (
        Action.CLEAVE, Action.DODGE, Action.SECOND_WIND, Action.ACTION_SURGE
    ))
    movement = (DecisionSpec(Action.ADVANCE), DecisionSpec(Action.DISENGAGE)) if has_ranks(stage) else ()
    return attacks + trips + core + movement + (DecisionSpec(Action.END_TURN),)


def ally_fields(stage: str) -> tuple[str, ...]:
    return _ALLY_FIELDS + (_ALLY_RANK_FIELDS if has_ranks(stage) else ())


def enemy_fields(stage: str) -> tuple[str, ...]:
    return _ENEMY_FIELDS + (_ENEMY_RANK_FIELDS if has_ranks(stage) else ())


def tactical_observation_fields(stage: str) -> tuple[str, ...]:
    _check_stage(stage)
    allies = tuple(f"ally_{slot}_{field}" for slot in range(N_ALLIES) for field in ally_fields(stage))
    enemies = tuple(f"enemy_{slot}_{field}" for slot in range(N_ENEMIES) for field in enemy_fields(stage))
    return ("active_actor_index",) + allies + enemies + ("round_number",)


def tactical_observation_high(stage: str) -> np.ndarray:
    """Per-field Box highs, derived from the presets and the stage's tuning data."""
    rules = STAGE_RULES[stage]
    profiles = rules.profiles.values()
    ally_high = [20, 2, 1, 1, 1, 1, rules.trip_charges, 1] + ([1, 1, 1] if has_ranks(stage) else [])
    enemy_high = [
        max(p.hp[1] for p in profiles), max(p.hp[1] for p in profiles),
        max(p.ac[1] for p in profiles), max(p.attack[1] for p in profiles),
        max(p.die for p in profiles), max(p.bonus[1] for p in profiles),
        1, 1, 1, 1, 1, 1, 1, rules.heal_charges, N_ALLIES,
    ] + ([1, 1] if has_ranks(stage) else [])
    high = [N_ALLIES - 1] + ally_high * N_ALLIES + enemy_high * N_ENEMIES + [MAX_ROUNDS]
    return np.array(high, dtype=np.float32)


def make_tactical_ally(stage: str) -> TacticalFighter:
    """The M4 Fighter preset with Cleave, Trip, Dodge, and M6 movement abilities."""
    base = _new_fighter()
    known = {Action.ATTACK, Action.SECOND_WIND, Action.ACTION_SURGE, Action.CLEAVE, Action.TRIP, Action.DODGE}
    resources = {
        Resource.ACTION: 1, Resource.BONUS_ACTION: 1, Resource.SECOND_WIND: 1,
        Resource.ACTION_SURGE: 1, Resource.CLEAVE: 1, Resource.TRIP: STAGE_RULES[stage].trip_charges,
    }
    refresh = {Resource.ACTION: 1, Resource.BONUS_ACTION: 1}
    if has_ranks(stage):
        known |= {Action.ADVANCE, Action.DISENGAGE}
        resources[Resource.MOVEMENT] = 1
        refresh[Resource.MOVEMENT] = 1
    return TacticalFighter(
        base.name, base.max_hp, base.hp, base.armor_class, base.attack_bonus, base.damage,
        resources=ResourcePool(resources),
        known_abilities=frozenset(known),
        turn_refresh=MappingProxyType(refresh),
    )


def make_tactical_enemy(
    stage: str, slot: int, role: Role, targeting: Targeting,
    max_hp: int, armor_class: int, attack_bonus: int, damage_bonus: int,
) -> TacticalEnemy:
    """Build a full-HP enemy; M5 puts every enemy in the front rank."""
    rules = STAGE_RULES[stage]
    profile = rules.profiles[role]
    ranks = has_ranks(stage)
    known = {Action.ATTACK} | ({Action.HEAL} if role is Role.HEALER else set())
    resources = {Resource.ACTION: 1}
    refresh = {Resource.ACTION: 1}
    if role is Role.HEALER:
        resources[Resource.HEAL] = rules.heal_charges
    if ranks:
        resources[Resource.REACTION] = 1
        refresh[Resource.REACTION] = 1
    return TacticalEnemy(
        f"{role.name.title()} {slot}", max_hp, max_hp, armor_class, attack_bonus,
        DamageSpec(1, profile.die, damage_bonus),
        resources=ResourcePool(resources),
        known_abilities=frozenset(known),
        turn_refresh=MappingProxyType(refresh),
        role=role,
        targeting=targeting,
        rank=profile.rank if ranks else Rank.FRONT,
    )


class TacticalCombatEnv(gym.Env[np.ndarray, int]):
    """One allied decision per step for two Fighters against Brute, Archer, and Healer."""

    metadata = {"render_modes": []}

    def __init__(self, stage: str = "m5", reward_mode: str = "terminal") -> None:
        _check_stage(stage)
        if reward_mode != "terminal":
            raise ValueError(f"Reward mode {reward_mode!r} is unavailable for {stage}")
        super().__init__()
        self.stage = stage
        self.reward_mode = reward_mode
        self.rules = STAGE_RULES[stage]
        self.ranks = has_ranks(stage)
        self.decisions = tactical_decisions(stage)
        self.action_names = tuple(decision.label for decision in self.decisions)
        self.observation_fields = tactical_observation_fields(stage)
        self.action_space = spaces.Discrete(len(self.decisions))
        high = tactical_observation_high(stage)
        low = np.zeros_like(high)
        low[-1] = 1
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.allies: tuple[TacticalFighter, ...] = ()
        self.enemies: tuple[TacticalEnemy, ...] = ()
        self.roster: CombatRoster | None = None
        self.turns: TurnManager | None = None
        self._controlled_refs = frozenset(ActorRef(Side.ALLY, slot) for slot in range(N_ALLIES))
        self.round_number = 1
        self._terminated = False
        self._truncated = False

    # --- episode setup -------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        allies = tuple(make_tactical_ally(self.stage) for _ in range(N_ALLIES))
        self._install(allies, self._sample_enemies(), round_number=1, current=None)
        begin_tactical_turn(self._roster().get(self._turns().current))
        return self._observation(), self._info(None)

    def _sample_enemies(self) -> tuple[TacticalEnemy, ...]:
        rng = self.np_random
        roles: list[Role] = [Role.BRUTE] * N_ENEMIES
        for role, slot in zip(ROLE_ORDER, rng.permutation(N_ENEMIES)):
            roles[int(slot)] = role
        enemies = []
        for slot, role in enumerate(roles):
            profile = self.rules.profiles[role]
            hp, ac, attack, bonus = (
                int(rng.integers(low, high + 1))
                for low, high in (profile.hp, profile.ac, profile.attack, profile.bonus)
            )
            targeting = TARGETING_ORDER[int(rng.integers(0, len(TARGETING_ORDER)))]
            enemies.append(make_tactical_enemy(self.stage, slot, role, targeting, hp, ac, attack, bonus))
        return tuple(enemies)

    def _install(
        self,
        allies: tuple[TacticalFighter, ...],
        enemies: tuple[TacticalEnemy, ...],
        *,
        round_number: int,
        current: ActorRef | None,
    ) -> None:
        """Adopt a roster and turn position without refreshing any resources."""
        self.allies = allies
        self.enemies = enemies
        self.roster = CombatRoster(allies=allies, enemies=enemies)
        order = tuple(ActorRef(Side.ALLY, slot) for slot in range(len(allies))) + tuple(
            ActorRef(Side.ENEMY, slot) for slot in range(len(enemies))
        )
        self.turns = TurnManager(order, round_number=round_number, current=current)
        self.round_number = round_number
        self._terminated = False
        self._truncated = False

    def _roster(self) -> CombatRoster:
        if self.roster is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.roster

    def _turns(self) -> TurnManager:
        if self.turns is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.turns

    def _is_policy_controlled(self, ref: ActorRef) -> bool:
        return ref in self._controlled_refs

    # --- decisions -------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        if self._terminated or self._truncated:
            return np.zeros(len(self.decisions), dtype=np.bool_)
        current = self._turns().current
        if not self._is_policy_controlled(current):
            return np.zeros(len(self.decisions), dtype=np.bool_)
        return np.array(
            tactical_legal_mask(current.slot, self.allies, self.enemies, self.decisions), dtype=np.bool_
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        roster = self._roster()
        turns = self._turns()
        if self._terminated or self._truncated:
            raise RuntimeError("Episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid {self.stage} action index: {action!r}")
        index = int(action)
        decision = self.decisions[index]
        if not self.action_masks()[index]:
            raise ValueError(f"{decision.label} is not legal now")

        slot = turns.current.slot
        ally = self.allies[slot]
        rng = self.np_random
        info: dict[str, Any] = {}
        kind = decision.action
        if kind is Action.ATTACK:
            target = self.enemies[decision.target_index]
            info["fighter_attack"] = asdict(use_tactical_attack(slot, ally, target, self.enemies, rng))
        elif kind is Action.TRIP:
            target = self.enemies[decision.target_index]
            info["trip"] = asdict(use_trip(slot, ally, target, self.enemies, rng))
        elif kind is Action.CLEAVE:
            results = use_tactical_cleave(slot, ally, self.enemies, rng)
            info["cleave_attacks"] = [asdict(result) if result is not None else None for result in results]
            info["cleave_living_targets"] = sum(result is not None for result in results)
        elif kind is Action.DODGE:
            use_dodge(ally)
        elif kind is Action.SECOND_WIND:
            info["healed"] = use_second_wind(ally, rng)
        elif kind is Action.ACTION_SURGE:
            use_action_surge(ally)
        elif kind is Action.ADVANCE:
            attacks = use_advance(ally, self.allies, self.enemies, rng)
            info["advance"] = {"opportunity_attacks": [asdict(attack) for attack in attacks], "deep": ally.deep}
        elif kind is Action.DISENGAGE:
            use_disengage(ally, self.enemies)

        allies_down = roster.side_defeated(Side.ALLY)
        enemies_down = roster.side_defeated(Side.ENEMY)
        # END_TURN, or an opportunity attack killed the active ally mid-turn.
        if not (allies_down or enemies_down) and (kind is Action.END_TURN or not ally.alive):
            info["enemy_turns"] = self._pass_turn()
            allies_down = roster.side_defeated(Side.ALLY)
            enemies_down = roster.side_defeated(Side.ENEMY)
        self._terminated = allies_down or enemies_down
        if allies_down:
            reward = -1.0  # Includes a (currently impossible) mutual defeat.
        elif enemies_down:
            reward = 1.0
        else:
            reward = 0.0
        info.update(self._info(decision))
        return self._observation(), reward, self._terminated, self._truncated, info

    def _pass_turn(self) -> list[dict[str, Any] | None]:
        """End the current ally's turn; run enemies until an ally decides or play stops."""
        roster = self._roster()
        turns = self._turns()
        end_tactical_turn(roster.get(turns.current))
        records: list[dict[str, Any] | None] = [None] * len(self.enemies)
        while True:
            _, would_wrap = turns.peek_next(roster.is_alive)
            if would_wrap and self.round_number == MAX_ROUNDS:
                self._truncated = True
                break
            if turns.advance(roster.is_alive):
                self.round_number = turns.round_number
            ref = turns.current
            actor = roster.get(ref)
            begin_tactical_turn(actor)
            if self._is_policy_controlled(ref):
                break
            engaged = self.ranks and archer_engaged(self.allies, self.enemies)
            result = run_enemy_turn(
                ref.slot, self.allies, self.enemies, self.np_random,
                heal=self.rules.heal, engaged_archer=engaged,
            )
            end_tactical_turn(actor)
            records[ref.slot] = _enemy_turn_record(result)
            if roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY):
                break
        return records

    # --- observation and info --------------------------------------------------

    def _active_ally_slot(self) -> int:
        turns = self._turns()
        if not self._terminated and not self._truncated and turns.current.side is Side.ALLY:
            return turns.current.slot
        for slot, ally in enumerate(self.allies):
            if ally.alive:
                return slot
        return 0

    def _observation(self) -> np.ndarray:
        values: list[float] = [float(self._active_ally_slot())]
        for ally in self.allies:
            values.extend((
                ally.hp,
                ally.resources.get(Resource.ACTION),
                ally.resources.get(Resource.BONUS_ACTION),
                ally.resources.get(Resource.SECOND_WIND),
                ally.resources.get(Resource.ACTION_SURGE),
                ally.resources.get(Resource.CLEAVE),
                ally.resources.get(Resource.TRIP),
                int(ally.dodging),
            ))
            if self.ranks:
                values.extend((ally.resources.get(Resource.MOVEMENT), int(ally.deep), int(ally.disengaged)))
        for enemy in self.enemies:
            values.extend((
                enemy.hp, enemy.max_hp, enemy.armor_class, enemy.attack_bonus,
                enemy.damage.die_size, enemy.damage.bonus,
            ))
            values.extend(int(enemy.role is role) for role in ROLE_ORDER)
            values.extend(int(enemy.targeting is policy) for policy in TARGETING_ORDER)
            values.extend((
                int(enemy.prone),
                enemy.resources.get(Resource.HEAL),
                0 if enemy.last_attacker is None else enemy.last_attacker + 1,
            ))
            if self.ranks:
                values.extend((int(enemy.rank is Rank.BACK), enemy.resources.get(Resource.REACTION)))
        values.append(self.round_number)
        return np.array(values, dtype=np.float32)

    def _info(self, decision: DecisionSpec | None) -> dict[str, Any]:
        turns = self._turns()
        payload: dict[str, Any] = {
            "round": self.round_number,
            "action": decision.action.name if decision else None,
            "decision": decision.label if decision else None,
            "target_index": decision.target_index if decision else None,
            "enemy_hps": [enemy.hp for enemy in self.enemies],
            "ally_hps": [ally.hp for ally in self.allies],
            "active_actor_index": self._active_ally_slot(),
            "active_actor_side": turns.current.side.name,
            "active_actor_slot": turns.current.slot,
            "turn_order": [f"{ref.side.name}/{ref.slot}" for ref in turns.order],
            "enemy_roles": [enemy.role.name for enemy in self.enemies],
            "enemy_targeting": [enemy.targeting.name for enemy in self.enemies],
            "enemy_prone": [enemy.prone for enemy in self.enemies],
            "ally_dodging": [ally.dodging for ally in self.allies],
        }
        if self.ranks:
            payload["ally_deep"] = [ally.deep for ally in self.allies]
        return payload


def _enemy_turn_record(result: EnemyTurnResult | None) -> dict[str, Any] | None:
    return asdict(result) if result is not None else None
