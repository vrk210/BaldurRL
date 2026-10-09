"""Small stage selector and Gymnasium environments for M1A, M1B, M2, M3, and M4."""

from dataclasses import asdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import Action
from .actors import ActorRef, CombatRoster, Side
from .characters import Fighter, Goblin, refresh_turn_resources
from .damage import DamageSpec
from .env import BaldurCombatEnv, MAX_ROUNDS, _advance_to_next_turn, _new_fighter
from .legality import DecisionSpec, legal_mask_for, resolve_target
from .mechanics import (
    AttackResult,
    use_action_surge,
    use_attack,
    use_cleave,
    use_second_wind,
)
from .resources import Resource
from .turns import TurnManager


_FIGHTER_FIELDS = (
    "fighter_hp", "fighter_action_count", "fighter_bonus_action_count",
    "fighter_second_wind_count", "fighter_action_surge_count",
)
_ENEMY_FIELDS = ("hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus")
_ALLY_FIELDS = (
    "hp", "action_count", "bonus_action_count",
    "second_wind_count", "action_surge_count", "cleave_count",
)
# Per-stage enemy sampling: (low, high) exclusive-upper bounds plus die choices.
# M4 scales toward Fighter parity (20 HP, +5 attack, ~7.5 damage) as a first
# fairness guess for 2v3; every other stage keeps the original M1B profile.
_DEFAULT_ENEMY_PROFILE = {"hp": (8, 21), "ac": (12, 19), "attack": (2, 7),
                          "dice": (4, 6, 8), "bonus": (0, 5)}
_ENEMY_PROFILES = {
    "m4": {"hp": (12, 25), "ac": (12, 19), "attack": (3, 8),
           "dice": (4, 6, 8), "bonus": (2, 7)},
}
_ALLY_COUNTS = {"m3": 2, "m4": 2}
_ENEMY_COUNTS = {"m1a": 1, "m4": 3}
_CLEAVE_STAGES = ("m2", "m3", "m4")


def _enemy_profile(stage: str) -> dict[str, tuple[int, ...]]:
    return _ENEMY_PROFILES.get(stage, _DEFAULT_ENEMY_PROFILE)


def _enemy_high(stage: str) -> list[int]:
    profile = _enemy_profile(stage)
    hp, ac, attack, dice, bonus = (profile["hp"], profile["ac"], profile["attack"],
                                  profile["dice"], profile["bonus"])
    return [hp[1] - 1, hp[1] - 1, ac[1] - 1, attack[1] - 1, max(dice), bonus[1] - 1]


def _fields(stage: str) -> tuple[str, ...]:
    if stage in ("m3", "m4"):
        n_enemies = _ENEMY_COUNTS.get(stage, 2)
        allies = tuple(f"ally_{slot}_{field}" for slot in range(2) for field in _ALLY_FIELDS)
        enemies = tuple(f"enemy_{slot}_{field}" for slot in range(n_enemies) for field in _ENEMY_FIELDS)
        return ("active_actor_index",) + allies + enemies + ("round_number",)
    fighter = _FIGHTER_FIELDS + (("fighter_cleave_count",) if stage == "m2" else ())
    if stage == "m1a":
        enemy = tuple(f"enemy_{field}" for field in _ENEMY_FIELDS)
    else:
        enemy = tuple(f"enemy_{slot}_{field}" for slot in range(2) for field in _ENEMY_FIELDS)
    return fighter + enemy + ("round_number",)


def _decisions(stage: str) -> tuple[DecisionSpec, ...]:
    if stage == "m1a":
        return tuple(DecisionSpec(action) for action in (
            Action.ATTACK, Action.SECOND_WIND, Action.ACTION_SURGE, Action.END_TURN
        ))
    n_enemies = _ENEMY_COUNTS.get(stage, 2)
    targeted = tuple(DecisionSpec(Action.ATTACK, slot) for slot in range(n_enemies))
    rest = (DecisionSpec(Action.SECOND_WIND), DecisionSpec(Action.ACTION_SURGE), DecisionSpec(Action.END_TURN))
    cleave = ((DecisionSpec(Action.CLEAVE),) if stage in _CLEAVE_STAGES else ())
    return targeted + cleave + rest


class StagedCombatEnv(gym.Env[np.ndarray, int]):
    """One allied decision per step; M3/M4 control two allies against two/three enemies."""

    metadata = {"render_modes": []}

    def __init__(self, stage: str, reward_mode: str = "terminal") -> None:
        if stage not in ("m1a", "m1b", "m2", "m3", "m4"):
            raise ValueError(f"Unknown staged environment: {stage}")
        if reward_mode not in ("terminal", "damage") or (reward_mode == "damage" and stage != "m2"):
            raise ValueError(f"Reward mode {reward_mode!r} is unavailable for {stage}")
        super().__init__()
        self.stage = stage
        self.reward_mode = reward_mode
        self.decisions = _decisions(stage)
        self.action_names = tuple(decision.label for decision in self.decisions)
        self.observation_fields = _fields(stage)
        self.action_space = spaces.Discrete(len(self.decisions))
        if stage in ("m3", "m4"):
            n_enemies = _ENEMY_COUNTS.get(stage, 2)
            high = [1] + [20, 2, 1, 1, 1, 1] * 2 + _enemy_high(stage) * n_enemies + [MAX_ROUNDS]
        else:
            fighter_high = [20, 2, 1, 1, 1] + ([1] if stage == "m2" else [])
            enemy_high = _enemy_high(stage) * (1 if stage == "m1a" else 2)
            high = fighter_high + enemy_high + [MAX_ROUNDS]
        self.observation_space = spaces.Box(
            low=np.array([0] * (len(self.observation_fields) - 1) + [1], dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.fighter: Fighter | None = None
        self.allies: tuple[Fighter, ...] = ()
        self.enemies: tuple[Goblin, ...] = ()
        self.roster: CombatRoster | None = None
        self.turns: TurnManager | None = None
        n_allies = _ALLY_COUNTS.get(stage, 1)
        self._controlled_refs = frozenset(ActorRef(Side.ALLY, slot) for slot in range(n_allies))
        self.round_number = 1
        self._terminated = False
        self._truncated = False
        self._initial_enemy_hp = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.allies = tuple(self._new_ally() for _ in range(_ALLY_COUNTS.get(self.stage, 1)))
        self.fighter = self.allies[0]
        self.enemies = tuple(self._new_enemy(slot) for slot in range(_ENEMY_COUNTS.get(self.stage, 2)))
        self._initial_enemy_hp = sum(enemy.max_hp for enemy in self.enemies)
        self.round_number = 1
        self._terminated = False
        self._truncated = False
        self.roster = CombatRoster(allies=self.allies, enemies=self.enemies)
        order = tuple(
            ActorRef(Side.ALLY, slot) for slot in range(len(self.allies))
        ) + tuple(
            ActorRef(Side.ENEMY, slot) for slot in range(len(self.enemies))
        )
        self.turns = TurnManager(order, round_number=1)
        refresh_turn_resources(self.roster.get(self.turns.current))
        return self._observation(), self._info(None)

    def _new_ally(self) -> Fighter:
        ally = _new_fighter()
        if self.stage in _CLEAVE_STAGES:
            ally.resources.set(Resource.CLEAVE, 1)
            ally.known_abilities = ally.known_abilities | {Action.CLEAVE}
        return ally

    def _new_enemy(self, slot: int) -> Goblin:
        profile = _enemy_profile(self.stage)
        hp = int(self.np_random.integers(*profile["hp"]))
        ac = int(self.np_random.integers(*profile["ac"]))
        bonus = int(self.np_random.integers(*profile["attack"]))
        dice = profile["dice"]
        die = dice[int(self.np_random.integers(0, len(dice)))]
        damage_bonus = int(self.np_random.integers(*profile["bonus"]))
        return Goblin(f"Enemy {slot}", hp, hp, ac, bonus, DamageSpec(1, die, damage_bonus))

    def _fighter(self) -> Fighter:
        if self.fighter is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.fighter

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

    def _run_automatic_turn(self, ref: ActorRef) -> AttackResult | None:
        """Fixed policy via the shared mask path: first legal ATTACK in stage order."""
        roster = self._roster()
        attacker = roster.get(ref)
        if not attacker.alive:
            return None
        mask = legal_mask_for(ref, roster, self.decisions)
        for decision, legal in zip(self.decisions, mask):
            if legal and decision.action is Action.ATTACK:
                target = roster.get(resolve_target(ref, decision.target_index, roster))
                return use_attack(attacker, target, self.np_random)
        return None

    def action_masks(self) -> np.ndarray:
        if self._terminated or self._truncated:
            return np.zeros(len(self.decisions), dtype=np.bool_)
        turns = self._turns()
        if not self._is_policy_controlled(turns.current):
            return np.zeros(len(self.decisions), dtype=np.bool_)
        return np.array(
            legal_mask_for(turns.current, self._roster(), self.decisions),
            dtype=np.bool_,
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

        acting = turns.current
        attacker = roster.get(acting)
        opponents = roster.enemies if acting.side is Side.ALLY else roster.allies
        before_enemy_hp = sum(enemy.hp for enemy in self.enemies)
        info: dict[str, Any] = {}
        if decision.action is Action.ATTACK:
            target = roster.get(resolve_target(acting, decision.target_index, roster))
            info["fighter_attack"] = asdict(use_attack(attacker, target, self.np_random))
        elif decision.action is Action.CLEAVE:
            results = use_cleave(attacker, opponents, self.np_random)
            info["cleave_attacks"] = [asdict(result) if result is not None else None for result in results]
            info["cleave_living_targets"] = sum(result is not None for result in results)
        elif decision.action is Action.SECOND_WIND:
            info["healed"] = use_second_wind(attacker, self.np_random)
        elif decision.action is Action.ACTION_SURGE:
            use_action_surge(attacker)
        else:
            attacks: list[dict[str, Any] | None] = [None] * len(self.enemies)
            _advance_to_next_turn(turns, roster)
            while not self._is_policy_controlled(turns.current):
                ref = turns.current
                result = self._run_automatic_turn(ref)
                attacks[ref.slot] = asdict(result) if result is not None else None
                if roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY):
                    self._terminated = True
                    break
                _, would_wrap = turns.peek_next(roster.is_alive)
                if would_wrap and self.round_number == MAX_ROUNDS:
                    self._truncated = True
                    break
                wrapped = _advance_to_next_turn(turns, roster)
                if wrapped:
                    self.round_number = turns.round_number
            info["enemy_attacks"] = attacks
            if len(self.enemies) == 1:
                info["goblin_attack"] = attacks[0]

        allies_down = roster.side_defeated(Side.ALLY)
        enemies_down = roster.side_defeated(Side.ENEMY)
        self._terminated = allies_down or enemies_down
        if allies_down and enemies_down:
            # Current mechanics cannot cause this; an injected mutual defeat is a loss.
            reward = -1.0
        elif enemies_down:
            reward = 1.0
        elif allies_down:
            reward = -1.0
        else:
            reward = 0.0
        if self.reward_mode == "damage":
            damage = before_enemy_hp - sum(enemy.hp for enemy in self.enemies)
            reward += 0.2 * damage / self._initial_enemy_hp
        info.update(self._info(decision))
        return self._observation(), reward, self._terminated, self._truncated, info

    def _active_ally_slot(self) -> int:
        turns = self._turns()
        if not self._terminated and not self._truncated and turns.current.side is Side.ALLY:
            return turns.current.slot
        for slot, ally in enumerate(self._roster().allies):
            if ally.alive:
                return slot
        return 0

    def _observation(self) -> np.ndarray:
        if self.stage in ("m3", "m4"):
            values: list[float] = [float(self._active_ally_slot())]
            for ally in self._roster().allies:
                values.extend((
                    ally.hp,
                    ally.resources.get(Resource.ACTION),
                    ally.resources.get(Resource.BONUS_ACTION),
                    ally.resources.get(Resource.SECOND_WIND),
                    ally.resources.get(Resource.ACTION_SURGE),
                    ally.resources.get(Resource.CLEAVE),
                ))
            for enemy in self.enemies:
                values.extend((enemy.hp, enemy.max_hp, enemy.armor_class, enemy.attack_bonus,
                               enemy.damage.die_size, enemy.damage.bonus))
            values.append(self.round_number)
            return np.array(values, dtype=np.float32)
        fighter = self._fighter()
        values = [
            fighter.hp,
            fighter.resources.get(Resource.ACTION),
            fighter.resources.get(Resource.BONUS_ACTION),
            fighter.resources.get(Resource.SECOND_WIND),
            fighter.resources.get(Resource.ACTION_SURGE),
        ]
        if self.stage == "m2":
            values.append(fighter.resources.get(Resource.CLEAVE))
        for enemy in self.enemies:
            values.extend((enemy.hp, enemy.max_hp, enemy.armor_class, enemy.attack_bonus,
                           enemy.damage.die_size, enemy.damage.bonus))
        values.append(self.round_number)
        return np.array(values, dtype=np.float32)

    def _info(self, decision: DecisionSpec | None) -> dict[str, Any]:
        fighter = self._fighter()
        turns = self._turns()
        payload: dict[str, Any] = {
            "round": self.round_number,
            "action": decision.action.name if decision else None,
            "decision": decision.label if decision else None,
            "target_index": decision.target_index if decision else None,
            "enemy_hps": [enemy.hp for enemy in self.enemies],
            "goblin_hp": self.enemies[0].hp,
            "active_actor_side": turns.current.side.name,
            "active_actor_slot": turns.current.slot,
            "turn_order": [f"{ref.side.name}/{ref.slot}" for ref in turns.order],
        }
        if self.stage in ("m3", "m4"):
            payload["ally_hps"] = [ally.hp for ally in self._roster().allies]
            payload["active_actor_index"] = self._active_ally_slot()
        else:
            payload["fighter_hp"] = fighter.hp
        return payload


STAGES: dict[str, Callable[..., gym.Env]] = {
    "m0": BaldurCombatEnv,
    "m1a": lambda reward_mode="terminal": StagedCombatEnv("m1a", reward_mode),
    "m1b": lambda reward_mode="terminal": StagedCombatEnv("m1b", reward_mode),
    "m2": lambda reward_mode="terminal": StagedCombatEnv("m2", reward_mode),
    "m3": lambda reward_mode="terminal": StagedCombatEnv("m3", reward_mode),
    "m4": lambda reward_mode="terminal": StagedCombatEnv("m4", reward_mode),
}


def make_env(stage: str, reward_mode: str = "terminal") -> gym.Env:
    try:
        factory = STAGES[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown combat stage: {stage}") from exc
    if stage == "m0":
        if reward_mode != "terminal":
            raise ValueError("M0 supports terminal reward only")
        return factory()
    return factory(reward_mode=reward_mode)
