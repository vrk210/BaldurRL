"""Small stage selector and Gymnasium environments for M1A, M1B, and M2."""

from dataclasses import asdict, dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import Action
from .characters import Fighter, Goblin
from .damage import DamageSpec
from .env import BaldurCombatEnv, MAX_ROUNDS, _new_fighter
from .mechanics import can_use_ability, use_action_surge, use_attack, use_cleave, use_second_wind
from .resources import Resource


@dataclass(frozen=True)
class DecisionSpec:
    action: Action
    target_index: int | None = None

    @property
    def label(self) -> str:
        return self.action.name if self.target_index is None else f"{self.action.name}_ENEMY_{self.target_index}"


_FIGHTER_FIELDS = (
    "fighter_hp", "fighter_action_count", "fighter_bonus_action_count",
    "fighter_second_wind_count", "fighter_action_surge_count",
)
_ENEMY_FIELDS = ("hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus")


def _fields(stage: str) -> tuple[str, ...]:
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
    targeted = (DecisionSpec(Action.ATTACK, 0), DecisionSpec(Action.ATTACK, 1))
    rest = (DecisionSpec(Action.SECOND_WIND), DecisionSpec(Action.ACTION_SURGE), DecisionSpec(Action.END_TURN))
    return targeted + ((DecisionSpec(Action.CLEAVE),) if stage == "m2" else ()) + rest


class StagedCombatEnv(gym.Env[np.ndarray, int]):
    """One Fighter decision per step against one or two randomized enemies."""

    metadata = {"render_modes": []}

    def __init__(self, stage: str, reward_mode: str = "terminal") -> None:
        if stage not in ("m1a", "m1b", "m2"):
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
        fighter_high = [20, 2, 1, 1, 1] + ([1] if stage == "m2" else [])
        enemy_high = [20, 20, 18, 6, 8, 4] * (1 if stage == "m1a" else 2)
        self.observation_space = spaces.Box(
            low=np.array([0] * (len(self.observation_fields) - 1) + [1], dtype=np.float32),
            high=np.array(fighter_high + enemy_high + [MAX_ROUNDS], dtype=np.float32),
            dtype=np.float32,
        )
        self.fighter: Fighter | None = None
        self.enemies: tuple[Goblin, ...] = ()
        self.round_number = 1
        self._terminated = False
        self._truncated = False
        self._initial_enemy_hp = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.fighter = _new_fighter()
        if self.stage == "m2":
            self.fighter.resources.set(Resource.CLEAVE, 1)
            self.fighter.known_abilities = self.fighter.known_abilities | {Action.CLEAVE}
        self.enemies = tuple(self._new_enemy(slot) for slot in range(1 if self.stage == "m1a" else 2))
        self._initial_enemy_hp = sum(enemy.max_hp for enemy in self.enemies)
        self.round_number = 1
        self._terminated = False
        self._truncated = False
        return self._observation(), self._info(None)

    def _new_enemy(self, slot: int) -> Goblin:
        hp = int(self.np_random.integers(8, 21))
        ac = int(self.np_random.integers(12, 19))
        bonus = int(self.np_random.integers(2, 7))
        die = (4, 6, 8)[int(self.np_random.integers(0, 3))]
        damage_bonus = int(self.np_random.integers(0, 5))
        return Goblin(f"Enemy {slot}", hp, hp, ac, bonus, DamageSpec(1, die, damage_bonus))

    def _fighter(self) -> Fighter:
        if self.fighter is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.fighter

    def action_masks(self) -> np.ndarray:
        fighter = self._fighter()
        if self._terminated or self._truncated or not fighter.alive or not any(enemy.alive for enemy in self.enemies):
            return np.zeros(len(self.decisions), dtype=np.bool_)
        mask = []
        for decision in self.decisions:
            action = decision.action
            if action is Action.ATTACK:
                target = self.enemies[decision.target_index or 0]
                legal = can_use_ability(fighter, action) and target.alive
            elif action is Action.CLEAVE:
                legal = can_use_ability(fighter, action) and any(enemy.alive for enemy in self.enemies)
            elif action is Action.SECOND_WIND:
                legal = can_use_ability(fighter, action) and fighter.hp < fighter.max_hp
            elif action is Action.ACTION_SURGE:
                legal = can_use_ability(fighter, action)
            else:
                legal = True
            mask.append(legal)
        return np.array(mask, dtype=np.bool_)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        fighter = self._fighter()
        if self._terminated or self._truncated:
            raise RuntimeError("Episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid {self.stage} action index: {action!r}")
        index = int(action)
        decision = self.decisions[index]
        if not self.action_masks()[index]:
            raise ValueError(f"{decision.label} is not legal now")

        before_enemy_hp = sum(enemy.hp for enemy in self.enemies)
        info: dict[str, Any] = {}
        if decision.action is Action.ATTACK:
            target = self.enemies[decision.target_index or 0]
            info["fighter_attack"] = asdict(use_attack(fighter, target, self.np_random))
        elif decision.action is Action.CLEAVE:
            results = use_cleave(fighter, self.enemies, self.np_random)
            info["cleave_attacks"] = [asdict(result) if result is not None else None for result in results]
            info["cleave_living_targets"] = sum(result is not None for result in results)
        elif decision.action is Action.SECOND_WIND:
            info["healed"] = use_second_wind(fighter, self.np_random)
        elif decision.action is Action.ACTION_SURGE:
            use_action_surge(fighter)
        else:
            attacks: list[dict[str, Any] | None] = []
            for enemy in self.enemies:
                if not fighter.alive:
                    attacks.append(None)
                    continue
                if enemy.alive:
                    enemy.resources.set(Resource.ACTION, 1)
                    attacks.append(asdict(use_attack(enemy, fighter, self.np_random)))
                else:
                    attacks.append(None)
            info["enemy_attacks"] = attacks
            if len(self.enemies) == 1:
                info["goblin_attack"] = attacks[0]
            if fighter.alive:
                if self.round_number == MAX_ROUNDS:
                    self._truncated = True
                else:
                    self.round_number += 1
                    fighter.resources.set(Resource.ACTION, 1)
                    fighter.resources.set(Resource.BONUS_ACTION, 1)

        self._terminated = not fighter.alive or not any(enemy.alive for enemy in self.enemies)
        reward = float((1 if fighter.alive else -1) if self._terminated else 0)
        if self.reward_mode == "damage":
            damage = before_enemy_hp - sum(enemy.hp for enemy in self.enemies)
            reward += 0.2 * damage / self._initial_enemy_hp
        info.update(self._info(decision))
        return self._observation(), reward, self._terminated, self._truncated, info

    def _observation(self) -> np.ndarray:
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
        return {
            "round": self.round_number,
            "action": decision.action.name if decision else None,
            "decision": decision.label if decision else None,
            "target_index": decision.target_index if decision else None,
            "fighter_hp": fighter.hp,
            "enemy_hps": [enemy.hp for enemy in self.enemies],
            "goblin_hp": self.enemies[0].hp,
        }


STAGES: dict[str, Callable[..., gym.Env]] = {
    "m0": BaldurCombatEnv,
    "m1a": lambda reward_mode="terminal": StagedCombatEnv("m1a", reward_mode),
    "m1b": lambda reward_mode="terminal": StagedCombatEnv("m1b", reward_mode),
    "m2": lambda reward_mode="terminal": StagedCombatEnv("m2", reward_mode),
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
