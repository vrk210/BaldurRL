"""Gymnasium interface for the fixed M0 Fighter-versus-Goblin encounter."""

from dataclasses import asdict, dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import Action
from .actors import ActorRef, CombatRoster, Side
from .characters import Fighter, Goblin, refresh_turn_resources
from .damage import DamageSpec
from .mechanics import AttackResult, can_use_ability, use_action_surge, use_attack, use_second_wind
from .resources import Resource
from .turns import TurnManager


MAX_ROUNDS = 50
_FIGHTER_MAX_HP = 20
_GOBLIN_MAX_HP = 15
M0_ACTIONS = (Action.ATTACK, Action.SECOND_WIND, Action.ACTION_SURGE, Action.END_TURN)
OBSERVATION_FIELDS = (
    "fighter_hp",
    "fighter_action_count",
    "fighter_bonus_action_count",
    "fighter_second_wind_count",
    "fighter_action_surge_count",
    "goblin_hp",
    "round_number",
)


@dataclass(frozen=True)
class RewardSnapshot:
    fighter_hp: int
    goblin_hp: int
    round_number: int
    fighter_resources: tuple[int, int, int, int]


RewardFunction = Callable[[RewardSnapshot, Action, RewardSnapshot, bool, bool], float]


def terminal_reward(
    before: RewardSnapshot, action: Action, after: RewardSnapshot, terminated: bool, truncated: bool
) -> float:
    """M0's sparse win/loss reward; truncation and intermediate steps score zero."""
    if terminated and after.goblin_hp == 0:
        return 1.0
    if terminated and after.fighter_hp == 0:
        return -1.0
    return 0.0


def _new_fighter() -> Fighter:
    return Fighter("Fighter", _FIGHTER_MAX_HP, _FIGHTER_MAX_HP, 16, 5, DamageSpec(1, 8, 3))


def _new_goblin() -> Goblin:
    return Goblin("Goblin", _GOBLIN_MAX_HP, _GOBLIN_MAX_HP, 15, 4, DamageSpec(1, 6, 2))


class BaldurCombatEnv(gym.Env[np.ndarray, int]):
    """One step applies exactly one Fighter decision."""

    metadata = {"render_modes": []}
    stage = "m0"
    action_names = tuple(action.name for action in M0_ACTIONS)
    observation_fields = OBSERVATION_FIELDS

    @property
    def decisions(self) -> tuple[Any, ...]:
        from .stages import DecisionSpec

        return tuple(DecisionSpec(action) for action in M0_ACTIONS)

    def __init__(self, reward_fn: RewardFunction = terminal_reward) -> None:
        super().__init__()
        self.reward_fn = reward_fn
        self.action_space = spaces.Discrete(len(M0_ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            high=np.array([_FIGHTER_MAX_HP, 2, 1, 1, 1, _GOBLIN_MAX_HP, MAX_ROUNDS], dtype=np.float32),
            dtype=np.float32,
        )
        self.fighter: Fighter | None = None
        self.goblin: Goblin | None = None
        self.roster: CombatRoster | None = None
        self.turns: TurnManager | None = None
        self._controlled_refs = frozenset({ActorRef(Side.ALLY, 0)})
        self.round_number = 1
        self._terminated = False
        self._truncated = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.fighter = _new_fighter()
        self.goblin = _new_goblin()
        self.round_number = 1
        self._terminated = False
        self._truncated = False
        self.roster = CombatRoster(allies=(self.fighter,), enemies=(self.goblin,))
        self.turns = TurnManager(
            (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0)), round_number=1
        )
        self._begin_fighter_turn()
        return self._get_observation(), self._get_info(action=None)

    def action_masks(self) -> np.ndarray:
        fighter, goblin = self._characters()
        if self._terminated or self._truncated or not fighter.alive or not goblin.alive:
            return np.zeros(len(M0_ACTIONS), dtype=np.bool_)
        return np.array(
            [
                can_use_ability(fighter, Action.ATTACK) and goblin.alive,
                can_use_ability(fighter, Action.SECOND_WIND) and fighter.hp < fighter.max_hp,
                can_use_ability(fighter, Action.ACTION_SURGE),
                fighter.alive and goblin.alive,
            ],
            dtype=np.bool_,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        fighter, goblin = self._characters()
        roster = self._roster()
        turns = self._turns()
        if self._terminated or self._truncated:
            raise RuntimeError("Episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid M0 action index: {action!r}")
        index = int(action)
        if not self.action_masks()[index]:
            raise ValueError(f"{M0_ACTIONS[index].name} is not legal now")

        semantic_action = M0_ACTIONS[index]
        before = self._reward_snapshot()
        fighter_attack: AttackResult | None = None
        goblin_attack: AttackResult | None = None
        healed: int | None = None

        if semantic_action is Action.ATTACK:
            fighter_attack = use_attack(fighter, goblin, self.np_random)
            self._terminated = roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY)
        elif semantic_action is Action.SECOND_WIND:
            healed = use_second_wind(fighter, self.np_random)
        elif semantic_action is Action.ACTION_SURGE:
            use_action_surge(fighter)
        else:
            turns.advance(roster.is_alive)
            while not self._is_policy_controlled(turns.current):
                result = self._run_automatic_turn(turns.current)
                if result is not None:
                    goblin_attack = result
                if roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY):
                    self._terminated = True
                    break
                _, would_wrap = turns.peek_next(roster.is_alive)
                if would_wrap and self.round_number == MAX_ROUNDS:
                    self._truncated = True
                    break
                wrapped = turns.advance(roster.is_alive)
                if wrapped:
                    self.round_number = turns.round_number
                    self._begin_fighter_turn()

        after = self._reward_snapshot()
        reward = float(self.reward_fn(before, semantic_action, after, self._terminated, self._truncated))
        info = self._get_info(semantic_action)
        if fighter_attack is not None:
            info["fighter_attack"] = asdict(fighter_attack)
        if goblin_attack is not None:
            info["goblin_attack"] = asdict(goblin_attack)
        if healed is not None:
            info["healed"] = healed
        return self._get_observation(), reward, self._terminated, self._truncated, info

    def _characters(self) -> tuple[Fighter, Goblin]:
        if self.fighter is None or self.goblin is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.fighter, self.goblin

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
        roster = self._roster()
        attacker = roster.get(ref)
        target = roster.get(ActorRef(Side.ALLY, 0))
        if not attacker.alive or not target.alive:
            return None
        refresh_turn_resources(attacker)
        return use_attack(attacker, target, self.np_random)

    def _begin_fighter_turn(self) -> None:
        fighter, _ = self._characters()
        refresh_turn_resources(fighter)

    def _get_observation(self) -> np.ndarray:
        fighter, goblin = self._characters()
        return np.array(
            [
                fighter.hp,
                fighter.resources.get(Resource.ACTION),
                fighter.resources.get(Resource.BONUS_ACTION),
                fighter.resources.get(Resource.SECOND_WIND),
                fighter.resources.get(Resource.ACTION_SURGE),
                goblin.hp,
                self.round_number,
            ],
            dtype=np.float32,
        )

    def _reward_snapshot(self) -> RewardSnapshot:
        fighter, goblin = self._characters()
        return RewardSnapshot(
            fighter_hp=fighter.hp,
            goblin_hp=goblin.hp,
            round_number=self.round_number,
            fighter_resources=tuple(fighter.resources.get(resource) for resource in (
                Resource.ACTION,
                Resource.BONUS_ACTION,
                Resource.SECOND_WIND,
                Resource.ACTION_SURGE,
            )),
        )

    def _get_info(self, action: Action | None) -> dict[str, Any]:
        fighter, goblin = self._characters()
        return {
            "round": self.round_number,
            "action": action.name if action is not None else None,
            "fighter_hp": fighter.hp,
            "goblin_hp": goblin.hp,
        }
