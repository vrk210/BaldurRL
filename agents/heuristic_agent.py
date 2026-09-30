"""Deterministic heuristic policy for the M0 Fighter-vs-Goblin encounter."""

import numpy as np

_ATTACK = 0
_SECOND_WIND = 1
_ACTION_SURGE = 2
_END_TURN = 3

_FIGHTER_HP = 0
_FIGHTER_ACTION_COUNT = 1
_FIGHTER_MAX_HP = 20


def _first_legal_index(action_mask: np.ndarray) -> int:
    legal_indices = np.flatnonzero(action_mask)
    if legal_indices.size == 0:
        raise ValueError("No legal actions available")
    return int(legal_indices[0])


class HeuristicAgent:
    """Priority policy: heal when low, attack, surge for an extra attack, end turn."""

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if np.flatnonzero(action_mask).size == 0:
            raise ValueError("No legal actions available")
        fighter_hp = int(observation[_FIGHTER_HP])
        fighter_action_count = int(observation[_FIGHTER_ACTION_COUNT])
        if (
            bool(action_mask[_SECOND_WIND])
            and fighter_hp <= self.second_wind_threshold
            and fighter_hp < _FIGHTER_MAX_HP
        ):
            return _SECOND_WIND
        if bool(action_mask[_ATTACK]):
            return _ATTACK
        if bool(action_mask[_ACTION_SURGE]) and fighter_action_count == 0:
            return _ACTION_SURGE
        if bool(action_mask[_END_TURN]):
            return _END_TURN
        return _first_legal_index(action_mask)
