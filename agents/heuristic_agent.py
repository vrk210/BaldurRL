"""Simple deterministic Fighter policies for each combat stage."""

from functools import lru_cache

import numpy as np

from combat.damage import DamageSpec
from combat.env import _new_fighter
from combat.distributions import attack_damage_distribution

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


class M1AHeuristicAgent(HeuristicAgent):
    """The M0 priority rule applied unchanged to randomized 1v1 fights."""


class M1BHeuristicAgent:
    """Heal at low HP, otherwise attack the living target with lowest HP."""

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if not np.any(action_mask):
            raise ValueError("No legal actions available")
        hp = int(observation[0])
        actions = int(observation[1])
        if action_mask[2] and hp <= self.second_wind_threshold:
            return 2
        legal_targets = [index for index in (0, 1) if action_mask[index]]
        if legal_targets:
            return min(legal_targets, key=lambda index: (observation[5 + 6 * index], index))
        if action_mask[3] and actions == 0:
            return 3
        if action_mask[4]:
            return 4
        return _first_legal_index(action_mask)


class M2HeuristicAgent:
    """Use Cleave against two living enemies, then focus the lower-HP target."""

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if not np.any(action_mask):
            raise ValueError("No legal actions available")
        hp = int(observation[0])
        actions = int(observation[1])
        if action_mask[3] and hp <= self.second_wind_threshold:
            return 3
        # M2 inserts Cleave resource after the four other Fighter resource fields.
        enemy_0_hp, enemy_1_hp = observation[6], observation[12]
        if action_mask[2] and enemy_0_hp > 0 and enemy_1_hp > 0:
            return 2
        legal_targets = [index for index in (0, 1) if action_mask[index]]
        if legal_targets:
            return min(legal_targets, key=lambda index: (observation[6 + 6 * index], index))
        if action_mask[4] and actions == 0:
            return 4
        if action_mask[5]:
            return 5
        return _first_legal_index(action_mask)


class M3HeuristicAgent:
    """M2 priorities applied to whichever ally is active."""

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if not np.any(action_mask):
            raise ValueError("No legal actions available")
        base = 1 + 6 * int(observation[0])
        hp = int(observation[base])
        actions = int(observation[base + 1])
        if action_mask[3] and hp <= self.second_wind_threshold:
            return 3
        enemy_0_hp, enemy_1_hp = observation[13], observation[19]
        if action_mask[2] and enemy_0_hp > 0 and enemy_1_hp > 0:
            return 2
        legal_targets = [index for index in (0, 1) if action_mask[index]]
        if legal_targets:
            return min(legal_targets, key=lambda index: (observation[13 + 6 * index], index))
        if action_mask[4] and actions == 0:
            return 4
        if action_mask[5]:
            return 5
        return _first_legal_index(action_mask)


class M4HeuristicAgent:
    """M3 priorities against three enemies; Cleave while two or more live."""

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if not np.any(action_mask):
            raise ValueError("No legal actions available")
        base = 1 + 6 * int(observation[0])
        hp = int(observation[base])
        actions = int(observation[base + 1])
        if action_mask[4] and hp <= self.second_wind_threshold:
            return 4
        enemy_hps = (observation[13], observation[19], observation[25])
        if action_mask[3] and sum(h > 0 for h in enemy_hps) >= 2:
            return 3
        legal_targets = [index for index in (0, 1, 2) if action_mask[index]]
        if legal_targets:
            return min(legal_targets, key=lambda index: (observation[13 + 6 * index], index))
        if action_mask[5] and actions == 0:
            return 5
        if action_mask[6]:
            return 6
        return _first_legal_index(action_mask)


def make_heuristic_agent(stage: str) -> HeuristicAgent | M1AHeuristicAgent | M1BHeuristicAgent | M2HeuristicAgent | M3HeuristicAgent | M4HeuristicAgent:
    """Select the fixed, untuned priority policy for a stage."""
    agents = {
        "m0": HeuristicAgent,
        "m1a": M1AHeuristicAgent,
        "m1b": M1BHeuristicAgent,
        "m2": M2HeuristicAgent,
        "m3": M3HeuristicAgent,
        "m4": M4HeuristicAgent,
    }
    try:
        return agents[stage]()
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {stage}") from exc


class M4ThreatAwareAgent:
    """M4 heuristic with threat-aware targeting (promoted from the M4 policy probe).

    Same priorities as `M4HeuristicAgent` (Second Wind at <= threshold HP, Cleave
    while two or more enemies live, Action Surge when out of Actions), but attacks
    the living enemy minimizing expected attacks-to-kill divided by its expected
    damage per attack against a Fighter. Both expectations come from the exact
    attack-damage distribution in `combat.mechanics`; this reproduces
    `runs/m4_policy_diagnostic_5000_9999/m4_policy_probe.py` (threshold 10,
    targeting "threat", cleave_min 2).
    """

    def __init__(self, second_wind_threshold: int = 10) -> None:
        self.second_wind_threshold = second_wind_threshold
        self._fighter = _new_fighter()

    def _score(self, observation: np.ndarray, index: int) -> tuple[float, int]:
        start = 13 + 6 * index
        hp, _, ac, attack, die, bonus = (int(value) for value in observation[start:start + 6])
        fighter = self._fighter
        attacks = _attacks_to_kill(hp, ac, fighter.attack_bonus, fighter.damage)
        threat = _mean_damage(attack, fighter.armor_class, DamageSpec(1, die, bonus))
        return attacks / threat, index

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if not np.any(action_mask):
            raise ValueError("No legal actions available")
        base = 1 + 6 * int(observation[0])
        if action_mask[4] and observation[base] <= self.second_wind_threshold:
            return 4
        living = [index for index in range(3) if observation[13 + 6 * index] > 0]
        if action_mask[3] and len(living) >= 2:
            return 3
        targets = [index for index in range(3) if action_mask[index]]
        if targets:
            return min(targets, key=lambda index: self._score(observation, index))
        if action_mask[5] and observation[base + 1] == 0:
            return 5
        if action_mask[6]:
            return 6
        return _first_legal_index(action_mask)


@lru_cache(maxsize=None)
def _mean_damage(attack_bonus: int, armor_class: int, damage: DamageSpec) -> float:
    return sum(amount * p for amount, p in attack_damage_distribution(attack_bonus, armor_class, damage).items())


@lru_cache(maxsize=None)
def _attacks_to_kill(hp: int, armor_class: int, attack_bonus: int, damage: DamageSpec) -> float:
    """Exact expected number of attacks to reduce `hp` to zero against one AC."""
    if hp <= 0:
        return 0.0
    distribution = attack_damage_distribution(attack_bonus, armor_class, damage)
    remaining = sum(
        p * _attacks_to_kill(hp - amount, armor_class, attack_bonus, damage)
        for amount, p in distribution.items() if amount > 0
    )
    return (1 + remaining) / (1 - distribution.get(0, 0.0))
