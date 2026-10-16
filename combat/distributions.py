"""Exact chance-outcome distributions built from the rule functions in `mechanics.py`.

Each helper enumerates every die face through the same functions the sampling
mechanics call, using exact fractions before converting to floats.
"""

from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from .damage import DamageSpec
from .mechanics import (
    D20_SIZE,
    SECOND_WIND_HEALING,
    attack_hit,
    damage_dice_count,
    damage_from_dice,
    scale_damage,
)


@lru_cache(maxsize=None)
def _dice_sum_distribution(dice_count: int, die_size: int) -> Mapping[int, Fraction]:
    sums = {0: Fraction(1)}
    for _ in range(dice_count):
        rolled: dict[int, Fraction] = {}
        for total, probability in sums.items():
            for face in range(1, die_size + 1):
                rolled[total + face] = rolled.get(total + face, Fraction(0)) + probability / die_size
        sums = rolled
    return MappingProxyType(sums)


@lru_cache(maxsize=None)
def attack_damage_distribution(
    attack_bonus: int, armor_class: int, damage: DamageSpec, damage_divisor: int = 1
) -> Mapping[int, float]:
    """Exact distribution of one attack's resolved damage before the defender-HP cap.

    Key 0 collects misses (and any zero-damage hit). Enumerates every d20 face and
    damage-dice sum through the same rule functions `resolve_attack` samples with.
    """
    outcomes: dict[int, Fraction] = {}
    for d20_roll in range(1, D20_SIZE + 1):
        hit, critical = attack_hit(d20_roll, attack_bonus, armor_class)
        if not hit:
            outcomes[0] = outcomes.get(0, Fraction(0)) + Fraction(1, D20_SIZE)
            continue
        dice = _dice_sum_distribution(damage_dice_count(damage, critical), damage.die_size)
        for dice_total, probability in dice.items():
            dealt = scale_damage(damage_from_dice(dice_total, damage), damage_divisor)
            outcomes[dealt] = outcomes.get(dealt, Fraction(0)) + probability / D20_SIZE
    return MappingProxyType({amount: float(p) for amount, p in sorted(outcomes.items())})


@lru_cache(maxsize=None)
def second_wind_healing_distribution() -> Mapping[int, float]:
    """Exact distribution of rolled Second Wind healing before the max-HP cap."""
    dice = _dice_sum_distribution(SECOND_WIND_HEALING.dice_count, SECOND_WIND_HEALING.die_size)
    return MappingProxyType({
        damage_from_dice(total, SECOND_WIND_HEALING): float(p) for total, p in sorted(dice.items())
    })
