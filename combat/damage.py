"""M0 damage dice and modifier as one value."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DamageSpec:
    dice_count: int
    die_size: int
    bonus: int
