"""Advantage/disadvantage d20 primitives and moded attack resolution (M5/M6).

Kept separate from ``combat.mechanics`` so M0–M4 rule code is untouched.
``roll_attack`` applies the same natural-1/natural-20/AC rules as
``mechanics.resolve_attack``, generalized to a roll mode; damage reuses
``mechanics.roll_damage``. Callers supply all randomness.
"""

from dataclasses import dataclass
from enum import Enum

from numpy.random import Generator

from .characters import Character
from .mechanics import roll_d20, roll_damage


class RollMode(Enum):
    """How many d20s an attack or check draws and which one it keeps."""

    NORMAL = "normal"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"


@dataclass(frozen=True)
class AttackRoll:
    """A resolved d20 attack roll before damage; ``d20_rolls`` lists every die drawn."""

    d20_rolls: tuple[int, ...]
    d20_roll: int
    total_attack: int
    hit: bool
    critical: bool


@dataclass(frozen=True)
class ModedAttackResult:
    """``mechanics.AttackResult`` counterpart that also records roll mode and every d20."""

    mode: str
    d20_rolls: tuple[int, ...]
    d20_roll: int
    hit: bool
    critical: bool
    damage: int
    total_attack: int


def combine_roll_mode(advantage: bool, disadvantage: bool) -> RollMode:
    """Any advantage plus any disadvantage cancel to NORMAL; sources never stack."""
    if advantage and not disadvantage:
        return RollMode.ADVANTAGE
    if disadvantage and not advantage:
        return RollMode.DISADVANTAGE
    return RollMode.NORMAL


def roll_d20_with_mode(rng: Generator, mode: RollMode) -> tuple[int, tuple[int, ...]]:
    """Return (kept die, all dice drawn in order); two dice unless NORMAL."""
    if mode is RollMode.NORMAL:
        roll = roll_d20(rng)
        return roll, (roll,)
    rolls = (roll_d20(rng), roll_d20(rng))
    return (max(rolls) if mode is RollMode.ADVANTAGE else min(rolls)), rolls


def roll_attack(
    attack_bonus: int, armor_class: int, rng: Generator, mode: RollMode = RollMode.NORMAL
) -> AttackRoll:
    """Natural 1 misses, natural 20 hits critically, otherwise compare total to AC."""
    d20_roll, rolls = roll_d20_with_mode(rng, mode)
    total_attack = d20_roll + attack_bonus
    critical = d20_roll == 20
    hit = critical or (d20_roll != 1 and total_attack >= armor_class)
    return AttackRoll(rolls, d20_roll, total_attack, hit, critical)


def resolve_attack_with_mode(
    attacker: Character,
    defender: Character,
    rng: Generator,
    mode: RollMode,
    *,
    damage_divisor: int = 1,
) -> ModedAttackResult:
    """Resolve one attack under a roll mode and apply any damage to the defender.

    Damage follows ``mechanics.resolve_attack``: critical hits double dice, a
    divisor halves resolved damage with minimum 1, and HP never drops below 0.
    """
    roll = roll_attack(attacker.attack_bonus, defender.armor_class, rng, mode)
    damage = 0
    if roll.hit:
        rolled = roll_damage(attacker, rng, critical=roll.critical)
        scaled = rolled if damage_divisor == 1 else max(1, rolled // damage_divisor)
        damage = min(defender.hp, scaled)
        defender.hp -= damage
    return ModedAttackResult(
        mode=mode.value,
        d20_rolls=roll.d20_rolls,
        d20_roll=roll.d20_roll,
        hit=roll.hit,
        critical=roll.critical,
        damage=damage,
        total_attack=roll.total_attack,
    )
