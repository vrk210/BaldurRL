"""M0 combat rules. Callers supply all randomness and coordinate turns.

The small rule functions (`attack_hit`, `damage_dice_count`, `damage_from_dice`,
`scale_damage`, `SECOND_WIND_HEALING`) are shared with the exact outcome
distributions in `distributions.py`, so planners cannot drift from sampling.
"""

from dataclasses import dataclass

from numpy.random import Generator

from .abilities import M2_ABILITIES
from .actions import Action
from .characters import Character, Fighter, Goblin
from .damage import DamageSpec
from .resources import Resource


D20_SIZE = 20
# Second Wind heals 1d10 + 2 for the M0 Level 2 Fighter preset.
SECOND_WIND_HEALING = DamageSpec(1, 10, 2)


@dataclass(frozen=True)
class AttackResult:
    d20_roll: int
    hit: bool
    critical: bool
    damage: int
    total_attack: int


def roll_d20(rng: Generator) -> int:
    return int(rng.integers(1, D20_SIZE + 1))


def attack_hit(d20_roll: int, attack_bonus: int, armor_class: int) -> tuple[bool, bool]:
    """Return (hit, critical): natural 1 misses, natural 20 hits critically."""
    critical = d20_roll == D20_SIZE
    hit = critical or (d20_roll != 1 and d20_roll + attack_bonus >= armor_class)
    return hit, critical


def damage_dice_count(damage: DamageSpec, critical: bool) -> int:
    """A critical hit doubles the number of damage dice, not the bonus."""
    return damage.dice_count * (2 if critical else 1)


def damage_from_dice(dice_total: int, damage: DamageSpec) -> int:
    return max(0, dice_total + damage.bonus)


def scale_damage(rolled_damage: int, damage_divisor: int) -> int:
    """Cleave-style halving keeps at least 1 damage on a hit."""
    return rolled_damage if damage_divisor == 1 else max(1, rolled_damage // damage_divisor)


def roll_damage(attacker: Character, rng: Generator, critical: bool = False) -> int:
    """Roll damage, doubling dice (but not the bonus) on a critical hit."""
    dice_count = damage_dice_count(attacker.damage, critical)
    dice_total = sum(int(rng.integers(1, attacker.damage.die_size + 1)) for _ in range(dice_count))
    return damage_from_dice(dice_total, attacker.damage)


def can_use_ability(actor: Character, action: Action) -> bool:
    """Check known ability and shared resource costs, not effect-specific rules."""
    spec = M2_ABILITIES.get(action)
    return spec is not None and action in actor.known_abilities and actor.resources.has(spec.costs)


def _spend_ability(actor: Character, action: Action) -> None:
    if not can_use_ability(actor, action):
        raise ValueError(f"{action.name} is not legal")
    actor.resources.spend(M2_ABILITIES[action].costs)


def resolve_attack(
    attacker: Character, defender: Character, rng: Generator, *, damage_divisor: int = 1
) -> AttackResult:
    """Resolve one attack and apply any damage to the defender's HP."""
    d20_roll = roll_d20(rng)
    total_attack = d20_roll + attacker.attack_bonus
    hit, critical = attack_hit(d20_roll, attacker.attack_bonus, defender.armor_class)
    if not hit:
        return AttackResult(d20_roll=d20_roll, hit=False, critical=False, damage=0, total_attack=total_attack)
    rolled_damage = roll_damage(attacker, rng, critical=critical)
    damage = min(defender.hp, scale_damage(rolled_damage, damage_divisor))
    defender.hp -= damage
    return AttackResult(d20_roll=d20_roll, hit=True, critical=critical, damage=damage, total_attack=total_attack)


def use_attack(attacker: Character, defender: Character, rng: Generator) -> AttackResult:
    """Spend the attacker's Action and attack a living target."""
    if not defender.alive:
        raise ValueError("ATTACK is not legal")
    _spend_ability(attacker, Action.ATTACK)
    return resolve_attack(attacker, defender, rng)


def use_fighter_attack(fighter: Fighter, goblin: Goblin, rng: Generator) -> AttackResult:
    """Retain the M0 Fighter attack entry point."""
    return use_attack(fighter, goblin, rng)


def use_second_wind(character: Character, rng: Generator) -> int:
    """Spend Second Wind and the bonus action; return HP actually restored."""
    if character.hp >= character.max_hp:
        raise ValueError("SECOND_WIND is not legal")
    _spend_ability(character, Action.SECOND_WIND)
    rolled = sum(
        int(rng.integers(1, SECOND_WIND_HEALING.die_size + 1))
        for _ in range(SECOND_WIND_HEALING.dice_count)
    )
    healing = damage_from_dice(rolled, SECOND_WIND_HEALING)
    restored = min(character.max_hp - character.hp, healing)
    character.hp += restored
    return restored


def use_action_surge(character: Character) -> None:
    """Spend Action Surge to grant one additional Action."""
    _spend_ability(character, Action.ACTION_SURGE)
    character.resources.gain(Resource.ACTION)


def use_cleave(
    attacker: Character, targets: tuple[Character, ...], rng: Generator
) -> tuple[AttackResult | None, ...]:
    """Spend one Cleave and attack each living target with halved resolved damage."""
    if not any(target.alive for target in targets):
        raise ValueError("CLEAVE is not legal")
    _spend_ability(attacker, Action.CLEAVE)
    return tuple(
        resolve_attack(attacker, target, rng, damage_divisor=2) if target.alive else None
        for target in targets
    )
