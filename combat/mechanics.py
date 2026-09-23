"""M0 combat rules. Callers supply all randomness and coordinate turns."""

from dataclasses import dataclass

from numpy.random import Generator

from .characters import Character, Fighter, Goblin


@dataclass(frozen=True)
class AttackResult:
    d20_roll: int
    hit: bool
    critical: bool
    damage: int
    total_attack: int


def roll_d20(rng: Generator) -> int:
    return int(rng.integers(1, 21))


def roll_damage(attacker: Character, rng: Generator, critical: bool = False) -> int:
    """Roll damage, doubling dice (but not the bonus) on a critical hit."""
    dice_count = attacker.damage_dice_count * (2 if critical else 1)
    dice_total = sum(int(rng.integers(1, attacker.damage_die_size + 1)) for _ in range(dice_count))
    return max(0, dice_total + attacker.damage_bonus)


def resolve_attack(attacker: Character, defender: Character, rng: Generator) -> AttackResult:
    """Resolve one attack and apply any damage to the defender's HP."""
    d20_roll = roll_d20(rng)
    total_attack = d20_roll + attacker.attack_bonus
    critical = d20_roll == 20
    hit = critical or (d20_roll != 1 and total_attack >= defender.armor_class)
    if not hit:
        return AttackResult(d20_roll=d20_roll, hit=False, critical=False, damage=0, total_attack=total_attack)
    rolled_damage = roll_damage(attacker, rng, critical=critical)
    damage = min(defender.hp, rolled_damage)
    defender.hp -= damage
    return AttackResult(d20_roll=d20_roll, hit=True, critical=critical, damage=damage, total_attack=total_attack)


def use_fighter_attack(fighter: Fighter, goblin: Goblin, rng: Generator) -> AttackResult:
    """Spend the Fighter's action and attack the Goblin."""
    if not fighter.action_available or not goblin.alive:
        raise ValueError("ATTACK is not legal")
    fighter.action_available = False
    return resolve_attack(fighter, goblin, rng)


def use_second_wind(fighter: Fighter, rng: Generator) -> int:
    """Spend Second Wind and the bonus action; return HP actually restored."""
    if not (fighter.second_wind_available and fighter.bonus_action_available and fighter.hp < fighter.max_hp):
        raise ValueError("SECOND_WIND is not legal")
    fighter.bonus_action_available = False
    fighter.second_wind_available = False
    healing = int(rng.integers(1, 11)) + 2
    restored = min(fighter.max_hp - fighter.hp, healing)
    fighter.hp += restored
    return restored


def use_action_surge(fighter: Fighter) -> None:
    """Spend Action Surge to make one action available."""
    if not fighter.action_surge_available:
        raise ValueError("ACTION_SURGE is not legal")
    fighter.action_surge_available = False
    fighter.action_available = True
