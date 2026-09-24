"""M0 combat rules. Callers supply all randomness and coordinate turns."""

from dataclasses import dataclass

from numpy.random import Generator

from .abilities import M0_ABILITIES
from .actions import Action
from .characters import Character, Fighter, Goblin
from .resources import Resource


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
    dice_count = attacker.damage.dice_count * (2 if critical else 1)
    dice_total = sum(int(rng.integers(1, attacker.damage.die_size + 1)) for _ in range(dice_count))
    return max(0, dice_total + attacker.damage.bonus)


def can_use_ability(actor: Character, action: Action) -> bool:
    """Check known ability and shared resource costs, not effect-specific rules."""
    spec = M0_ABILITIES.get(action)
    return spec is not None and action in actor.known_abilities and actor.resources.has(spec.costs)


def _spend_ability(actor: Character, action: Action) -> None:
    if not can_use_ability(actor, action):
        raise ValueError(f"{action.name} is not legal")
    actor.resources.spend(M0_ABILITIES[action].costs)


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


def use_attack(attacker: Character, defender: Character, rng: Generator) -> AttackResult:
    """Spend the attacker's Action and attack a living target."""
    if not defender.alive:
        raise ValueError("ATTACK is not legal")
    _spend_ability(attacker, Action.ATTACK)
    return resolve_attack(attacker, defender, rng)


def use_fighter_attack(fighter: Fighter, goblin: Goblin, rng: Generator) -> AttackResult:
    """Retain the M0 Fighter attack entry point."""
    return use_attack(fighter, goblin, rng)


def use_second_wind(fighter: Fighter, rng: Generator) -> int:
    """Spend Second Wind and the bonus action; return HP actually restored."""
    if fighter.hp >= fighter.max_hp:
        raise ValueError("SECOND_WIND is not legal")
    _spend_ability(fighter, Action.SECOND_WIND)
    healing = int(rng.integers(1, 11)) + 2
    restored = min(fighter.max_hp - fighter.hp, healing)
    fighter.hp += restored
    return restored


def use_action_surge(fighter: Fighter) -> None:
    """Spend Action Surge to make one action available."""
    _spend_ability(fighter, Action.ACTION_SURGE)
    fighter.resources.set(Resource.ACTION, 1)
