"""M5/M6 combat rules: roles, conditions, reach, enemy behavior, opportunity attacks.

Every rule here is a SIMULATOR CHOICE specified in docs/STAGES_SPEC.md (M5 and
M6 sections), not a verified BG3 rule. Callers supply all randomness, decide
legality through ``combat.legality.tactical_legal_mask``, and coordinate turns.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from numpy.random import Generator

from .abilities import TACTICAL_ABILITIES
from .actions import Action
from .characters import (
    Character,
    Rank,
    Role,
    TacticalEnemy,
    TacticalFighter,
    Targeting,
    refresh_turn_resources,
)
from .damage import DamageSpec
from .resources import Resource
from .rolls import ModedAttackResult, RollMode, combine_roll_mode, resolve_attack_with_mode, roll_attack


@dataclass(frozen=True)
class RoleProfile:
    """Inclusive per-role sampling ranges, fixed damage die, and M6 rank."""

    hp: tuple[int, int]
    ac: tuple[int, int]
    attack: tuple[int, int]
    die: int
    bonus: tuple[int, int]
    rank: Rank


@dataclass(frozen=True)
class TacticalRules:
    """Per-stage tuning data: role profiles, Healer dice and charges, Trip charges."""

    profiles: Mapping[Role, RoleProfile]
    heal: DamageSpec
    heal_charges: int
    trip_charges: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        if set(self.profiles) != set(Role):
            raise ValueError("Every role needs a profile")


@dataclass(frozen=True)
class TripResult:
    d20_roll: int
    total: int
    success: bool


@dataclass(frozen=True)
class OpportunityAttack:
    enemy_slot: int
    attack: ModedAttackResult


@dataclass(frozen=True)
class EnemyTurnResult:
    """One automatic enemy turn: ``kind`` is ``"attack"`` or ``"heal"``."""

    kind: str
    target_slot: int
    attack: ModedAttackResult | None = None
    healed: int | None = None


# --- Ability costs and healing -------------------------------------------------


def can_use_tactical(actor: Character, action: Action) -> bool:
    """Known ability and ``TACTICAL_ABILITIES`` costs, not effect-specific rules."""
    spec = TACTICAL_ABILITIES.get(action)
    return spec is not None and action in actor.known_abilities and actor.resources.has(spec.costs)


def spend_tactical(actor: Character, action: Action) -> None:
    """Spend an M5/M6 ability's catalog costs, refusing unknown or unaffordable ones."""
    if not can_use_tactical(actor, action):
        raise ValueError(f"{action.name} is not legal")
    actor.resources.spend(TACTICAL_ABILITIES[action].costs)


def heal_character(character: Character, amount: int) -> int:
    """Restore up to ``amount`` HP to a living character; return HP actually restored."""
    if not character.alive:
        raise ValueError("Cannot heal a dead character")
    if amount < 0:
        raise ValueError("Healing cannot be negative")
    restored = min(character.max_hp - character.hp, amount)
    character.hp += restored
    return restored


# --- Reach and engagement (M6 ranks; M5 enemies are all front rank) ---------


def front_broken(enemies: Sequence[TacticalEnemy]) -> bool:
    """True when no living enemy remains in the front rank."""
    return not any(enemy.alive and enemy.rank is Rank.FRONT for enemy in enemies)


def can_reach(ally: TacticalFighter, enemy: TacticalEnemy, enemies: Sequence[TacticalEnemy]) -> bool:
    """Melee reach: front-rank enemy, a deep ally, or a broken front."""
    return enemy.rank is Rank.FRONT or ally.deep or front_broken(enemies)


def archer_engaged(allies: Sequence[TacticalFighter], enemies: Sequence[TacticalEnemy]) -> bool:
    """M6 only: back-rank Archers are engaged by any deep ally or a broken front."""
    return front_broken(enemies) or any(ally.alive and ally.deep for ally in allies)


def advance_available(ally: TacticalFighter, enemies: Sequence[TacticalEnemy]) -> bool:
    """Effect conditions shared by ADVANCE and DISENGAGE (costs checked separately)."""
    return (
        not ally.deep
        and not front_broken(enemies)
        and any(enemy.alive and enemy.rank is Rank.BACK for enemy in enemies)
    )


# --- Roll modes ---------------------------------------------------------------


def ally_attack_mode(target: TacticalEnemy) -> RollMode:
    """Ally attacks are melee: advantage against a Prone target."""
    return combine_roll_mode(advantage=target.prone, disadvantage=False)


def enemy_attack_mode(
    attacker: TacticalEnemy, target: TacticalFighter, *, engaged_archer: bool
) -> RollMode:
    """Disadvantage if Prone, the target is Dodging, or an engaged Archer attacks."""
    disadvantage = (
        attacker.prone
        or target.dodging
        or (engaged_archer and attacker.role is Role.ARCHER)
    )
    return combine_roll_mode(advantage=False, disadvantage=disadvantage)


# --- Ally effects --------------------------------------------------------------


def can_attack(ally: TacticalFighter, target: TacticalEnemy, enemies: Sequence[TacticalEnemy]) -> bool:
    return target.alive and can_reach(ally, target, enemies)


def can_trip(ally: TacticalFighter, target: TacticalEnemy, enemies: Sequence[TacticalEnemy]) -> bool:
    return can_attack(ally, target, enemies) and not target.prone


def cleave_targets(ally: TacticalFighter, enemies: Sequence[TacticalEnemy]) -> tuple[bool, ...]:
    """Which slots a Cleave would attack; reach is fixed before the first roll."""
    return tuple(can_attack(ally, enemy, enemies) for enemy in enemies)


def use_tactical_attack(
    ally_slot: int,
    ally: TacticalFighter,
    target: TacticalEnemy,
    enemies: Sequence[TacticalEnemy],
    rng: Generator,
) -> ModedAttackResult:
    """Spend the Action and make one melee attack; record the last attacker."""
    if not can_attack(ally, target, enemies):
        raise ValueError("ATTACK is not legal")
    spend_tactical(ally, Action.ATTACK)
    target.last_attacker = ally_slot
    return resolve_attack_with_mode(ally, target, rng, ally_attack_mode(target))


def use_trip(
    ally_slot: int,
    ally: TacticalFighter,
    target: TacticalEnemy,
    enemies: Sequence[TacticalEnemy],
    rng: Generator,
) -> TripResult:
    """Spend the Bonus Action and a Trip charge; a NORMAL roll against AC knocks the target Prone."""
    if not can_trip(ally, target, enemies):
        raise ValueError("TRIP is not legal")
    spend_tactical(ally, Action.TRIP)
    target.last_attacker = ally_slot
    roll = roll_attack(ally.attack_bonus, target.armor_class, rng)
    if roll.hit:
        target.prone = True
    return TripResult(d20_roll=roll.d20_roll, total=roll.total_attack, success=roll.hit)


def use_tactical_cleave(
    ally_slot: int, ally: TacticalFighter, enemies: Sequence[TacticalEnemy], rng: Generator
) -> tuple[ModedAttackResult | None, ...]:
    """Spend Cleave and attack every reachable living enemy for halved damage."""
    targets = cleave_targets(ally, enemies)
    if not any(targets):
        raise ValueError("CLEAVE is not legal")
    spend_tactical(ally, Action.CLEAVE)
    results: list[ModedAttackResult | None] = []
    for enemy, attacked in zip(enemies, targets):
        if not attacked:
            results.append(None)
            continue
        enemy.last_attacker = ally_slot
        results.append(
            resolve_attack_with_mode(ally, enemy, rng, ally_attack_mode(enemy), damage_divisor=2)
        )
    return tuple(results)


def use_dodge(ally: TacticalFighter) -> None:
    """Spend the Action; attacks against the ally have disadvantage until its next turn."""
    if ally.dodging:
        raise ValueError("DODGE is not legal")
    spend_tactical(ally, Action.DODGE)
    ally.dodging = True


def can_disengage(ally: TacticalFighter, enemies: Sequence[TacticalEnemy]) -> bool:
    return (
        not ally.disengaged
        and ally.resources.get(Resource.MOVEMENT) >= 1
        and advance_available(ally, enemies)
    )


def use_disengage(ally: TacticalFighter, enemies: Sequence[TacticalEnemy]) -> None:
    """Spend the Action; this turn's ADVANCE provokes no opportunity attacks."""
    if not can_disengage(ally, enemies):
        raise ValueError("DISENGAGE is not legal")
    spend_tactical(ally, Action.DISENGAGE)
    ally.disengaged = True


def use_advance(
    ally: TacticalFighter,
    allies: Sequence[TacticalFighter],
    enemies: Sequence[TacticalEnemy],
    rng: Generator,
) -> tuple[OpportunityAttack, ...]:
    """Spend Movement, take opportunity attacks unless Disengaged, then move deep.

    Front-rank enemies with a Reaction attack in slot order; once the ally
    drops to 0 HP no further opportunity attacks are made and it stays put.
    """
    if not advance_available(ally, enemies):
        raise ValueError("ADVANCE is not legal")
    spend_tactical(ally, Action.ADVANCE)
    attacks: list[OpportunityAttack] = []
    if not ally.disengaged:
        engaged = archer_engaged(allies, enemies)
        for slot, enemy in enumerate(enemies):
            if not ally.alive:
                break
            if enemy.alive and enemy.rank is Rank.FRONT and enemy.resources.get(Resource.REACTION) >= 1:
                enemy.resources.spend({Resource.REACTION: 1})
                mode = enemy_attack_mode(enemy, ally, engaged_archer=engaged)
                attacks.append(OpportunityAttack(slot, resolve_attack_with_mode(enemy, ally, rng, mode)))
    if ally.alive:
        ally.deep = True
    return tuple(attacks)


# --- Enemy behavior --------------------------------------------------------------


def weakest_ally(allies: Sequence[TacticalFighter]) -> int | None:
    """Living ally with the lowest current HP; ties go to the lowest slot."""
    living = [slot for slot, ally in enumerate(allies) if ally.alive]
    if not living:
        return None
    return min(living, key=lambda slot: (allies[slot].hp, slot))


def choose_enemy_target(
    enemy: TacticalEnemy, allies: Sequence[TacticalFighter], rng: Generator
) -> int | None:
    """Apply the enemy's targeting policy; RANDOM draws only with two living allies."""
    living = [slot for slot, ally in enumerate(allies) if ally.alive]
    if not living:
        return None
    if enemy.targeting is Targeting.RETALIATE:
        last = enemy.last_attacker
        if last is not None and allies[last].alive:
            return last
        return weakest_ally(allies)
    if enemy.targeting is Targeting.RANDOM:
        if len(living) == 1:
            return living[0]
        return living[int(rng.integers(0, len(living)))]
    return weakest_ally(allies)


def heal_target(enemies: Sequence[TacticalEnemy]) -> int | None:
    """Most-missing-HP living enemy at or below half max HP; ties to the lowest slot."""
    eligible = [slot for slot, enemy in enumerate(enemies) if enemy.alive and 2 * enemy.hp <= enemy.max_hp]
    if not eligible:
        return None
    return min(eligible, key=lambda slot: (-(enemies[slot].max_hp - enemies[slot].hp), slot))


def use_heal(healer: TacticalEnemy, target: TacticalEnemy, heal: DamageSpec, rng: Generator) -> int:
    """Spend the Action and one heal charge; restore the heal dice, capped at max HP."""
    if not target.alive:
        raise ValueError("HEAL is not legal")
    spend_tactical(healer, Action.HEAL)
    rolled = sum(int(rng.integers(1, heal.die_size + 1)) for _ in range(heal.dice_count))
    return heal_character(target, rolled + heal.bonus)


def run_enemy_turn(
    slot: int,
    allies: Sequence[TacticalFighter],
    enemies: Sequence[TacticalEnemy],
    rng: Generator,
    *,
    heal: DamageSpec,
    engaged_archer: bool,
) -> EnemyTurnResult | None:
    """Healers heal an eligible enemy if they can; otherwise attack per targeting."""
    enemy = enemies[slot]
    if not enemy.alive:
        return None
    if can_use_tactical(enemy, Action.HEAL):
        target = heal_target(enemies)
        if target is not None:
            return EnemyTurnResult("heal", target, healed=use_heal(enemy, enemies[target], heal, rng))
    target = choose_enemy_target(enemy, allies, rng)
    if target is None:
        return None
    spend_tactical(enemy, Action.ATTACK)
    mode = enemy_attack_mode(enemy, allies[target], engaged_archer=engaged_archer)
    return EnemyTurnResult("attack", target, attack=resolve_attack_with_mode(enemy, allies[target], rng, mode))


# --- Turn lifecycle ----------------------------------------------------------------


def begin_tactical_turn(actor: Character) -> None:
    """Refresh per-turn resources; an ally's Dodging and Disengaged end."""
    refresh_turn_resources(actor)
    if isinstance(actor, TacticalFighter):
        actor.dodging = False
        actor.disengaged = False


def end_tactical_turn(actor: Character) -> None:
    """An enemy stands up from Prone; an ally's Disengaged ends."""
    if isinstance(actor, TacticalEnemy):
        actor.prone = False
    if isinstance(actor, TacticalFighter):
        actor.disengaged = False
