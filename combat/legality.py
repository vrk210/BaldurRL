"""Per-actor targeting and legality without turns, dice, or rewards."""

from collections.abc import Sequence
from dataclasses import dataclass

from .actions import Action
from .actors import ActorRef, CombatRoster, Side
from .characters import TacticalEnemy, TacticalFighter
from .mechanics import can_use_ability
from .tactics import (
    advance_available,
    can_attack,
    can_disengage,
    can_trip,
    can_use_tactical,
    cleave_targets,
)


@dataclass(frozen=True)
class DecisionSpec:
    action: Action
    target_index: int | None = None

    @property
    def label(self) -> str:
        return self.action.name if self.target_index is None else f"{self.action.name}_ENEMY_{self.target_index}"


def _opposing_side(side: Side) -> Side:
    return Side.ENEMY if side is Side.ALLY else Side.ALLY


def resolve_target(acting: ActorRef, target_index: int | None, roster: CombatRoster) -> ActorRef:
    """Resolve a decision target slot to an ActorRef on the opposing side.

    A None index addresses slot 0, preserving single-opponent decisions.
    """
    side = _opposing_side(acting.side)
    opponents = roster.enemies if side is Side.ENEMY else roster.allies
    slot = 0 if target_index is None else target_index
    if not 0 <= slot < len(opponents):
        raise IndexError(f"Target slot {slot} is out of range for {side.name}")
    return ActorRef(side, slot)


def legal_mask_for(
    actor: ActorRef, roster: CombatRoster, decisions: Sequence[DecisionSpec]
) -> list[bool]:
    """Legal decisions for one actor on its turn, independent of turn order.

    Turn-agnostic by design: callers only query the active actor (gate at
    the caller). Episode-over (terminated/truncated) stays with the env.
    Uses the same known-ability and cost checks as execution, so a legal
    mask never disagrees with the mechanics it gates. A decision whose
    target slot does not exist for the acting side is illegal, not an
    error, so any stage decision list is evaluable for any actor.
    """
    character = roster.get(actor)
    opponents = roster.enemies if actor.side is Side.ALLY else roster.allies
    if not character.alive or not any(opponent.alive for opponent in opponents):
        return [False] * len(decisions)
    mask = []
    for decision in decisions:
        action = decision.action
        if action is Action.ATTACK:
            try:
                target = roster.get(resolve_target(actor, decision.target_index, roster))
            except IndexError:
                legal = False
            else:
                legal = can_use_ability(character, action) and target.alive
        elif action is Action.CLEAVE:
            legal = can_use_ability(character, action) and any(
                opponent.alive for opponent in opponents
            )
        elif action is Action.SECOND_WIND:
            legal = can_use_ability(character, action) and character.hp < character.max_hp
        elif action is Action.ACTION_SURGE:
            legal = can_use_ability(character, action)
        elif action is Action.END_TURN:
            legal = character.alive and any(opponent.alive for opponent in opponents)
        else:
            raise ValueError(f"Unknown action: {action!r}")
        mask.append(legal)
    return mask


def tactical_legal_mask(
    ally_slot: int,
    allies: Sequence[TacticalFighter],
    enemies: Sequence[TacticalEnemy],
    decisions: Sequence[DecisionSpec],
) -> list[bool]:
    """M5/M6 legal decisions for one ally on its turn (see docs/STAGES_SPEC.md).

    Combines known abilities and catalog costs with each effect's own
    conditions from ``combat.tactics`` (reach, Prone, Dodging, ranks), so the
    mask never disagrees with the mechanics it gates. Turn gating stays with
    the caller, as for ``legal_mask_for``.
    """
    ally = allies[ally_slot]
    if not ally.alive or not any(enemy.alive for enemy in enemies):
        return [False] * len(decisions)

    def affordable(action: Action) -> bool:
        return can_use_tactical(ally, action)

    mask = []
    for decision in decisions:
        action = decision.action
        target = None if decision.target_index is None else enemies[decision.target_index]
        if action is Action.ATTACK and target is not None:
            legal = affordable(action) and can_attack(ally, target, enemies)
        elif action is Action.TRIP and target is not None:
            legal = affordable(action) and can_trip(ally, target, enemies)
        elif action is Action.CLEAVE:
            legal = affordable(action) and any(cleave_targets(ally, enemies))
        elif action is Action.DODGE:
            legal = affordable(action) and not ally.dodging
        elif action is Action.SECOND_WIND:
            legal = affordable(action) and ally.hp < ally.max_hp
        elif action is Action.ACTION_SURGE:
            legal = affordable(action)
        elif action is Action.ADVANCE:
            legal = affordable(action) and advance_available(ally, enemies)
        elif action is Action.DISENGAGE:
            legal = affordable(action) and can_disengage(ally, enemies)
        elif action is Action.END_TURN:
            legal = True
        else:
            raise ValueError(f"Unsupported tactical decision: {decision!r}")
        mask.append(legal)
    return mask


def automatic_attack_decision(
    actor: ActorRef, roster: CombatRoster, decisions: Sequence[DecisionSpec]
) -> DecisionSpec | None:
    """Fixed automatic-turn policy: the first legal ATTACK decision in stage order.

    The same rule as the environments' `_run_automatic_turn`; exact transition
    enumeration uses it (tests check the two agree through `env.step`).
    """
    for decision, legal in zip(decisions, legal_mask_for(actor, roster, decisions)):
        if legal and decision.action is Action.ATTACK:
            return decision
    return None
