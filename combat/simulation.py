"""Restore independent terminal-reward simulators from live policy observations."""

import numpy as np

from .actors import ActorRef, Side
from .characters import Rank, Role, TacticalEnemy, TacticalFighter
from .damage import DamageSpec
from .env import BaldurCombatEnv
from .resources import Resource
from .stages import StagedCombatEnv, _enemy_profile, make_env
from .tactical_env import (
    N_ALLIES,
    N_ENEMIES,
    ROLE_ORDER,
    STAGE_RULES,
    TARGETING_ORDER,
    TacticalCombatEnv,
    has_ranks,
    make_tactical_ally,
    make_tactical_enemy,
)
from .turns import TurnManager


_RESOURCE_FIELDS = (
    ("action_count", Resource.ACTION),
    ("bonus_action_count", Resource.BONUS_ACTION),
    ("second_wind_count", Resource.SECOND_WIND),
    ("action_surge_count", Resource.ACTION_SURGE),
    ("cleave_count", Resource.CLEAVE),
)


def simulation_from_observation(
    stage: str, observation: np.ndarray, *, seed: int
) -> BaldurCombatEnv | StagedCombatEnv | TacticalCombatEnv:
    """Reconstruct a live decision without consulting any actual episode or RNG.

    Only observations at ongoing allied decisions are supported. In particular,
    a round-50 observation cannot distinguish a live decision from truncation;
    callers must supply a live decision. Presets/known abilities follow the stage
    contract. Enemy resources are unobserved but refresh before their next turn.
    """
    env = make_env(stage)
    if not isinstance(env, (BaldurCombatEnv, StagedCombatEnv, TacticalCombatEnv)):
        raise TypeError(f"Unsupported simulation environment for {stage}")
    values = np.asarray(observation)
    if (
        values.shape != env.observation_space.shape
        or not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
        or np.any(values < env.observation_space.low)
        or np.any(values > env.observation_space.high)
    ):
        raise ValueError(f"Invalid {stage} decision observation")
    fields = dict(zip(env.observation_fields, (int(value) for value in values)))
    if isinstance(env, TacticalCombatEnv):
        return _restore_tactical(env, fields, seed)
    # reset creates independent characters/resources and stage-specific abilities.
    # Its sampling is discarded; subsequent rolls start at the supplied seed.
    env.reset(seed=seed)
    env.np_random = np.random.default_rng(seed)
    roster = env._roster()
    for slot, ally in enumerate(roster.allies):
        prefix = f"ally_{slot}_" if stage in ("m3", "m4") else "fighter_"
        ally.hp = fields[prefix + "hp"]
        for field, resource in _RESOURCE_FIELDS:
            if prefix + field in fields:
                ally.resources.set(resource, fields[prefix + field])
    for slot, enemy in enumerate(roster.enemies):
        if stage == "m0":
            enemy.hp = fields["goblin_hp"]
            continue
        prefix = "enemy_" if stage == "m1a" else f"enemy_{slot}_"
        enemy.hp = fields[prefix + "hp"]
        enemy.max_hp = fields[prefix + "max_hp"]
        enemy.armor_class = fields[prefix + "ac"]
        enemy.attack_bonus = fields[prefix + "attack_bonus"]
        die = fields[prefix + "damage_die_size"]
        profile = _enemy_profile(stage)
        stats = (
            (enemy.max_hp, profile["hp"]),
            (enemy.armor_class, profile["ac"]),
            (enemy.attack_bonus, profile["attack"]),
            (fields[prefix + "damage_bonus"], profile["bonus"]),
        )
        if (
            not 0 <= enemy.hp <= enemy.max_hp
            or die not in profile["dice"]
            or any(not low <= value < high for value, (low, high) in stats)
        ):
            raise ValueError(f"Invalid enemy state in {stage} observation")
        enemy.damage = DamageSpec(1, die, fields[prefix + "damage_bonus"])
    active = ActorRef(Side.ALLY, fields.get("active_actor_index", 0))
    if roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY) or not roster.is_alive(active):
        raise ValueError("Simulation requires an ongoing living ally decision")
    env.round_number = fields["round_number"]
    env.turns = TurnManager(env._turns().order, env.round_number, current=active)
    if isinstance(env, StagedCombatEnv):
        env._initial_enemy_hp = sum(enemy.max_hp for enemy in roster.enemies)
    return env


def _one_hot(fields: dict[str, int], names: tuple[str, ...]) -> int:
    flags = [fields[name] for name in names]
    if sorted(flags) != [0] * (len(flags) - 1) + [1]:
        raise ValueError(f"Expected exactly one of {names}")
    return flags.index(1)


def _restore_tactical_ally(stage: str, fields: dict[str, int], slot: int) -> TacticalFighter:
    prefix = f"ally_{slot}_"
    ally = make_tactical_ally(stage)
    ally.hp = fields[prefix + "hp"]
    for field, resource in _RESOURCE_FIELDS + (("trip_count", Resource.TRIP),):
        ally.resources.set(resource, fields[prefix + field])
    ally.dodging = bool(fields[prefix + "dodging"])
    if has_ranks(stage):
        ally.resources.set(Resource.MOVEMENT, fields[prefix + "movement_count"])
        ally.deep = bool(fields[prefix + "deep"])
        ally.disengaged = bool(fields[prefix + "disengaged"])
    return ally


def _restore_tactical_enemy(stage: str, fields: dict[str, int], slot: int) -> TacticalEnemy:
    prefix = f"enemy_{slot}_"
    role = ROLE_ORDER[_one_hot(fields, tuple(prefix + f"is_{r.name.lower()}" for r in ROLE_ORDER))]
    targeting = TARGETING_ORDER[_one_hot(
        fields, tuple(prefix + f"targets_{t.name.lower()}" for t in TARGETING_ORDER)
    )]
    rules = STAGE_RULES[stage]
    profile = rules.profiles[role]
    hp, max_hp, ac, attack, die, bonus = (fields[prefix + name] for name in (
        "hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus"
    ))
    if (
        not 0 <= hp <= max_hp
        or die != profile.die
        or any(not low <= value <= high for value, (low, high) in (
            (max_hp, profile.hp), (ac, profile.ac), (attack, profile.attack), (bonus, profile.bonus)
        ))
    ):
        raise ValueError(f"Invalid {role.name} statistics in {stage} observation")
    enemy = make_tactical_enemy(stage, slot, role, targeting, max_hp, ac, attack, bonus)
    enemy.hp = hp
    enemy.prone = bool(fields[prefix + "prone"])
    heal_count = fields[prefix + "heal_count"]
    if heal_count > (rules.heal_charges if role is Role.HEALER else 0):
        raise ValueError(f"Invalid heal charges for {role.name} in {stage} observation")
    if role is Role.HEALER:
        enemy.resources.set(Resource.HEAL, heal_count)
    last = fields[prefix + "last_attacker"]
    enemy.last_attacker = None if last == 0 else last - 1
    if has_ranks(stage):
        if fields[prefix + "back_rank"] != int(enemy.rank is Rank.BACK):
            raise ValueError(f"Invalid rank for {role.name} in {stage} observation")
        enemy.resources.set(Resource.REACTION, fields[prefix + "reaction_count"])
    return enemy


def _restore_tactical(env: TacticalCombatEnv, fields: dict[str, int], seed: int) -> TacticalCombatEnv:
    """Rebuild M5/M6 state; every transition-relevant fact is in the observation."""
    stage = env.stage
    allies = tuple(_restore_tactical_ally(stage, fields, slot) for slot in range(N_ALLIES))
    enemies = tuple(_restore_tactical_enemy(stage, fields, slot) for slot in range(N_ENEMIES))
    if sorted(enemy.role.value for enemy in enemies) != [role.value for role in ROLE_ORDER]:
        raise ValueError(f"{stage} observation must contain one enemy of each role")
    active = ActorRef(Side.ALLY, fields["active_actor_index"])
    env._install(allies, enemies, round_number=fields["round_number"], current=active)
    roster = env._roster()
    if roster.side_defeated(Side.ALLY) or roster.side_defeated(Side.ENEMY) or not roster.is_alive(active):
        raise ValueError("Simulation requires an ongoing living ally decision")
    env.np_random = np.random.default_rng(seed)
    return env
