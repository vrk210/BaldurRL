"""Restore independent terminal-reward simulators from live policy observations."""

import numpy as np

from .actors import ActorRef, Side
from .damage import DamageSpec
from .env import BaldurCombatEnv
from .resources import Resource
from .stages import StagedCombatEnv, _enemy_profile, make_env
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
) -> BaldurCombatEnv | StagedCombatEnv:
    """Reconstruct a live decision without consulting any actual episode or RNG.

    Only observations at ongoing allied decisions are supported. In particular,
    a round-50 observation cannot distinguish a live decision from truncation;
    callers must supply a live decision. Presets/known abilities follow the stage
    contract. Enemy resources are unobserved but refresh before their next turn.
    """
    env = make_env(stage)
    if not isinstance(env, (BaldurCombatEnv, StagedCombatEnv)):
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
