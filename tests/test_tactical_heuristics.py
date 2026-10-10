"""Documented priorities of the naive and smarter M5/M6 heuristics."""

import pytest

from agents.heuristic_agent import make_heuristic_agent
from agents.tactical_heuristics import (
    VARIANTS,
    TacticalHeuristicAgent,
    TacticalHeuristicConfig,
    expected_damage,
    make_tactical_heuristic,
)
from combat.characters import Role
from combat.resources import Resource
from combat.stages import make_env
from evaluation.evaluate import evaluate


def state(stage: str, seed: int):
    env = make_env(stage)
    env.reset(seed=seed)
    return env


def choose(agent, env) -> str:
    return env.action_names[agent.choose_action(env._observation(), env.action_masks())]


def slot_of(env, role: Role) -> int:
    return next(slot for slot, enemy in enumerate(env.enemies) if enemy.role is role)


def test_naive_applies_m4_priorities_and_ignores_new_options() -> None:
    env = state("m5", 7)  # BRUTE 32, HEALER 12, ARCHER 10
    naive = make_heuristic_agent("m5")
    assert choose(naive, env) == "CLEAVE"  # three living enemies
    env.allies[0].resources.set(Resource.CLEAVE, 0)
    assert choose(naive, env) == "ATTACK_ENEMY_2"  # lowest HP
    env.allies[0].hp = 10
    assert choose(naive, env) == "SECOND_WIND"
    env.allies[0].resources.set(Resource.BONUS_ACTION, 0)
    env.allies[0].resources.set(Resource.ACTION, 0)
    assert choose(naive, env) == "ACTION_SURGE"
    env.allies[0].resources.set(Resource.ACTION_SURGE, 0)
    assert choose(naive, env) == "END_TURN"  # never trips or dodges


def test_trip_focus_trips_its_target_first_and_tuned_saves_cleave() -> None:
    env = state("m5", 7)
    env.allies[0].resources.set(Resource.CLEAVE, 0)
    agent = make_tactical_heuristic("m5", "trip_focus")
    assert choose(agent, env) == "TRIP_ENEMY_2"
    env.enemies[2].prone = True
    assert choose(agent, env) == "ATTACK_ENEMY_2"
    env = state("m5", 7)
    tuned = make_tactical_heuristic("m5", "tuned")
    env.enemies[1].hp = 0  # two living enemies: tuned waits for three to Cleave
    assert choose(tuned, env) == "TRIP_ENEMY_2"
    env.enemies[2].prone = True
    assert choose(tuned, env) == "ATTACK_ENEMY_2"
    env.allies[0].hp = 12
    assert choose(tuned, env) == "SECOND_WIND"


def test_healer_first_targets_a_healer_with_charges() -> None:
    env = state("m5", 7)
    env.allies[0].resources.set(Resource.CLEAVE, 0)
    env.allies[0].resources.set(Resource.TRIP, 0)
    agent = make_tactical_heuristic("m5", "healer_first")
    healer, archer = slot_of(env, Role.HEALER), slot_of(env, Role.ARCHER)
    assert choose(agent, env) == f"ATTACK_ENEMY_{healer}"
    env.enemies[healer].resources.set(Resource.HEAL, 0)
    assert choose(agent, env) == f"ATTACK_ENEMY_{archer}"


def test_m6_hold_dive_and_adaptive_movement() -> None:
    env = state("m6", 4)  # BRUTE in front
    brute = env.enemies[slot_of(env, Role.BRUTE)]
    assert choose(make_tactical_heuristic("m6", "naive"), env) == "CLEAVE"  # counts every living enemy
    assert choose(make_tactical_heuristic("m6", "hold"), env).startswith("TRIP_ENEMY_0")
    assert choose(make_tactical_heuristic("m6", "dive_all"), env) == "ADVANCE"
    adaptive = make_tactical_heuristic("m6", "adaptive")
    brute.armor_class, brute.hp = 16, 26  # 26 / 0.5 = 52 hits-equivalent >= 42: dive
    assert choose(adaptive, env) == "ADVANCE"
    brute.armor_class, brute.hp = 14, 20  # 20 / 0.6 = 33: hold
    assert choose(adaptive, env) != "ADVANCE"
    env.allies[1].deep = True  # commit: follow a partner who already dove
    assert choose(adaptive, env) == "ADVANCE"


def test_factory_variants_and_validation() -> None:
    assert set(VARIANTS["m5"]) == {"naive", "trip_focus", "healer_first", "tuned"}
    assert set(VARIANTS["m6"]) == {"naive", "hold", "dive_all", "adaptive"}
    assert make_heuristic_agent("m6", "hold").config == VARIANTS["m6"]["hold"]
    with pytest.raises(ValueError, match="variant"):
        make_heuristic_agent("m5", "bogus")
    with pytest.raises(ValueError, match="m5 and m6"):
        make_heuristic_agent("m4", "tuned")
    with pytest.raises(ValueError):
        TacticalHeuristicConfig(priority="nearest")
    with pytest.raises(ValueError):
        TacticalHeuristicConfig(advance="sometimes")


def test_expected_damage_estimate() -> None:
    assert expected_damage(5, 8, 3, 16) == pytest.approx(0.5 * 7.5 + 0.05 * 4.5)
    assert expected_damage(5, 8, 3, 16, advantage=True) > expected_damage(5, 8, 3, 16)
    assert expected_damage(5, 8, 3, 16, disadvantage=True) < expected_damage(5, 8, 3, 16)
    assert expected_damage(5, 8, 3, 16, advantage=True, disadvantage=True) == expected_damage(5, 8, 3, 16)


@pytest.mark.parametrize("stage", ["m5", "m6"])
def test_every_variant_completes_seeded_evaluation(stage) -> None:
    for name in VARIANTS[stage]:
        agent = TacticalHeuristicAgent(stage, VARIANTS[stage][name])
        first = evaluate(agent, range(5), stage=stage)
        assert first == evaluate(make_tactical_heuristic(stage, name), range(5), stage=stage)
        assert first.wins + first.losses + first.truncations == 5


def test_development_knobs_used_in_the_tuning_history() -> None:
    env = state("m5", 7)  # targeting: Brute WEAKEST, Healer RANDOM, Archer RANDOM
    env.allies[0].hp = 6
    env.allies[0].resources.set(Resource.SECOND_WIND, 0)
    dodger = TacticalHeuristicAgent("m5", TacticalHeuristicConfig(trip=True, dodge=True))
    assert choose(dodger, env) == "DODGE"  # weakest ally expecting lethal damage
    env.allies[0].hp = 20
    assert choose(dodger, env) != "DODGE"
    defensive = TacticalHeuristicAgent("m5", TacticalHeuristicConfig(trip=True, trip_target="threat"))
    # Threat scores a Healer with charges by its mean heal (2d6+2 = 9), above the Archer's ~5.1.
    assert choose(defensive, env) == f"TRIP_ENEMY_{slot_of(env, Role.HEALER)}"
    env.enemies[slot_of(env, Role.HEALER)].resources.set(Resource.HEAL, 0)
    # Brute +7, 1d10+5 (~6.6 expected) out-threatens Archer +8, 1d8+3 (~5.1).
    assert choose(defensive, env) == f"TRIP_ENEMY_{slot_of(env, Role.BRUTE)}"

    env = state("m6", 4)
    env.allies[1].hp = 20
    env.allies[0].hp = 15
    single = TacticalHeuristicAgent("m6", TacticalHeuristicConfig(trip=True, advance="healthiest"))
    assert choose(single, env) != "ADVANCE"  # ally 1 is healthier and will dive instead
    env.allies[0].hp = 20
    assert choose(single, env) == "ADVANCE"
    careful = TacticalHeuristicAgent("m6", TacticalHeuristicConfig(trip=True, advance="all", disengage=True))
    assert choose(careful, env) == "DISENGAGE"  # Action Surge still available
    env.allies[0].resources.set(Resource.ACTION_SURGE, 0)
    assert choose(careful, env) == "ADVANCE"
