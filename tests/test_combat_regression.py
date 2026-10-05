"""Seeded regression snapshots for pre-refactor stage behavior.

These lock HP, rounds, targets, attack rolls/damage, resources, enemy order,
and termination for fixed action sequences so the turn-model refactor cannot
silently alter RNG consumption or turn ordering.
"""

from combat.env import BaldurCombatEnv
from combat.resources import Resource
from combat.stages import StagedCombatEnv


def _enemy_stats(env: StagedCombatEnv) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(
        (e.max_hp, e.armor_class, e.attack_bonus, e.damage.die_size, e.damage.bonus)
        for e in env.enemies
    )


def test_m1b_seed2_attack_surge_attack_end_turn() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    assert _enemy_stats(env) == ((18, 13, 2, 4, 2), (18, 15, 2, 6, 3))

    _, _, term, _, info = env.step(0)  # ATTACK enemy 0
    assert info["fighter_attack"]["d20_roll"] == 17
    assert info["fighter_attack"]["damage"] == 9
    assert info["enemy_hps"] == [9, 18] and info["round"] == 1 and not term

    obs, _, _, _, _ = env.step(3)  # ACTION_SURGE
    assert obs[1] == 1 and obs[4] == 0  # ACTION restored, SURGE spent

    _, _, term, _, info = env.step(0)  # ATTACK enemy 0 again (crit)
    assert info["fighter_attack"]["d20_roll"] == 20
    assert info["fighter_attack"]["critical"] is True
    assert info["enemy_hps"] == [0, 18] and not term  # victory needs both deaths

    obs, reward, term, trunc, info = env.step(4)  # END_TURN: dead slot skipped
    assert info["enemy_attacks"][0] is None
    assert info["enemy_attacks"][1]["d20_roll"] == 2
    assert info["enemy_attacks"][1]["hit"] is False
    assert (obs[0], obs[5], obs[17]) == (20, 0, 2)  # fighter HP, enemy0 HP, round
    assert (reward, term, trunc) == (0.0, False, False)
    fighter = env.fighter
    assert fighter is not None
    assert fighter.resources.get(Resource.ACTION) == 1
    assert fighter.resources.get(Resource.BONUS_ACTION) == 1

    _, _, _, _, info = env.step(1)  # ATTACK enemy 1
    assert info["fighter_attack"]["d20_roll"] == 12
    assert info["fighter_attack"]["damage"] == 6
    assert info["enemy_hps"] == [0, 12]

    _, _, _, _, info = env.step(4)  # END_TURN round 2
    assert info["enemy_attacks"][1]["d20_roll"] == 5
    assert info["round"] == 3 and info["fighter_hp"] == 20


def test_m1b_seed17_enemy_slot_order_and_rounds() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=17)
    assert _enemy_stats(env) == ((17, 17, 2, 4, 2), (15, 17, 3, 4, 1))

    _, _, _, _, info = env.step(1)  # ATTACK enemy 1: miss
    assert info["fighter_attack"]["d20_roll"] == 10
    assert info["fighter_attack"]["hit"] is False
    assert info["enemy_hps"] == [17, 15]

    env.step(3)  # ACTION_SURGE
    _, _, _, _, info = env.step(1)  # ATTACK enemy 1 again: miss
    assert info["fighter_attack"]["d20_roll"] == 8
    assert info["enemy_hps"] == [17, 15]

    _, _, _, _, info = env.step(4)  # END_TURN: both enemies hit in slot order
    assert [a["d20_roll"] for a in info["enemy_attacks"]] == [19, 13]
    assert [a["damage"] for a in info["enemy_attacks"]] == [4, 4]
    assert info["fighter_hp"] == 12 and info["round"] == 2

    _, _, _, _, info = env.step(0)  # ATTACK enemy 0: miss
    assert info["fighter_attack"]["d20_roll"] == 2

    _, _, _, _, info = env.step(4)  # END_TURN round 2
    assert [a["d20_roll"] for a in info["enemy_attacks"]] == [15, 1]
    assert info["fighter_hp"] == 9 and info["round"] == 3


def test_m2_seed2_cleave_then_targeted_sequence() -> None:
    env = StagedCombatEnv("m2")
    env.reset(seed=2)
    assert _enemy_stats(env) == ((18, 13, 2, 4, 2), (18, 15, 2, 6, 3))

    fighter = env.fighter
    assert fighter is not None
    assert fighter.resources.get(Resource.CLEAVE) == 1

    _, _, term, _, info = env.step(2)  # CLEAVE both living
    assert info["cleave_living_targets"] == 2
    assert [a["d20_roll"] for a in info["cleave_attacks"]] == [17, 20]
    assert [a["damage"] for a in info["cleave_attacks"]] == [4, 6]
    assert info["enemy_hps"] == [14, 12] and not term
    assert fighter.resources.get(Resource.ACTION) == 0
    assert fighter.resources.get(Resource.CLEAVE) == 0

    env.step(4)  # ACTION_SURGE
    _, _, _, _, info = env.step(0)  # ATTACK enemy 0: miss consumes same RNG slot
    assert info["fighter_attack"]["d20_roll"] == 2
    assert info["enemy_hps"] == [14, 12]

    _, _, _, _, info = env.step(5)  # END_TURN: both miss
    assert [a["d20_roll"] for a in info["enemy_attacks"]] == [12, 6]
    assert info["fighter_hp"] == 20 and info["round"] == 2


def test_m1a_seed17_single_enemy_sequence() -> None:
    env = StagedCombatEnv("m1a")
    env.reset(seed=17)
    assert _enemy_stats(env) == ((17, 17, 2, 4, 2),)

    _, _, _, _, info = env.step(0)  # ATTACK: hit for 10
    assert info["fighter_attack"]["d20_roll"] == 12
    assert info["fighter_attack"]["damage"] == 10
    assert info["enemy_hps"] == [7]

    _, _, _, _, info = env.step(3)  # END_TURN: enemy miss, round 2
    assert info["enemy_attacks"][0]["d20_roll"] == 8
    assert info["goblin_attack"]["d20_roll"] == 8
    assert info["fighter_hp"] == 20 and info["round"] == 2


def test_m0_seed13_two_round_sequence() -> None:
    env = BaldurCombatEnv()
    obs, _ = env.reset(seed=13)
    assert obs.tolist() == [20, 1, 1, 1, 1, 15, 1]

    _, _, _, _, info = env.step(0)  # ATTACK: hit for 10
    assert info["fighter_attack"]["d20_roll"] == 18
    assert info["goblin_hp"] == 5

    _, _, _, _, info = env.step(3)  # END_TURN: goblin hits for 8
    assert info["goblin_attack"]["d20_roll"] == 17
    assert info["goblin_attack"]["damage"] == 8
    assert info["fighter_hp"] == 12 and info["round"] == 2
