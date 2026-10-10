"""Isolation and integration tests for the shallow turn model.

TurnManager and CombatRoster are exercised without Gymnasium. Environment
tests lock round, refresh, termination, and debug-info behavior.
"""

from dataclasses import FrozenInstanceError

import pytest

from combat.actors import ActorRef, CombatRoster, Side
from combat.actions import Action
from combat.characters import Fighter, Goblin, refresh_turn_resources
from combat.damage import DamageSpec
from combat.env import MAX_ROUNDS, BaldurCombatEnv
from combat.resources import Resource
from combat.stages import StagedCombatEnv
from combat.turns import TurnManager


def _fighter(hp: int = 20) -> Fighter:
    return Fighter("Fighter", 20, hp, 16, 5, DamageSpec(1, 8, 3))


def _goblin(hp: int = 15) -> Goblin:
    return Goblin("Goblin", 15, hp, 15, 4, DamageSpec(1, 6, 2))


def _alive_map(*living: ActorRef):
    living_set = frozenset(living)
    return lambda ref: ref in living_set


# ActorRef and CombatRoster.


def test_actor_ref_is_immutable_and_has_no_character_pointer() -> None:
    ref = ActorRef(Side.ALLY, 0)
    assert (ref.side, ref.slot) == (Side.ALLY, 0)
    with pytest.raises(FrozenInstanceError):
        ref.slot = 1  # type: ignore[misc]
    assert not hasattr(ref, "character")
    with pytest.raises(ValueError, match="negative"):
        ActorRef(Side.ENEMY, -1)


def test_roster_resolves_and_tracks_liveness() -> None:
    roster = CombatRoster(allies=(_fighter(),), enemies=(_goblin(hp=0), _goblin()))
    assert roster.get(ActorRef(Side.ALLY, 0)).name == "Fighter"
    assert roster.get(ActorRef(Side.ENEMY, 1)).alive
    assert roster.is_alive(ActorRef(Side.ALLY, 0))
    assert not roster.is_alive(ActorRef(Side.ENEMY, 0))
    assert len(roster.living_allies()) == 1
    assert len(roster.living_enemies()) == 1
    assert not roster.side_defeated(Side.ALLY)
    assert not roster.side_defeated(Side.ENEMY)
    with pytest.raises(IndexError):
        roster.get(ActorRef(Side.ENEMY, 5))


def test_roster_side_defeated_needs_every_member_dead() -> None:
    roster = CombatRoster(allies=(_fighter(hp=0), _fighter()), enemies=(_goblin(hp=0),))
    assert not roster.side_defeated(Side.ALLY)  # one ally survives
    assert roster.side_defeated(Side.ENEMY)
    roster.allies[1].hp = 0
    assert roster.side_defeated(Side.ALLY)


# TurnManager ordering.


def test_one_ally_one_enemy_order_and_round_wrap() -> None:
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0))
    turns = TurnManager(order)
    alive = _alive_map(*order)
    assert turns.current == order[0] and turns.round_number == 1
    assert turns.advance(alive) is False  # ALLY 0 -> ENEMY 0, same round
    assert turns.current == order[1] and turns.round_number == 1
    assert turns.advance(alive) is True  # wrap -> round 2
    assert turns.current == order[0] and turns.round_number == 2


def test_one_ally_two_enemy_order() -> None:
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1))
    turns = TurnManager(order)
    alive = _alive_map(*order)
    assert [turns.current] == [order[0]]
    turns.advance(alive)
    assert turns.current == order[1]
    turns.advance(alive)
    assert turns.current == order[2] and turns.round_number == 1
    assert turns.advance(alive) is True
    assert turns.current == order[0] and turns.round_number == 2


def test_hypothetical_two_allies_two_enemies_order() -> None:
    order = (
        ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1),
        ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1),
    )
    turns = TurnManager(order)
    alive = _alive_map(*order)
    seen = [turns.current]
    for _ in range(4):
        seen.append(turns.peek_next(alive)[0])
        turns.advance(alive)
    assert seen == [order[0], order[1], order[2], order[3], order[0]]
    assert turns.round_number == 2


def test_dead_actors_are_skipped() -> None:
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1))
    turns = TurnManager(order)
    # ENEMY 0 died on ALLY 0's turn; advance must skip it without wrapping.
    alive = _alive_map(order[0], order[2])
    assert turns.advance(alive) is False
    assert turns.current == order[2]
    # Multiple dead: only ALLY 0 remains; next advance wraps to itself.
    solo = _alive_map(order[0])
    assert turns.advance(solo) is True
    assert turns.current == order[0] and turns.round_number == 2


def test_active_actor_death_before_future_turn_is_skipped() -> None:
    order = (
        ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1),
        ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1),
    )
    turns = TurnManager(order)
    turns.advance(_alive_map(*order))  # now ALLY 1
    # ALLY 1 and ENEMY 0 die before their turns complete the round.
    alive = _alive_map(order[0], order[3])
    assert turns.advance(alive) is False
    assert turns.current == order[3]  # ENEMY 1, skipping dead ENEMY 0
    assert turns.advance(alive) is True  # wrap to ALLY 0, round 2


def test_no_infinite_loop_when_a_side_is_fully_dead() -> None:
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1))
    turns = TurnManager(order)
    # All enemies dead: every advance stays on the living ally and wraps.
    for expected_round in range(2, 12):
        assert turns.advance(_alive_map(order[0])) is True
        assert turns.current == order[0] and turns.round_number == expected_round
    # Nobody alive: advance stays put, keeps the round, and never hangs.
    dead = TurnManager(order, round_number=3)
    for _ in range(10):
        assert dead.advance(_alive_map()) is False
        assert dead.current == order[0] and dead.round_number == 3
        assert dead.peek_next(_alive_map()) == (None, False)


def test_turn_manager_is_deterministic_and_uses_no_rng() -> None:
    order = (ActorRef(Side.ALLY, 0), ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1))
    first, second = TurnManager(order), TurnManager(order)
    alive = _alive_map(order[0], order[2])
    for _ in range(20):
        assert first.advance(alive) == second.advance(alive)
        assert first.current == second.current
        assert first.round_number == second.round_number
    # No RNG parameter exists; identical alive sets always give identical paths.
    assert TurnManager(order).peek_next(alive) == TurnManager(order).peek_next(alive)


# Resource refresh.


def test_fighter_refresh_restores_only_per_turn_resources() -> None:
    fighter = _fighter()
    fighter.resources.set(Resource.ACTION, 0)
    fighter.resources.set(Resource.BONUS_ACTION, 0)
    fighter.resources.set(Resource.SECOND_WIND, 0)
    fighter.resources.set(Resource.ACTION_SURGE, 0)
    refresh_turn_resources(fighter)
    assert fighter.resources.get(Resource.ACTION) == 1
    assert fighter.resources.get(Resource.BONUS_ACTION) == 1
    assert fighter.resources.get(Resource.SECOND_WIND) == 0
    assert fighter.resources.get(Resource.ACTION_SURGE) == 0


def test_cleave_does_not_refresh() -> None:
    fighter = _fighter()
    fighter.resources.set(Resource.CLEAVE, 0)
    fighter.known_abilities = fighter.known_abilities | {Action.CLEAVE}
    refresh_turn_resources(fighter)
    assert fighter.resources.get(Resource.CLEAVE) == 0


def test_goblin_refresh_restores_action_only() -> None:
    goblin = _goblin()
    goblin.resources.set(Resource.ACTION, 0)
    refresh_turn_resources(goblin)
    assert goblin.resources.get(Resource.ACTION) == 1


# Environment turn integration.


def test_next_controlled_ally_refreshes_without_round_wrap_or_automatic_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    assert env.fighter is not None
    ally_1 = _fighter()
    ally_1.resources.set(Resource.ACTION, 0)
    ally_1.resources.set(Resource.BONUS_ACTION, 0)
    # Extend only the internal turn flow; no new stage or observation contract.
    env.roster = CombatRoster(allies=(env.fighter, ally_1), enemies=env.enemies)
    order = (
        ActorRef(Side.ALLY, 0), ActorRef(Side.ALLY, 1),
        ActorRef(Side.ENEMY, 0), ActorRef(Side.ENEMY, 1),
    )
    env.turns = TurnManager(order)
    env._controlled_refs = frozenset(order[:2])
    automatic = Mock(side_effect=AssertionError("No automatic turn should run"))
    monkeypatch.setattr(env, "_run_automatic_turn", automatic)
    for enemy in env.enemies:
        enemy.resources.set(Resource.ACTION, 0)

    assert env.fighter.resources.get(Resource.ACTION) == 1
    assert ally_1.resources.get(Resource.ACTION) == 0
    assert ally_1.resources.get(Resource.BONUS_ACTION) == 0
    _, reward, terminated, truncated, info = env.step(4)  # ALLY 0 END_TURN

    assert env.turns.current == order[1]
    assert env.turns.current in env._controlled_refs
    assert env.round_number == env.turns.round_number == 1
    assert ally_1.resources.get(Resource.ACTION) == 1
    assert ally_1.resources.get(Resource.BONUS_ACTION) == 1
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["enemy_attacks"] == [None, None]
    assert all(enemy.resources.get(Resource.ACTION) == 0 for enemy in env.enemies)
    automatic.assert_not_called()


def test_end_turn_refreshes_only_participants_and_skips_dead() -> None:
    env = StagedCombatEnv("m1b")
    env.reset(seed=2)
    assert env.fighter is not None
    env.enemies[0].hp = 0
    env.enemies[0].resources.set(Resource.ACTION, 0)
    env.enemies[1].resources.set(Resource.ACTION, 0)
    env.fighter.resources.set(Resource.ACTION, 0)
    env.fighter.resources.set(Resource.BONUS_ACTION, 0)

    _, _, _, _, info = env.step(4)

    assert info["enemy_attacks"][0] is None
    assert info["enemy_attacks"][1] is not None
    assert env.enemies[0].resources.get(Resource.ACTION) == 0  # dead, not refreshed
    assert env.enemies[1].resources.get(Resource.ACTION) == 0  # refreshed then spent
    assert env.fighter.resources.get(Resource.ACTION) == 1  # new turn refresh
    assert env.fighter.resources.get(Resource.BONUS_ACTION) == 1
    assert info["round"] == 2


def test_abilities_do_not_begin_a_new_turn() -> None:
    m0 = BaldurCombatEnv()
    m0.reset(seed=1)
    assert m0.fighter is not None
    m0.fighter.hp = 10
    m0.fighter.resources.set(Resource.ACTION, 0)
    m0.step(1)  # SECOND_WIND
    assert m0.fighter.resources.get(Resource.ACTION) == 0
    assert m0.fighter.resources.get(Resource.SECOND_WIND) == 0

    m0.fighter.resources.set(Resource.BONUS_ACTION, 0)
    m0.step(2)  # ACTION_SURGE grants an Action, without refreshing Bonus Action
    assert m0.fighter.resources.get(Resource.ACTION) == 1
    assert m0.fighter.resources.get(Resource.BONUS_ACTION) == 0
    assert m0.fighter.resources.get(Resource.ACTION_SURGE) == 0

    m2 = StagedCombatEnv("m2")
    m2.reset(seed=2)
    assert m2.fighter is not None
    m2.fighter.resources.set(Resource.BONUS_ACTION, 0)
    m2.step(2)  # CLEAVE
    assert m2.fighter.resources.get(Resource.ACTION) == 0
    assert m2.fighter.resources.get(Resource.BONUS_ACTION) == 0
    assert m2.fighter.resources.get(Resource.CLEAVE) == 0


@pytest.mark.parametrize("stage", ["m0", "m1a", "m1b", "m2"])
def test_no_new_round_or_refresh_after_fighter_death(stage: str) -> None:
    env = BaldurCombatEnv() if stage == "m0" else StagedCombatEnv(stage)
    env.reset(seed=2)
    assert env.fighter is not None
    env.fighter.hp = 1
    env.fighter.resources.set(Resource.ACTION, 0)
    env.fighter.resources.set(Resource.BONUS_ACTION, 0)
    roster = env._roster()
    for enemy in roster.enemies:
        enemy.resources.set(Resource.ACTION, 0)
    from unittest.mock import Mock

    rng = Mock()
    rng.integers.side_effect = [20, 1, 1]  # first enemy crits for the kill
    env.np_random = rng  # type: ignore[method-assign]

    _, reward, terminated, truncated, info = env.step(env.action_names.index("END_TURN"))

    assert terminated and not truncated and reward == -1
    if len(roster.enemies) == 2:
        assert info["enemy_attacks"][1] is None  # second enemy never acts
    assert all(enemy.resources.get(Resource.ACTION) == 0 for enemy in roster.enemies)
    assert info["round"] == 1  # no wrap, no new round
    assert env.fighter.resources.get(Resource.ACTION) == 0  # no refresh after death
    assert env.fighter.resources.get(Resource.BONUS_ACTION) == 0


@pytest.mark.parametrize("stage", ["m0", "m1a", "m1b", "m2"])
def test_no_refresh_after_fighter_victory(stage: str) -> None:
    from unittest.mock import Mock

    env = BaldurCombatEnv() if stage == "m0" else StagedCombatEnv(stage)
    env.reset(seed=2)
    assert env.fighter is not None
    roster = env._roster()
    roster.enemies[0].hp = 1
    for enemy in roster.enemies[1:]:
        enemy.hp = 0
    env.fighter.resources.set(Resource.BONUS_ACTION, 0)
    rng = Mock()
    rng.integers.side_effect = [20, 1, 1]  # guaranteed victory on this attack
    env.np_random = rng

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated and not truncated and reward == 1
    assert info["round"] == env.round_number == 1
    assert env.fighter.resources.get(Resource.ACTION) == 0
    assert env.fighter.resources.get(Resource.BONUS_ACTION) == 0


@pytest.mark.parametrize("stage", ["m0", "m1a", "m1b", "m2"])
def test_round_fifty_truncation_preserves_round_and_resources(stage: str) -> None:
    env = BaldurCombatEnv() if stage == "m0" else StagedCombatEnv(stage)
    env.reset(seed=0)
    for enemy in env._roster().enemies:
        enemy.damage = DamageSpec(1, 6, -12)  # enemy hits deal zero
    end_turn = env.action_names.index("END_TURN")
    for _ in range(MAX_ROUNDS - 1):
        _, _, terminated, truncated, _ = env.step(end_turn)
        assert not terminated and not truncated
    assert env.fighter is not None
    env.fighter.resources.set(Resource.ACTION, 0)
    env.fighter.resources.set(Resource.BONUS_ACTION, 0)

    _, _, terminated, truncated, info = env.step(end_turn)

    assert not terminated and truncated
    assert info["round"] == MAX_ROUNDS == env.round_number
    assert env.fighter.resources.get(Resource.ACTION) == 0  # no round-51 refresh
    assert env.fighter.resources.get(Resource.BONUS_ACTION) == 0


def test_staged_debug_info_preserves_existing_keys() -> None:
    env = StagedCombatEnv("m1b")
    _, reset_info = env.reset(seed=2)
    for key in ("round", "action", "decision", "target_index", "fighter_hp", "enemy_hps", "goblin_hp"):
        assert key in reset_info
    assert reset_info["active_actor_side"] == "ALLY"
    assert reset_info["active_actor_slot"] == 0
    assert reset_info["turn_order"] == ["ALLY/0", "ENEMY/0", "ENEMY/1"]
    _, _, _, _, step_info = env.step(0)
    assert step_info["active_actor_side"] == "ALLY"  # enemy turns stay hidden
    assert "fighter_attack" in step_info


def test_m0_info_contract_unchanged() -> None:
    env = BaldurCombatEnv()
    _, reset_info = env.reset(seed=7)
    assert reset_info == {"round": 1, "action": None, "fighter_hp": 20, "goblin_hp": 15}


def test_observation_and_action_contracts_unchanged() -> None:
    m0 = BaldurCombatEnv()
    m0.reset(seed=0)
    assert m0.observation_fields == (
        "fighter_hp", "fighter_action_count", "fighter_bonus_action_count",
        "fighter_second_wind_count", "fighter_action_surge_count", "goblin_hp", "round_number",
    )
    assert m0.action_names == ("ATTACK", "SECOND_WIND", "ACTION_SURGE", "END_TURN")
    assert m0.action_space.n == 4 and m0.observation_space.shape == (7,)

    expectations = {
        "m1a": (12, ("ATTACK", "SECOND_WIND", "ACTION_SURGE", "END_TURN")),
        "m1b": (18, ("ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "SECOND_WIND", "ACTION_SURGE", "END_TURN")),
        "m2": (19, ("ATTACK_ENEMY_0", "ATTACK_ENEMY_1", "CLEAVE", "SECOND_WIND", "ACTION_SURGE", "END_TURN")),
    }
    for stage, (fields, actions) in expectations.items():
        env = StagedCombatEnv(stage)
        obs, _ = env.reset(seed=17)
        assert len(env.observation_fields) == fields == len(obs)
        assert env.action_names == actions
        assert env.observation_space.shape == (fields,)
        assert env.action_space.n == len(actions)
