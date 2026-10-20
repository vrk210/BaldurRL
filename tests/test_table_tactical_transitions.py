"""Independent replay-oracle checks for tactical table dynamics and edge cases."""

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from combat.resources import Resource
from combat.stages import make_env
from combat.tactical_transitions import TacticalTransitionModel
from combat.table_tactical_transitions import TableTacticalTransitionModel


def compare(stage, obs):
    oracle = TacticalTransitionModel.from_observation(stage, obs)
    table = TableTacticalTransitionModel.from_observation(stage, obs)
    state = oracle.state_from_observation(obs)
    assert state == table.state_from_observation(obs)
    assert table.legal_actions(state) == oracle.legal_actions(state)
    for action in oracle.legal_actions(state):
        expected = {s: p for p, s in oracle.outcomes(state, action)}
        actual = {s: p for p, s in table.outcomes(state, action)}
        assert actual.keys() == expected.keys()
        assert sum(actual.values()) == pytest.approx(1, abs=1e-12)
        assert actual == pytest.approx(expected, abs=1e-12)
        states = list(actual)
        np.testing.assert_array_equal(table.observations(states), oracle.observations(states))


@pytest.mark.parametrize('stage', ['m5', 'm6'])
@pytest.mark.parametrize('field,value', [
    ('ally_0_hp', 2.5),
    ('ally_0_hp', -1),
    ('ally_0_hp', 21),
    ('ally_0_hp', np.nan),
    ('ally_0_hp', np.inf),
    ('ally_0_action_count', 3),
    ('ally_0_bonus_action_count', -1),
    ('ally_0_dodging', 2),
    ('enemy_0_last_attacker', 3),
    ('active_actor_index', 2),
    ('round_number', 0),
])
def test_reused_model_rejects_invalid_dynamic_fields(stage, field, value):
    env = make_env(stage)
    obs, _ = env.reset(seed=210005)
    table = TableTacticalTransitionModel.from_observation(stage, obs)
    table.state_from_observation(obs)  # Exercise decoding after valid construction.
    changed = obs.copy()
    changed[list(env.observation_fields).index(field)] = value
    with pytest.raises(ValueError, match=f'Invalid {stage} decision observation'):
        table.state_from_observation(changed)


@pytest.mark.parametrize('stage', ['m5', 'm6'])
@pytest.mark.parametrize('case', [
    'enemy_hp_above_max', 'nonhealer_heal_charge',
    'dead_active_ally', 'defeated_allies', 'defeated_enemies',
])
def test_reused_model_preserves_live_decision_validation(stage, case):
    env = make_env(stage)
    obs, _ = env.reset(seed=210005)
    oracle = TacticalTransitionModel.from_observation(stage, obs)
    table = TableTacticalTransitionModel.from_observation(stage, obs)
    assert table.state_from_observation(obs) == oracle.state_from_observation(obs)
    fields = list(env.observation_fields)
    changed = obs.copy()
    if case == 'enemy_hp_above_max':
        changed[fields.index('enemy_0_hp')] = env.enemies[0].max_hp + 1
    elif case == 'nonhealer_heal_charge':
        slot = next(i for i, enemy in enumerate(env.enemies) if enemy.role.name != 'HEALER')
        changed[fields.index(f'enemy_{slot}_heal_count')] = 1
    else:
        hp_fields = {
            'dead_active_ally': ['ally_0_hp'],
            'defeated_allies': ['ally_0_hp', 'ally_1_hp'],
            'defeated_enemies': [f'enemy_{i}_hp' for i in range(3)],
        }[case]
        for field in hp_fields:
            changed[fields.index(field)] = 0
    for model in (oracle, table):
        with pytest.raises(ValueError, match='Invalid|ongoing living ally'):
            model.state_from_observation(changed)


@pytest.mark.parametrize('stage', ['m5', 'm6'])
def test_random_play_states_against_independent_oracle(stage):
    env = make_env(stage)
    policy = RandomAgent(210000)
    for seed in range(210000, 210003):
        obs, _ = env.reset(seed=seed)
        for step in range(24):
            if step % 5 == 0:
                compare(stage, obs)
            obs, _, terminated, truncated, _ = env.step(policy.choose_action(obs, env.action_masks()))
            if terminated or truncated:
                break


@pytest.mark.parametrize('stage', ['m5', 'm6'])
def test_healing_deaths_retaliation_and_last_round(stage):
    env = make_env(stage)
    env.reset(seed=210005)
    env.round_number = 50
    env.turns._round_number = 50
    env.allies[0].hp = 0
    env.allies[1].hp = 3
    env.allies[1].dodging = True
    env.turns._current_index = 1
    for enemy in env.enemies:
        enemy.hp = min(2, enemy.max_hp)
        enemy.prone = True
        enemy.last_attacker = 0  # dead retaliation target falls back
    compare(stage, env._observation())


def test_advance_death_and_disengagement():
    env = make_env('m6')
    env.reset(seed=210007)
    env.allies[0].hp = 1
    env.allies[0].dodging = True
    compare('m6', env._observation())
    env.allies[1].hp = 0  # advancing actor can now lose the encounter
    compare('m6', env._observation())
    env.allies[1].hp = 20
    env.allies[0].disengaged = True
    compare('m6', env._observation())
    for enemy in env.enemies:
        enemy.resources.set(Resource.REACTION, 0)
    env.allies[0].disengaged = False
    compare('m6', env._observation())


def test_cleave_keeps_original_reach_and_terminal_canonicalization():
    env = make_env('m6')
    env.reset(seed=210011)
    brute = next(e for e in env.enemies if e.rank.name == 'FRONT')
    brute.hp = 1
    obs = env._observation()
    table = TableTacticalTransitionModel.from_observation('m6', obs)
    state = table.state_from_observation(obs)
    for p, successor in table.outcomes(state, 6):
        restored = table.load(successor)
        assert all(e.hp == e.max_hp for e in restored.enemies if e.rank.name == 'BACK')
    compare('m6', obs)
    brute.hp = 0
    env.allies[0].deep = True
    for enemy in env.enemies:
        if enemy.alive:
            enemy.hp = 1
    compare('m6', env._observation())



def test_planner_reusing_its_model_rejects_fractional_hp():
    """A planner reusing its model for a later observation of the encounter still rejects fractional HP."""
    from agents.expectimax_agent import ExpectimaxAgent

    env = make_env('m5')
    obs, _ = env.reset(seed=210005)
    mask = env.action_masks()
    value = lambda observations: np.full(len(observations), 0.5)  # noqa: E731
    agent = ExpectimaxAgent('m5', depth=1, leaf_value=value, afterstate_value=value, decision_horizon=1)
    agent.choose_action(obs, mask)
    later = obs.copy()
    later[list(env.observation_fields).index(f'ally_{int(obs[0])}_hp')] = 2.5
    with pytest.raises(ValueError, match='Invalid m5 decision observation'):
        agent.choose_action(later, mask)
