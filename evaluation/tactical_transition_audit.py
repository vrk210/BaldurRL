"""Audit the table model against the replay oracle on visited and constructed edge states.

Every decision of random and heuristic play is checked, and every third decision is
also mutated into edge cases: lethal Advance, spent Reactions, Prone with Dodge,
Healer HP caps and spent charges, dead actors, and the round limit. Slow: the
default 30 seeds per stage took about 50 minutes on one core.
"""
import json
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np

from agents.heuristic_agent import make_heuristic_agent
from agents.random_agent import RandomAgent
from combat.env import MAX_ROUNDS
from combat.stages import make_env
from combat.table_tactical_transitions import TableTacticalTransitionModel
from combat.tactical_transitions import TacticalTransitionModel
from combat.transitions import Status

BEST_HEURISTIC = {'m5': 'tuned', 'm6': 'adaptive'}


def compare(stage: str, obs: np.ndarray, tag: str, counts: Counter, failures: list) -> None:
    """Check legal actions, restored observations and every successor distribution."""
    try:
        oracle = TacticalTransitionModel.from_observation(stage, obs)
        table = TableTacticalTransitionModel.from_observation(stage, obs)
        state = oracle.state_from_observation(obs)
    except ValueError:  # the mutation is not a valid state of this encounter
        counts['invalid_' + tag] += 1
        return
    problems = []
    if table.state_from_observation(obs) != state:
        problems.append('state')
    elif not np.array_equal(table.observation(state), oracle.observation(state)):
        problems.append('observation')
    elif oracle.status(state) is Status.ONGOING and table.legal_actions(state) != oracle.legal_actions(state):
        problems.append('legal actions')
    if problems or oracle.status(state) is not Status.ONGOING:
        if problems:
            failures.append(dict(stage=stage, tag=tag, problem=problems[0], obs=obs.tolist()))
        return
    for action in oracle.legal_actions(state):
        counts[tag] += 1
        expected = {s: p for p, s in oracle.outcomes(state, action)}
        actual = {s: p for p, s in table.outcomes(state, action)}
        problem = None
        if actual.keys() != expected.keys():
            problem = 'successor set'
        elif max(abs(actual[s] - expected[s]) for s in actual) > 1e-12 or abs(sum(actual.values()) - 1) > 1e-12:
            problem = 'probability'
        elif any(not np.array_equal(table.observation(s), oracle.observation(s)) for s in actual):
            problem = 'successor observation'
        if problem:
            failures.append(dict(stage=stage, tag=tag, action=action, problem=problem, obs=obs.tolist()))
        for successor in actual:
            counts['reach_' + Status(successor[0]).name] += 1


def mutations(fields: list[str], obs: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Edge states derived from a visited allied decision; fields absent on a stage are skipped."""
    ally = int(obs[0])
    other = 1 - ally
    enemies = sum(1 for name in fields if name.startswith('enemy_') and name.endswith('_max_hp'))

    def mutate(tag: str, values: dict) -> Iterator[tuple[str, np.ndarray]]:
        if all(name in fields for name in values):
            changed = obs.copy()
            for name, value in values.items():
                changed[fields.index(name)] = value
            yield tag, changed

    yield from mutate('lethal_hp1', {f'ally_{ally}_hp': 1})
    yield from mutate('lethal_hp3', {f'ally_{ally}_hp': 3})
    yield from mutate('lethal_hp1_disengaged', {f'ally_{ally}_hp': 1, f'ally_{ally}_disengaged': 1})
    yield from mutate('reactions_spent', {f'enemy_{i}_reaction_count': 0 for i in range(enemies)})
    yield from mutate('prone_dodge', {f'enemy_{i}_prone': 1 for i in range(enemies)} | {f'ally_{other}_dodging': 1})
    no_heals = {f'enemy_{j}_heal_count': 0 for j in range(enemies)}
    for i in range(enemies):
        max_hp = obs[fields.index(f'enemy_{i}_max_hp')]
        yield from mutate(f'heal_cap_e{i}', {f'enemy_{i}_hp': max(1, max_hp - 1)})
        yield from mutate(f'heal_low_e{i}', {f'enemy_{i}_hp': 1})
        yield from mutate(f'heal_spent_low_e{i}', {f'enemy_{i}_hp': 1} | no_heals)
        yield from mutate(f'dead_e{i}', {f'enemy_{i}_hp': 0})
    yield from mutate('other_ally_dead', {f'ally_{other}_hp': 0})
    yield from mutate('round_limit', {'round_number': MAX_ROUNDS})
    yield from mutate('round_limit_lethal', {'round_number': MAX_ROUNDS, f'ally_{ally}_hp': 1})


def audit(seeds: range, mutate_every: int = 3) -> dict:
    counts, failures = Counter(), []
    tick = perf_counter()
    for stage in ('m5', 'm6'):
        env = make_env(stage)
        fields = list(env.observation_fields)
        for seed in seeds:
            for kind in ('random', 'heuristic'):
                obs, _ = env.reset(seed=seed)
                policy = RandomAgent(seed) if kind == 'random' else make_heuristic_agent(stage, BEST_HEURISTIC[stage])
                step = 0
                while True:
                    compare(stage, obs, f'{kind}_visited', counts, failures)
                    if step % mutate_every == 0:
                        for tag, changed in mutations(fields, obs):
                            compare(stage, changed, tag, counts, failures)
                    obs, _, terminated, truncated, _ = env.step(policy.choose_action(obs, env.action_masks()))
                    step += 1
                    if terminated or truncated:
                        break
        env.close()
    cases = sum(v for k, v in counts.items() if not k.startswith(('reach_', 'invalid_')))
    return dict(seconds=perf_counter() - tick, cases=cases, failures=len(failures),
                by_tag=dict(sorted(counts.items())), first_failures=failures[:5])


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed-start', type=int, default=211000)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--mutate-every', type=int, default=3)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.mutate_every < 1:
        parser.error('episodes and mutate-every must be positive')
    result = audit(range(args.seed_start, args.seed_start + args.episodes), args.mutate_every)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(vars(args) | result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('cases', 'failures', 'seconds')}))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
