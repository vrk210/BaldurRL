"""Compare every successor with the replay oracle and time both implementations."""
import json
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np

from agents.random_agent import RandomAgent
from combat.stages import make_env
from combat.table_tactical_transitions import TableTacticalTransitionModel
from combat.tactical_transitions import TacticalTransitionModel
from evaluation.seeds import require_not_frozen


def benchmark(seeds: range, stride: int = 5) -> dict:
    rows = []
    for stage in ('m5', 'm6'):
        env = make_env(stage)
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy = RandomAgent(seed)
            oracle = TacticalTransitionModel.from_observation(stage, obs)
            table = TableTacticalTransitionModel.from_observation(stage, obs)
            step = 0
            while True:
                if step % stride == 0:
                    state = oracle.state_from_observation(obs)
                    for action in oracle.legal_actions(state):
                        tick = perf_counter()
                        expected = {s: p for p, s in oracle.outcomes(state, action)}
                        oracle_time = perf_counter() - tick
                        tick = perf_counter()
                        actual = {s: p for p, s in table.outcomes(state, action)}
                        table_time = perf_counter() - tick
                        if actual.keys() != expected.keys():
                            raise AssertionError((stage, seed, step, action, 'successor mismatch'))
                        error = max(abs(actual[s] - expected[s]) for s in actual)
                        if error > 1e-12 or abs(sum(actual.values()) - 1) > 1e-12:
                            raise AssertionError((stage, seed, step, action, error))
                        rows.append(dict(stage=stage, seed=seed, step=step, action=env.action_names[action],
                                         successors=len(actual), max_probability_error=error,
                                         oracle_seconds=oracle_time, table_seconds=table_time))
                obs, _, terminated, truncated, _ = env.step(policy.choose_action(obs, env.action_masks()))
                step += 1
                if terminated or truncated:
                    break
        env.close()
    summary = {}
    for stage in ('m5', 'm6'):
        selected = [r for r in rows if r['stage'] == stage]
        oracle_time = sum(r['oracle_seconds'] for r in selected)
        table_time = sum(r['table_seconds'] for r in selected)
        summary[stage] = dict(cases=len(selected), max_probability_error=max(r['max_probability_error'] for r in selected),
                              oracle_seconds=oracle_time, table_seconds=table_time,
                              speedup=oracle_time / table_time)
    return dict(summary=summary, cases=rows)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed-start', type=int, default=210000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.stride < 1:
        parser.error('episodes and stride must be positive')
    require_not_frozen(range(args.seed_start, args.seed_start + args.episodes))
    result = benchmark(range(args.seed_start, args.seed_start + args.episodes), args.stride)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(vars(args) | result, indent=2) + '\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
