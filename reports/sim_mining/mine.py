"""Reproduce the tactics playbook; historical inputs are opened read-only.

Run from repo root with PYTHONPATH=. python reports/sim_mining/mine.py COMMAND.
Compact outputs go only to this worktree's ignored runs/sim_mining directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from math import exp, lgamma, log, sqrt
from pathlib import Path
from statistics import NormalDist
import time
from typing import Any

import numpy as np

from agents.tactical_heuristics import TacticalHeuristicAgent, VARIANTS
from combat.actions import Action
from combat.simulation import simulation_from_observation
from combat.stages import make_env
from combat.tactical_env import tactical_observation_fields

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT.parent / 'decision-stages' / 'runs'
OUT = ROOT / 'runs' / 'sim_mining'
BASE = {'m5': 'tuned', 'm6': 'adaptive'}
CHANGES = {
    'm5': ['split_focus', 'healer_first', 'surge_round2', 'sw10', 'cleave2',
           'trip_threat', 'dodge', 'no_surge'],
    'm6': ['one_diver', 'disengage', 'dodge', 'hold', 'dive'],
}
FAMILY = sum(map(len, CHANGES.values()))


def write(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    # No accidental overwriting of a once-only confirmation/replay.
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def exact_mcnemar(plus: int, minus: int) -> float:
    n=plus+minus
    if n==0:
        return 1.0
    k=min(plus,minus)
    tail=sum(exp(lgamma(n+1)-lgamma(j+1)-lgamma(n-j+1)-n*log(2)) for j in range(k+1))
    return min(1.0,2*tail)


def wilson(k: int, n: int) -> list[float]:
    z=NormalDist().inv_cdf(.975)
    p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def contrast(a: list[float], b: list[float], family: int = 1) -> dict[str, Any]:
    x, y = np.array(a), np.array(b)
    if x.shape != y.shape or not len(x):
        raise ValueError('Need matching nonempty paired arrays')
    d = x-y
    se = float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else 0.0
    z = NormalDist().inv_cdf(1-.05/(2*family))
    nplus, nminus = int(np.sum(d>0)), int(np.sum(d<0))
    binary = bool(np.isin(x, [0,1]).all() and np.isin(y, [0,1]).all())
    p = exact_mcnemar(nplus,nminus) if binary else None
    return {'n':len(d), 'first_rate':float(x.mean()), 'second_rate':float(y.mean()),
            'difference':float(d.mean()), 'paired_se':se,
            'ci95':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
            'simultaneous_ci':[float(d.mean()-z*se),float(d.mean()+z*se)],
            'first_only':nplus if binary else None, 'second_only':nminus if binary else None,
            'mcnemar_p':p, 'bonferroni_p':min(1,p*family) if p is not None else None}


class MiningPolicy(TacticalHeuristicAgent):
    """Named, research-only single-rule ablations; no existing variants mutate."""
    def __init__(self, stage: str, name: str = 'baseline') -> None:
        config = VARIANTS[stage][BASE[stage]]
        kwargs = {'healer_first': {'priority':'role'}, 'sw10':{'second_wind_threshold':10},
                  'cleave2':{'cleave_min_targets':2}, 'trip_threat':{'trip_target':'threat'},
                  'dodge':{'dodge':True}, 'one_diver':{'advance':'adaptive'},
                  'disengage':{'disengage':True}, 'hold':{'advance':'never'},
                  'dive':{'advance':'all'}}
        super().__init__(stage, replace(config, **kwargs.get(name, {})))
        self.name = name
        self._split = False

    def _priority_key(self, enemy: Any) -> tuple[float,...]:
        return (-enemy.hp, enemy.slot) if self._split else super()._priority_key(enemy)

    def _movement(self, mask: np.ndarray, me: Any, allies: list[Any], enemies: list[Any]) -> int | None:
        if self.name == 'one_diver' and any(a.alive and a.deep for a in allies):
            return None
        return super()._movement(mask, me, allies, enemies)

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        self._split = self.name == 'split_focus' and int(observation[0]) == 1
        mask = np.array(action_mask, copy=True)
        if self.name == 'no_surge' or (self.name == 'surge_round2' and observation[-1] < 2):
            mask[self._decision(Action.ACTION_SURGE)] = False
        return super().choose_action(observation, mask)


def fight(env: Any, policy: MiningPolicy, obs: np.ndarray) -> int:
    for _ in range(1000):
        action = policy.choose_action(obs, env.action_masks())
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            return int(reward > 0)
    raise RuntimeError('Unexpected decision limit')


def confirmation() -> None:
    start, cpu = time.perf_counter(), time.process_time()
    outputs = {}
    for stage, changes in CHANGES.items():
        env = make_env(stage)
        results = {}
        for name in ['baseline']+changes:
            policy = MiningPolicy(stage,name)
            outcomes = []
            for seed in range(232000,236000):
                obs, _ = env.reset(seed=seed)
                outcomes.append(fight(env, policy, obs))
            results[name] = outcomes
            print(stage, name, sum(outcomes), flush=True)
        contrasts = {name:contrast(results['baseline'],results[name],FAMILY) for name in changes}
        outputs[stage] = {'outcomes':results, 'baseline_minus_variant':contrasts}
    outputs['metadata'] = {'seed_start':232000,'seed_end':235999,'family':FAMILY,
        'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,
        'plan_commit':'e2e25d3','source_commit': 'b082262'}
    write('confirmation.json',outputs)


def opening_features(stage: str, obs: list[float]) -> dict[str, float]:
    f = dict(zip(tactical_observation_fields(stage),obs))
    brute = next(i for i in range(3) if f[f'enemy_{i}_is_brute'])
    archer = next(i for i in range(3) if f[f'enemy_{i}_is_archer'])
    durability = f[f'enemy_{brute}_hp']/((26-f[f'enemy_{brute}_ac'])/20)
    return {'durable_brute':float(durability>=42),
            'soft_brute':float(durability<42),
            'low_hp_archer':float(f[f'enemy_{archer}_hp']<=14),
            'weakest_ge2':float(sum(f[f'enemy_{i}_targets_weakest'] for i in range(3))>=2),
            'total_hp_ge60':float(sum(f[f'enemy_{i}_hp'] for i in range(3))>=60),
            'total_hp':sum(f[f'enemy_{i}_hp'] for i in range(3)),
            'brute_attack':f[f'enemy_{brute}_attack_bonus'],
            'brute_durability':durability}


def episode_metrics(stage: str, episode: dict[str,Any]) -> dict[str,Any]:
    fields = {name:i for i,name in enumerate(tactical_observation_fields(stage))}
    counts: Counter[str] = Counter()
    first = episode['steps'][0]['observation']
    for step in episode['steps']:
        o, mask, info = step['observation'],step['action_mask'],step['info']
        action = step['semantic_action']
        counts['decisions'] += 1
        if action == 'END_TURN':
            counts['ends'] += 1
            for key, available in {'attack':any(mask[:3]),'surge':mask[9],
                                   'wind':mask[8], 'trip':any(mask[3:6])}.items():
                counts['end_legal_'+key] += available
        attacks = []
        if 'fighter_attack' in info:
            attacks = [(step['target_index'], info['fighter_attack'])]
        elif 'cleave_attacks' in info:
            # Cleave targets living reachable enemies at the pre-action state.
            me = int(o[0])
            deep = stage=='m6' and bool(o[fields[f'ally_{me}_deep']])
            front_alive = stage=='m6' and any(o[fields[f'enemy_{i}_hp']]>0 and not o[fields[f'enemy_{i}_back_rank']] for i in range(3))
            targets = [i for i in range(3) if o[fields[f'enemy_{i}_hp']]>0 and
                       (stage=='m5' or deep or not front_alive or not o[fields[f'enemy_{i}_back_rank']])]
            assert len(targets)==len(info['cleave_attacks'])
            attacks=list(zip(targets,info['cleave_attacks']))
        for target, attack in attacks:
            damage=int(attack['damage'])
            counts['rolled_damage'] += damage
            counts['overkill'] += max(0,damage-int(o[fields[f'enemy_{target}_hp']]))
        counts[action] += 1
    return {'seed':episode['result']['seed'],'won':int(episode['result']['won']),
        'features':opening_features(stage,first),'counts':dict(counts),
        'first_kill':next(iter(episode['result'].get('kill_order_roles') or []),None)}


def traces() -> None:
    start,cpu = time.perf_counter(),time.process_time()
    all_outputs, inventory = {}, []
    for stage in ('m5','m6'):
        outputs = {}
        for directory in sorted((HIST/'v2'/stage).iterdir()):
            path=directory/'episodes.jsonl'
            if not path.exists():
                continue
            digest=hashlib.sha256()
            episodes=[]
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line)
                    episodes.append(episode_metrics(stage,json.loads(line)))
            assert len(episodes)==5000
            assert [e['seed'] for e in episodes]==list(range(3000,8000))
            totals=Counter()
            for e in episodes:
                totals.update(e['counts'])
            kills=Counter(e['first_kill'] for e in episodes)
            outputs[directory.name]={'episodes':episodes,'totals':dict(totals),'first_kills':dict(kills)}
            inventory.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'episodes':len(episodes)})
            print(stage,directory.name,'read',flush=True)
        baseline=outputs[BASE[stage]]['episodes']
        bw=[e['won'] for e in baseline]
        paired={}
        features=list(baseline[0]['features'])[:5]
        for name,data in outputs.items():
            paired[name]=contrast([e['won'] for e in data['episodes']],bw)
        pooled=np.mean([[e['won'] for e in outputs[f'ppo_s{i}']['episodes']] for i in range(3)],axis=0)
        strata={}
        for feature in features:
            for value in (0,1):
                idx=[i for i,e in enumerate(baseline) if e['features'][feature]==value]
                if idx:
                    strata[f'{feature}={value}']={
                        'ppo_minus_baseline':contrast(pooled[idx].tolist(),[bw[i] for i in idx]),
                        'baseline_loss_rate':float(1-np.mean([bw[i] for i in idx]))}
        all_fail=sum(not any(data['episodes'][i]['won'] for data in outputs.values()) for i in range(5000))
        all_outputs[stage]={'runs':outputs,'paired_minus_baseline':paired,'pooled_strata':strata,
                            'all_tested_policies_lost':all_fail,
                            'pooled_ppo_minus_baseline':contrast(pooled.tolist(),bw)}
    local={}
    for stage in ('m5','m6'):
        local[stage]=json.loads((HIST/'v2'/'analysis'/'stratified'/stage/'summary.json').read_text())
    m2path=ROOT.parent.parent/'BaldurRL'/'runs'/'m2_exact_optimum_9000_13999'/'optimal_and_heuristic_by_seed.json'
    m2=json.loads(m2path.read_text())
    m2_summary={key:float(np.mean([r[key] for r in m2])) for key in ('optimal','heuristic')}
    # Compact observations/metrics, never copy giant source traces.
    write('trace_analysis.json',{'stages':all_outputs,'inventory':inventory,'historical_local':local,
        'm2_exact_expected':m2_summary,'cpu_seconds':time.process_time()-cpu,
        'wall_seconds':time.perf_counter()-start})


def replay() -> None:
    start,cpu=time.perf_counter(),time.process_time()
    results={}
    for stage in ('m5','m6'):
        path=HIST/'v2'/stage/BASE[stage]/'episodes.jsonl'
        starts={0:[],1:[]}
        with path.open() as handle:
            for line in handle:
                episode=json.loads(line)
                starts[int(episode['result']['won'])].append((episode['result']['seed'],episode['steps'][0]['observation']))
        selected=[]
        for won, rows in starts.items():
            for idx in np.linspace(0,len(rows)-1,32,dtype=int):
                seed,obs=rows[idx]
                selected.append((won,seed,obs))
        policy=MiningPolicy(stage)
        records=[]
        for original_won,seed,obs in selected:
            wins=[]
            for rollout_seed in range(236000,236128):
                env=simulation_from_observation(stage,np.array(obs,dtype=np.float32),seed=rollout_seed)
                wins.append(fight(env,policy,np.array(obs,dtype=np.float32)))
            k=sum(wins)
            ci=wilson(k,len(wins))
            records.append({'old_encounter_seed':seed,'old_won':original_won,
                            'features':opening_features(stage,obs),'wins':k,'n':len(wins),
                            'binomial_ci95':ci})
        results[stage]=records
        print(stage,'replay complete',flush=True)
    results['metadata']={'rng_seed_start':236000,'rng_seed_end':236127,
        'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    write('replay.json',results)


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['confirmation','traces','replay'])
    args=parser.parse_args()
    {'confirmation':confirmation,'traces':traces,'replay':replay}[args.command]()

if __name__=='__main__':
    main()
