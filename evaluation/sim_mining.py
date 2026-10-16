"""Mine simulator tactics without treating simulator findings as BG3 facts.

Inventory/provenance (2026-10-15)
-------------------------------
Read-only primary inputs: 22 v2 M5/M6 episodes.jsonl files under sibling
``decision-stages/runs/v2/{m5,m6}/<policy>/``: 110,000 fights, 5,175,127,173
bytes, exactly seeds 3000–7999 in order per file. All heuristic variants,
random, three final PPO policies and three validation-best checkpoints are
kept separate; pooled PPO findings use final policies only. Exact input paths,
byte sizes and SHA-256 digests are in ``evaluation/sim_mining_summary.json``.
Full and partial role kill-order counts, every policy's legal actions at
END_TURN, encounter-clustered ratio CIs, conditional expected overkill,
per-model unique wins and losses, and all planned opening strata are retained.

Earlier corrected local counterfactual summaries are read from sibling
``decision-stages/runs/v2/analysis/stratified/{m5,m6}/summary.json``. Trip→Attack
has 36 sampled PPO disagreement states per stage, 1,024 common-RNG rollouts
per candidate, with heuristic continuation. Targets may differ: this is not
an isolated same-target timing experiment. Their B intervals are historical,
pointwise and unadjusted, not whole-fight PPO repair effects. M2 exact expected
values come from original ``BaldurRL/runs/m2_exact_optimum_9000_13999/``
``optimal_and_heuristic_by_seed.json``; optimal expected loss is 28.21% on that
fixed set, not a sampled M5/M6 loss bound. M4 and guided-planning reports were
context, not pooled evidence. All v1 traces were excluded because rules differ.
Historical integration M5 value files can have nonfinite search values; no
value dataset was used or altered. Other worktrees' runs were read-only.

Damage fields are HP-capped; realized uncapped dice are absent. Expected
excess damage is reconstructed conditional on the logged hit/critical via
shared damage-dice functions, and is C/model-derived, not realized waste or a
causal benefit of changing targets. Cleave events are slot-aligned and include
None for dead/unreachable targets. A provisional capped-damage analysis was
preserved as ignored ``trace_analysis_capped_v0.json`` but is not used.

Methods, audit trail and limitations
-----------------------------------
Predeclared plan was committed before any confirmation as 743ba90 in an older
prompt's requested PROGRESS.md, with single-diver clarification e2e25d3 and
code constant CHANGES committed in a5fb02b/e2e25d3. That standalone plan was
removed when the external prompt's documentation preference changed during
execution. CONFIRMATION_PLAN below preserves its text; relocation did not
precede the old experiments. Once-only confirmation ran on 232000–235999;
13 planned comparisons, same-seed encounters, normal paired CIs with
Bonferroni simultaneous coverage, plus two-sided exact McNemar adjusted p.
Grade A requires both an interval excluding zero AND adjusted p<.05.
Target/resource ablations are M5; positioning is M6; Dodge is both.
Trip-threat's approximate simultaneous CI barely excludes zero but its exact
adjusted p=.0531: deliberately no confirmed directional tactic. Cleave2 is
also inconclusive. Zero normal intervals describe the sample, not population
equivalence. Seed pairing fixes encounters, not future dice after divergence.
All C strata are exploratory observational hypotheses, never causal claims.

Dice replay selects 32 old wins and 32 old losses per stage by evenly spaced
sorted indices and replays each with baseline continuation on the same 128
combat RNG seeds 236000–236127. All old-loss states sometimes win. Per-state
Wilson binomial CIs are retained; shared roll seeds correlate encounters.
Replay does not bound optimal policy value or decompose losses into dice
versus decisions. The all-tested-policies-loss count is not an optimal bound.
No established PPO-favorable opening cohort or local decision type was found;
unique PPO wins can reflect changed dice paths rather than superior tactics.

Reproduce from repository root with PYTHONPATH=. and the shared Python:
``python -m evaluation.sim_mining traces``;
``python -m evaluation.sim_mining confirmation``;
``python -m evaluation.sim_mining replay``;
``python -m evaluation.sim_mining killorders``;
``python -m evaluation.sim_mining_render``.
New raw outputs use this worktree's ignored runs/sim_mining and refuse to
replace existing files. For independent reruns use a fresh worktree/output
location (--historical-root, --legacy-root and --output-root are configurable);
never rerun confirmation to tune rules. A fresh checkout needs no
raw data to reproduce the existing GAME_KNOWLEDGE section:
``python -m evaluation.sim_mining_render --from-summary``.
The compact committed JSON contains final raw-output digests and all evidence.
``pytest -q tests/test_sim_mining.py`` exercises wrappers/statistics; run the
full suite serially. No mechanics, masks, observations or existing variants
were changed. Successful final analysis commands used 303.05 CPU-seconds; the provisional
trace scan added about 40 seconds. No network training. Actual encounter seed accounting is
in evaluation/seeds.py; 236000–236127 were replay combat RNG seeds only.
130000–132999 was never used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from functools import lru_cache
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
from combat.damage import DamageSpec
from combat.distributions import _dice_sum_distribution
from combat.mechanics import damage_dice_count, damage_from_dice, scale_damage
from combat.simulation import simulation_from_observation
from combat.stages import make_env
from combat.tactical_env import tactical_observation_fields

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parent / 'decision-stages' / 'runs'
LEGACY = ROOT.parent.parent / 'BaldurRL'
if (ROOT / 'runs' / 'm2_exact_optimum_9000_13999').exists():
    LEGACY = ROOT
OUT = ROOT / 'runs' / 'sim_mining'
BASE = {'m5': 'tuned', 'm6': 'adaptive'}
CHANGES = {
    'm5': ['split_focus', 'healer_first', 'surge_round2', 'sw10', 'cleave2',
           'trip_threat', 'dodge', 'no_surge'],
    'm6': ['one_diver', 'disengage', 'dodge', 'hold', 'dive'],
}
FAMILY = sum(map(len, CHANGES.values()))

# Historical pre-run declaration, relocated after the prompt changed.
CONFIRMATION_PLAN = """# Simulator tactics mining — predeclared plan (2026-10-15)

Base b082262. Existing runs are read-only. No combat rules or existing policy
variants change; named research variants live in the one-off mining script.

## Confirmation protocol (declared before any new experiment)

Run each fixed policy once on every encounter 232000–235999, identically paired
by seed within stage. Baselines are M5 tuned and M6 adaptive. Two-sided exact
McNemar tests, paired normal 95% CIs, and Bonferroni simultaneous 95% CIs across
the following 13 comparisons; only exclusions of zero in the simultaneous CI
and adjusted p<.05 merit a new grade-A claim. No reruns, threshold tuning or
optional stopping. Counts and full per-seed results are retained.

M5 single-rule changes: (1) highest-HP target for ally 1, lowest for ally 0
(split focus); (2) Healer-first targeting with tuned resources; (3) Action Surge
held until round 2; (4) Second Wind threshold 10 rather than 12; (5) Cleave at
2 rather than 3 living targets; (6) Trip greatest estimated threat rather than
focus; (7) existing incoming-damage Dodge trigger enabled; (8) never spend
Action Surge (turn-hygiene cost).
M6 changes: (9) adaptive opening trigger, but stop advancing once one living ally is deep
(single-diver plan);
(10) Disengage before an otherwise planned dive when Surge remains;
(11) incoming-damage Dodge trigger; (12) hold everywhere; (13) dive everywhere.
Direction hypotheses: baseline better except Dodge (unknown); conditional plan
should benefit durable Brutes but not soft ones. The two prespecified M6 strata
(opening Brute HP / normal hit chance >=42, <42) are descriptive secondary
analyses, not additional confirmed claims.

## Read-only mining

Stream all final-policy traces, not v1; summarize first kills, damage in excess
of target's remaining HP, END_TURN with legal Attack/Surge/Second Wind/Trip,
and encounter features. PPO final models s0–s2 remain separate for outcomes;
pooled comparisons average models within encounter. Report both unique PPO
wins and unique heuristic wins. Inspect opening-feature strata (durable Brute,
Archer HP<=14, >=2 WEAKEST enemies, total opening enemy HP>=60); all trace
stratification is grade C, with pointwise paired CIs, not causal or confirmatory.
Recompute quoted historical contrasts and corrected local diagnostic values
from preserved raw outputs wherever available.

## Fixed-policy dice replay (not optimal loss attribution)

Choose 32 baseline wins and 32 baseline losses per stage by evenly spaced
indices in sorted old encounter seeds (3000–7999). Restore each opening
observation and replay the same baseline with the same 128 combat RNG seeds
236000–236127. This estimates fixed-policy win probabilities and whether old
losses can win under other dice; it cannot identify truly unwinnable fights or
fraction of losses attributable to decisions. No action-value maxima and no
training. Record binomial uncertainty, sampled-state selection and costs.

Seeds: exploration 230000–230007 smoke validation; 230008–231999 unused; confirmation 232000–235999;
spare 236000–236127 combat RNG only (no newly sampled encounters).
Compute: serial worker, compact outcomes only, target <3 CPU-hours.

Deliverables: <=2-page main playbook, separate inventory/provenance appendix,
committed reproduction script; raw outputs runs/sim_mining (ignored).

Pre-run clarification: split_focus measures ally-1 target-priority change,
not a pure general focus-fire effect. one_diver retains adaptive durability
trigger; only the number of divers changes.
"""



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
    """Named research ablations; existing variants never mutate.

    2026-10-15, seeds 232000–235999, baseline minus variant, simultaneous 95%:
    M5 split_focus +22.38 [19.78,24.97] pp; healer_first +4.05 [1.93,6.17];
    surge_round2 +11.70 [8.91,14.49]; sw10 +1.65 [.51,2.79]; dodge +1.70
    [.48,2.92]; no_surge +31.18 [28.31,34.04]. These pass exact adjusted p.
    M6 one_diver +6.78 [4.88,8.67]; disengage +9.18 [7.11,11.24]; dodge
    +3.57 [2.07,5.08]; hold +2.00 [.03,3.97]; dive +1.80 [.06,3.54], A.
    Cleave2 and trip_threat are unresolved (see module caveats/summary).
    Split focus changes ally-1 HP priority; it is not a universal causal
    focus-fire value. One-diver retains the adaptive opening trigger, stopping
    further Advances while a living ally is deep. Disengage occurs only before
    an otherwise planned dive while Surge remains. Dodge uses the existing
    incoming>=75% HP trigger. These are simulator findings, TODO_VERIFY in BG3.
    """
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


@lru_cache(maxsize=None)
def expected_overkill(hp: int, critical: bool, divisor: int) -> tuple[float,float]:
    damage=DamageSpec(1,8,3)
    dice=_dice_sum_distribution(damage_dice_count(damage,critical),damage.die_size)
    values=[(scale_damage(damage_from_dice(total,damage),divisor),float(p)) for total,p in dice.items()]
    return (sum(max(0,d-hp)*p for d,p in values),sum(d*p for d,p in values))


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
            # Trace array is slot-aligned, with None for dead/unreachable targets.
            assert len(info['cleave_attacks'])==3
            attacks=[(i,attack) for i,attack in enumerate(info['cleave_attacks']) if attack is not None]
        for target, attack in attacks:
            damage=int(attack['damage'])
            counts['recorded_capped_damage'] += damage
            if attack['hit']:
                waste,uncapped=expected_overkill(int(o[fields[f'enemy_{target}_hp']]),bool(attack['critical']),2 if action=='CLEAVE' else 1)
                counts['expected_overkill'] += waste
                counts['expected_uncapped_damage'] += uncapped
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
    m2path=LEGACY/'runs'/'m2_exact_optimum_9000_13999'/'optimal_and_heuristic_by_seed.json'
    m2=json.loads(m2path.read_text())
    m2_summary={key:float(np.mean([r[key] for r in m2])) for key in ('optimal','heuristic')}
    # Compact observations/metrics, never copy giant source traces.
    write('trace_analysis.json',{'stages':all_outputs,'inventory':inventory,'historical_local':local,
        'm2_exact_expected':m2_summary,'cpu_seconds':time.process_time()-cpu,
        'wall_seconds':time.perf_counter()-start})


def killorders() -> None:
    start,cpu=time.perf_counter(),time.process_time()
    output={}
    for stage in BASE:
        output[stage]={}
        for directory in sorted((HIST/'v2'/stage).iterdir()):
            path=directory/'episodes.jsonl'
            if path.exists():
                counts=Counter()
                with path.open() as handle:
                    for line in handle:
                        roles=json.loads(line)['result'].get('kill_order_roles') or []
                        counts['>'.join(roles) if roles else 'NONE']+=1
                output[stage][directory.name]=dict(counts)
    output['metadata']={'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    write('killorders.json',output)


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
    global HIST, OUT, LEGACY
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['confirmation','traces','replay','killorders'])
    parser.add_argument('--historical-root',type=Path,default=HIST,help='decision-stages/runs input directory (read-only)')
    parser.add_argument('--legacy-root',type=Path,default=LEGACY,help='Original BaldurRL repository containing M2 runs')
    parser.add_argument('--output-root',type=Path,default=OUT,help='New compact output directory')
    args=parser.parse_args()
    HIST,OUT,LEGACY=args.historical_root,args.output_root,args.legacy_root
    filename={'confirmation':'confirmation.json','traces':'trace_analysis.json','replay':'replay.json','killorders':'killorders.json'}[args.command]
    if (OUT/filename).exists():
        raise FileExistsError(f'Refusing to rerun into existing output: {OUT/filename}')
    {'confirmation':confirmation,'traces':traces,'replay':replay,'killorders':killorders}[args.command]()

if __name__=='__main__':
    main()
