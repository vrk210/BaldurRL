"""Reduce compact mining outputs and regenerate the committed playbook/appendix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.sim_mining import BASE, OUT, ROOT, contrast, wilson


def pct(x: float) -> str:
    return f'{100*x:.2f}'


def effect(x: dict[str,Any], simultaneous: bool = True) -> str:
    lo,hi=x['simultaneous_ci' if simultaneous else 'ci95']
    return f"{100*x['difference']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]"


def ratio(rows: list[dict[str,Any]], numerator: str, denominator: str) -> dict[str,Any]:
    x=np.array([r['counts'].get(numerator,0) for r in rows],dtype=float)
    y=np.array([r['counts'].get(denominator,0) for r in rows],dtype=float)
    r=float(x.sum()/y.sum()) if y.sum() else 0.
    se=float((x-r*y).std(ddof=1)/y.mean()/np.sqrt(len(x))) if y.sum() else 0.
    return {'ratio':r,'ci95':[max(0.,r-1.96*se),min(1.,r+1.96*se)]}


def rate(x: dict[str,Any]) -> str:
    return f"{pct(x['ratio'])}% [{pct(x['ci95'][0])}, {pct(x['ci95'][1])}]"


def reduce() -> dict[str,Any]:
    t=json.loads((OUT/'trace_analysis.json').read_text())
    c=json.loads((OUT/'confirmation.json').read_text())
    r=json.loads((OUT/'replay.json').read_text())
    orders=json.loads((OUT/'killorders.json').read_text())
    s={'confirmation':{stage:c[stage]['baseline_minus_variant'] for stage in BASE},
       'confirmation_metadata':c['metadata'],'inventory':t['inventory'],
       'historical_local':{stage:[{k:v for k,v in row.items() if k in
           ('kind','count','weighted_mean_difference','approximate_95pct_ci','sample_count')} for row in t['historical_local'][stage]['kinds']] for stage in BASE},
       'm2_exact_expected':t['m2_exact_expected'],'stages':{},'replay':r,
       'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (OUT/'trace_analysis.json',OUT/'confirmation.json',OUT/'replay.json',OUT/'killorders.json')},
       'kill_order_metadata':orders['metadata'],
       'cpu_seconds':t['cpu_seconds']+c['metadata']['cpu_seconds']+r['metadata']['cpu_seconds']+orders['metadata']['cpu_seconds']}
    for stage in BASE:
        ts=t['stages'][stage]
        runs={}
        for name,run in ts['runs'].items():
            rows=run['episodes']
            runs[name]={
                'win_rate':sum(e['won'] for e in rows)/len(rows),
                'first_kills':run['first_kills'],
                'full_and_partial_kill_order_counts':orders[stage][name],
                'end_legal':{a:ratio(rows,'end_legal_'+a,'ends') for a in ('attack','surge','wind','trip')},
                'expected_overkill':ratio(rows,'expected_overkill','expected_uncapped_damage'),
                'ppo_minus_baseline':ts['paired_minus_baseline'][name]}
        s['stages'][stage]={'runs':runs,'pooled_strata':ts['pooled_strata'],
            'pooled_ppo_minus_baseline':ts['pooled_ppo_minus_baseline'],
            'all_tested_policies_lost':ts['all_tested_policies_lost']}
    return s


def render(s: dict[str,Any]) -> str:
    m5,m6=s['confirmation']['m5'],s['confirmation']['m6']
    run5=s['stages']['m5']['runs'];run6=s['stages']['m6']['runs']
    def ci(st: str, kind: list[str]) -> str:
        row=next(r for r in s['historical_local'][st] if r['kind']==kind)
        lo,hi=row['approximate_95pct_ci']
        return f"{100*row['weighted_mean_difference']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]"
    def uniques(stage: str) -> str:
        return '/'.join(str(s['stages'][stage]['runs'][f'ppo_s{i}']['ppo_minus_baseline']['first_only']) for i in range(3))
    def lossrange(stage: str) -> str:
        rows=[r for r in s['replay'][stage] if not r['old_won']]
        return f"{100*min(r['wins']/r['n'] for r in rows):.1f}–{100*max(r['wins']/r['n'] for r in rows):.1f}%"
    def order_rate(stage: str, order: str) -> str:
        k=s['stages'][stage]['runs'][BASE[stage]]['full_and_partial_kill_order_counts'][order]
        ci=wilson(k,5000)
        return f"{k/50:.2f}% [{100*ci[0]:.2f}, {100*ci[1]:.2f}]"
    def loss_feature_effect(stage: str, feature: str) -> str:
        rows=s['stages'][stage]['pooled_strata']
        a,b=rows[feature+'=1'],rows[feature+'=0']
        pa,pb=a['baseline_loss_rate'],b['baseline_loss_rate']
        na,nb=a['ppo_minus_baseline']['n'],b['ppo_minus_baseline']['n']
        se=(pa*(1-pa)/na+pb*(1-pb)/nb)**.5
        return f"{100*(pa-pb):+.2f} pp [{100*(pa-pb-1.96*se):+.2f}, {100*(pa-pb+1.96*se):+.2f}]"
    main=f'''## Simulator tactical evidence (hypotheses for BG3)

Use these as simulator-backed starting rules, then verify the named mechanics
in BG3. **A** = paired single-rule intervention with simultaneous CI excluding
zero **and** Bonferroni-adjusted exact McNemar p<.05; **B** = replayed local
counterfactual; **C** = observational hypothesis. Brackets are 95% CIs; new A
intervals adjust across all 13 predeclared contrasts. Effects are win-rate
percentage points (pp), conditional on the tested continuation, not guarantees.
New target/resource tests use M5, positioning uses M6, and Dodge uses both.

1. **TODO_VERIFY — Focus an affordable kill; do not automatically prioritize the healer.**
   **A · general:** both allies using lowest current target HP beat changing
   ally 2 to highest-HP targeting by {effect(m5['split_focus'])}. This tests
   that exact priority change, not a universal value of concentrating attacks.
   **A · mechanic-dependent:** lowest-HP targeting beats healer-first by
   {effect(m5['healer_first'])}; check actual healing triggers, charges and
   healer durability. **C · general:** expected overkill is
   {rate(run5['tuned']['expected_overkill'])} of modeled uncapped M5 damage.
   Consider low-damage finishers, but this is not a proven benefit of switching
   targets. Logs cap damage; waste is reconstructed from exact dice given each
   observed hit/critical. **C · mechanic-dependent:** modal full kill orders
   are Archer→Healer→Brute in {order_rate('m5','ARCHER>HEALER>BRUTE')} of all
   M5 fights and Brute→Archer→Healer in {order_rate('m6','BRUTE>ARCHER>HEALER')}
   on M6; actual role stats/healing need checking.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

2. **TODO_VERIFY — Spend burst damage while it can remove enemy turns; heal before critical
   danger.** **A · general:** early Action Surge beats withholding it until
   round 2 by {effect(m5['surge_round2'])}. **A · mechanic-dependent:** healing
   at 12/20 HP instead of 10/20 gains {effect(m5['sw10'])}; verify BG3 healing
   amount, action cost and incoming damage before adopting the threshold.
   **B · mechanic-dependent:** replacing PPO's Attack with the heuristic's
   focus Trip in sampled disagreement states improves local win probability by {ci('m5',['TRIP','ATTACK'])} on M5 and
   {ci('m6',['TRIP','ATTACK'])} on M6. Targets can differ; this does not isolate same-target timing.
   Check Prone duration, accuracy,
   advantage and its competition with healing. **Unresolved intervention · mechanic-dependent:**
   requiring 3 targets rather than 2 for half-damage Cleave gives
   {effect(m5['cleave2'])}; no validated universal Cleave threshold or
   threat-target Trip advantage. Verify Cleave's real damage and reach.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

3. **TODO_VERIFY — Commit allies to one positioning plan.** **A · mechanic-dependent:**
   adaptive positioning beats stopping after one diver by
   {effect(m6['one_diver'])}, holding everywhere by {effect(m6['hold'])}, and
   diving everywhere by {effect(m6['dive'])}. The tested trigger is front
   enemy HP / normal hit chance ≥42; dive when durable, otherwise break the
   front together. Check real reach, path costs and ranged disadvantage.
   **A · mechanic-dependent:** taking the opportunity attack beats
   Disengaging before a planned dive while Surge remains by
   {effect(m6['disengage'])}. This does not show Disengage never helps.
   Earlier Advance timing remains unresolved in local replays.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

4. **TODO_VERIFY — Keep attacking by default; do not add the tested emergency Dodge rule.**
   **A · mechanic-dependent:** disabling Dodge when estimated incoming damage
   reaches 75% of current HP gains {effect(m5['dodge'])} on M5 and
   {effect(m6['dodge'])} on M6. This rejects that trigger, not every defensive
   use. Check real enemy targeting, Dodge duration and action economy; no
   beneficial Dodge state was established by this work.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

5. **TODO_VERIFY — Audit legal actions before ending a turn.** **A · general:** spending
   Surge versus leaving it unused gains {effect(m5['no_surge'])}.
   **C · general:** tuned/adaptive never end with a legal attack in these logs;
   final PPO s0 does so on {rate(run5['ppo_s0']['end_legal']['attack'])} /
   {rate(run6['ppo_s0']['end_legal']['attack'])} of M5/M6 END_TURN decisions.
   These frequencies do not measure the cost of each skipped attack. A legal
   heal or Trip can reasonably be saved; availability alone is not waste.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

6. **TODO_VERIFY — Do not label one lost fight unwinnable.** **C · unknown:** high total enemy
   HP (M5) and durable front enemies (M6) predict more baseline losses.
   M6 durable-minus-soft front loss rate: {loss_feature_effect('m6','durable_brute')}.
   Every sampled old loss won under some of the 128 replay dice seeds;
   fixed-policy replay rates span {lossrange('m5')} on M5 and {lossrange('m6')}
   on M6. These selected 32 losses per stage do not identify the irreducible
   dice share of the remaining losses. That needs an optimal-policy bound;
   verify real encounter balance and randomness before transferring.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

7. **TODO_VERIFY — Use PPO's exceptions as review examples, not replacement rules.**
   **C · unknown:** PPO s0/s1/s2 win {uniques('m5')} M5 and {uniques('m6')}
   M6 encounters that the best heuristic loses, out of 5,000 paired encounters
   each. None of the predeclared opening-feature strata establishes PPO
   superiority, and no tested local decision type reliably favors PPO.
   Pooled PPO-minus-heuristic gaps are {effect(s['stages']['m5']['pooled_ppo_minus_baseline'],False)}
   / {effect(s['stages']['m6']['pooled_ppo_minus_baseline'],False)} (M5/M6). Changed dice paths can
   produce these unique wins; inspect concrete BG3 states before inferring a
   transferable tactic.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

Sources, per-policy audits, full kill orders, inventory and provenance:
[`evaluation/sim_mining.py`](../evaluation/sim_mining.py) and its committed
[`sim_mining_summary.json`](../evaluation/sim_mining_summary.json).
'''
    return main


def replace_knowledge_section(current: str, section: str) -> str:
    """Replace only this section, preserving unrelated preceding/following text."""
    header='## Simulator tactical evidence (hypotheses for BG3)'
    start=0 if current.startswith(header) else current.find('\n'+header)+1
    if start==0 and not current.startswith(header):
        separator='' if current.endswith('\n\n') else ('\n' if current.endswith('\n') else '\n\n')
        return current+separator+section
    end=current.find('\n## ',start+len(header))
    if end<0:
        return current[:start]+section
    return current[:start]+section.rstrip('\n')+'\n'+current[end:]


def main() -> None:
    global OUT
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--from-summary',action='store_true',help='Regenerate Markdown using only committed summary.json')
    parser.add_argument('--output-root',type=Path,default=OUT,help='Raw outputs to reduce, when not using --from-summary')
    args=parser.parse_args()
    OUT=args.output_root
    summary=json.loads((ROOT/'evaluation'/'sim_mining_summary.json').read_text()) if args.from_summary else reduce()
    section=render(summary)
    summary_path=ROOT/'evaluation'/'sim_mining_summary.json'
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    knowledge=ROOT/'docs'/'GAME_KNOWLEDGE.md'
    current=knowledge.read_text()
    knowledge.write_text(replace_knowledge_section(current,section))
    print('Playbook section words:',len(section.split()))

if __name__=='__main__':
    main()
