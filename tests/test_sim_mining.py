"""Focused checks for research wrappers and paired statistics."""
import numpy as np

from agents.tactical_heuristics import make_tactical_heuristic
from combat.stages import make_env
from evaluation.sim_mining import MiningPolicy, contrast, exact_mcnemar, expected_overkill, wilson


def test_exact_discordant_binomial_and_pairing() -> None:
    assert exact_mcnemar(0,0)==1
    assert abs(exact_mcnemar(10,0)-2/1024)<1e-12
    result=contrast([1,1,0,0],[0,0,1,0],13)
    assert result['first_only']==2 and result['second_only']==1
    assert result['difference']==.25
    assert result['simultaneous_ci'][0]<result['ci95'][0]
    assert 0< wilson(0,128)[1]<.03
    assert wilson(128,128)[0]>.97


def test_baseline_wrapper_reproduces_full_seeded_fights() -> None:
    # Assigned exploratory smoke seeds, never confirmation/frozen seeds.
    for stage,base in [('m5','tuned'),('m6','adaptive')]:
        a,b=make_env(stage),make_env(stage)
        oa,_=a.reset(seed=230000)
        ob,_=b.reset(seed=230000)
        pa,pb=MiningPolicy(stage),make_tactical_heuristic(stage,base)
        for _ in range(1000):
            np.testing.assert_array_equal(oa,ob)
            aa=pa.choose_action(oa,a.action_masks())
            ab=pb.choose_action(ob,b.action_masks())
            assert aa==ab
            oa,ra,da,ta,_=a.step(aa)
            ob,rb,db,tb,_=b.step(ab)
            assert (ra,da,ta)==(rb,db,tb)
            if da or ta:
                break
        else:
            raise AssertionError('Fight did not terminate')


def test_one_diver_keeps_trigger_and_prevents_second_advance() -> None:
    env=make_env('m6')
    obs,_=env.reset(seed=230001)
    policy=MiningPolicy('m6','one_diver')
    fields=policy._index
    allies=[policy._ally(obs,i) for i in range(2)]
    enemies=[policy._enemy(obs,i) for i in range(3)]
    assert policy._movement(env.action_masks(),allies[0],allies,enemies)==MiningPolicy('m6')._movement(env.action_masks(),allies[0],allies,enemies)
    obs[fields['ally_0_deep']]=1
    obs[0]=1
    allies=[policy._ally(obs,i) for i in range(2)]
    assert policy._movement(np.ones(13,dtype=bool),allies[1],allies,enemies) is None


def test_overkill_models_uncapped_dice_instead_of_capped_trace_damage() -> None:
    waste,total=expected_overkill(1,False,1)
    assert total==7.5 and waste==6.5
    assert expected_overkill(20,False,1)[0]==0


def test_document_replacement_preserves_neighbors_and_is_idempotent() -> None:
    from evaluation.sim_mining_render import replace_knowledge_section
    header='## Simulator tactical evidence (hypotheses for BG3)'
    current='# Existing document\n\nUnrelated intro.\n\n'+header+'\n\nOld.\n\n## User notes\n\nKeep this exactly.\n'
    section=header+'\n\nNew evidence.\n'
    replaced=replace_knowledge_section(current,section)
    assert replaced.startswith('# Existing document\n\nUnrelated intro.\n\n')
    assert replaced.endswith('## User notes\n\nKeep this exactly.\n')
    assert 'Old.' not in replaced
    assert replace_knowledge_section(replaced,section)==replaced
    assert replace_knowledge_section('# Existing document\n',section)=='# Existing document\n\n'+section


def test_existing_confirmation_output_is_rejected_before_simulation(tmp_path, monkeypatch) -> None:
    import sys
    import evaluation.sim_mining as mining
    (tmp_path/'confirmation.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['sim_mining','confirmation','--output-root',str(tmp_path)])
    import pytest
    with pytest.raises(FileExistsError,match='Refusing to rerun'):
        mining.main()


def test_committed_summary_renders_seven_verified_hypothesis_entries() -> None:
    import json
    from evaluation.sim_mining_render import ROOT, render
    summary=json.loads((ROOT/'evaluation'/'sim_mining_summary.json').read_text())
    section=render(summary)
    for i in range(1,8):
        assert f'{i}. **TODO_VERIFY' in section
    assert section.count('Reproducer:')==7
    assert len(section.split())<1000
    assert 'appendix' not in section
