"""Boundary-aware paired inference and process-scheduling seed invariance."""
import pytest
from evaluation.headroom_probe import paired_summary, run_episode


def test_no_discordances_has_nonzero_upper_and_lower_bounds():
    wins = {i: int(i % 2) for i in range(20)}
    result = paired_summary(wins, wins)
    assert result['difference'] == 0
    assert result['mcnemar_exact_two_sided_p'] == 1
    lo, hi = result['ci95_conservative']
    assert lo < 0 < hi
    assert hi == pytest.approx(1 - .0125**(1 / 20))


def test_known_mcnemar_and_direction():
    first = {i: int(i < 16) for i in range(20)}
    second = {i: 0 for i in range(20)}
    result = paired_summary(first, second)
    assert result['difference'] == .8
    assert result['mcnemar_exact_two_sided_p'] == 2**-15
    assert result['ci95_conservative'][0] > 0


def test_episode_same_ordinal_same_stream_independent_of_execution_order(tmp_path):
    configs = []
    for name in ('first', 'second'):
        out = tmp_path / name
        out.mkdir()
        configs.append(dict(stage='m5', ordinal=7, encounter=220006, agent_seed=8,
                            alpha=0., looks=[2], futility_margin=0., out=str(out)))
    first, second = [run_episode(config) for config in configs]
    for key in ('planner_seed', 'won', 'heuristic_won', 'decisions', 'deviations'):
        assert first[key] == second[key]


def test_confirmation_rejects_uncommitted_configuration_before_writing(monkeypatch, tmp_path, capsys):
    import sys
    from evaluation.headroom_probe import main

    out = tmp_path / 'not-created'
    monkeypatch.setattr(sys, 'argv', ['headroom_probe', 'run', '--stage', 'm5',
                        '--phase', 'confirmation', '--seed-start', '223000',
                        '--episodes', '999', '--out', str(out)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'must match committed CONFIRMATION_CONFIG' in capsys.readouterr().err
    assert not out.exists()
