"""Paired run-card comparison arithmetic."""

import json
from math import sqrt

import pytest

from evaluation.paired import compare_cards, paired_comparison, wins_from_card


def test_wins_from_card_excludes_losses_and_truncations() -> None:
    card = {"episode_seeds": [1, 2, 3, 4], "loss_seeds": [2], "truncation_seeds": [4]}
    assert wins_from_card(card) == {1: 1, 2: 0, 3: 1, 4: 0}


def test_paired_difference_and_standard_error() -> None:
    first = {1: 1, 2: 1, 3: 0, 4: 1}
    second = {1: 1, 2: 0, 3: 0, 4: 0}
    result = paired_comparison(first, second)
    assert result["difference"] == 0.5
    assert result["paired_se"] == pytest.approx(sqrt((2 * 0.25 + 2 * 0.25) / 3 / 4))
    assert (result["first_only_wins"], result["second_only_wins"]) == (2, 0)
    with pytest.raises(ValueError):
        paired_comparison(first, {1: 1})


def test_compare_cards_reads_files_and_checks_stage(tmp_path) -> None:
    base = {"episode_seeds": [5, 6], "truncation_seeds": []}
    (tmp_path / "a.json").write_text(json.dumps({**base, "stage": "m5", "loss_seeds": [6]}))
    (tmp_path / "b.json").write_text(json.dumps({**base, "stage": "m5", "loss_seeds": [5, 6]}))
    (tmp_path / "c.json").write_text(json.dumps({**base, "stage": "m6", "loss_seeds": []}))
    assert compare_cards(tmp_path / "a.json", tmp_path / "b.json")["difference"] == 0.5
    with pytest.raises(ValueError, match="stages"):
        compare_cards(tmp_path / "a.json", tmp_path / "c.json")


def test_tactical_behavior_summary_reads_traces(tmp_path) -> None:
    from agents.heuristic_agent import make_heuristic_agent
    from evaluation.evaluate import evaluate
    from evaluation.tactical_behavior import summarize_run

    for stage, variant in (("m5", "trip_focus"), ("m6", "dive_all")):
        path = tmp_path / f"{stage}.jsonl"
        summary = evaluate(make_heuristic_agent(stage, variant), range(10), trace_path=path, stage=stage)
        report = summarize_run(stage, path)
        assert report["episodes"] == 10 and report["win_rate"] == summary.win_rate
        assert report["per_episode"]["TRIP"] > 0 and 0 < report["trip_then_attack_rate"] <= 1
        assert sum(report["first_kill_role"].values()) <= 1
        if stage == "m6":
            assert report["m6"]["episodes_with_advance"] == 1.0
