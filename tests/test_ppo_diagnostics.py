"""Guard against early-only sampling and biased equal-quota aggregation."""

import numpy as np
import pytest

from evaluation.ppo_diagnostics import Reservoir, summarize_strata


def test_reservoir_retains_late_occurrences_and_is_reproducible() -> None:
    samples = []
    for _ in range(2):
        reservoir = Reservoir(12)
        rng = np.random.default_rng(42)
        for occurrence in range(10000):
            reservoir.add({"occurrence": occurrence}, rng)
        assert reservoir.count == 10000
        assert len(reservoir.items) == 12
        chosen = [item["occurrence"] for item in reservoir.items]
        assert len(set(chosen)) == 12
        assert min(chosen) < 5000 < max(chosen)
        samples.append(chosen)
    assert samples[0] == samples[1]


def scored(model: int, effect: float, seed: int, mc_variance: float = 0) -> dict:
    return {"model_seed": model, "difference": effect, "episode_seed": seed,
            "mc_variance": mc_variance, "rollouts_per_action": 1024}


def test_equal_model_quotas_do_not_reweight_the_occurrence_population() -> None:
    scores = [scored(0, 1, 1), scored(0, 1, 2), scored(1, -1, 3), scored(1, -1, 4)]
    summary = summarize_strata({0: 90, 1: 10, 2: 0}, scores)
    assert summary["weighted_mean_difference"] == pytest.approx(.8)
    assert summary["by_model"][2]["sample_count"] == 0


def test_census_has_monte_carlo_uncertainty_but_no_state_sampling_uncertainty() -> None:
    summary = summarize_strata({0: 2}, [scored(0, -.2, 1, .01), scored(0, .2, 2, .01)])
    assert summary["weighted_mean_difference"] == 0
    assert summary["stratified_se"] == pytest.approx(np.sqrt(.01 / 2))
    assert summary["approximate_95pct_ci"][0] < 0 < summary["approximate_95pct_ci"][1]


def test_nonempty_stratum_cannot_be_silently_omitted() -> None:
    with pytest.raises(ValueError, match="stratum"):
        summarize_strata({0: 100, 1: 100}, [scored(0, 1, 1), scored(0, 1, 2)])
