"""Nominal M6 headroom evidence; conservative CIs remain inconclusive (2026-10-15).

Confirmation: 1,000 paired encounters per stage, seeds 223000–223999.
M5: probe 80.3% vs tuned 78.6%; delta +1.70 pp;
  conservative 95% CI [-1.23, +4.61] pp; exact McNemar p=0.07842.
  approximate paired-normal 95% CI [-0.08, +3.48] pp.
M6: probe 81.6% vs adaptive 79.6%; delta +2.00 pp;
  conservative 95% CI [-1.03, +5.02] pp; exact McNemar p=0.0446.
  approximate paired-normal 95% CI [+0.14, +3.86] pp.
P values are pointwise, unadjusted for the two stage comparisons.
Conservative intervals crossing zero do not establish improvement. Their upper bounds
apply to this tested policy, not all one-step policies or the optimum.

Method: tuned/adaptive continuation; common simulation seeds; one-sided
exact paired McNemar gates, alpha=.05 divided by all alternatives and
three looks (32/128/1024). Nonpositive candidates stop after each look.
Every legal action is estimated. Per-action budgets are logged. Gate
control is per decision, not an episode-wide safety guarantee.
Configuration and analysis committed before confirmation: 6c2162e.

Measured experiment cost 18.44 CPU-hours (worker/recheck bodies; tests/startup excluded).
M5: 1.134 planning wall seconds/decision; 421/28557 deviations
  (1.47%); 326/1000 fights (32.6%).
M6: 1.160 planning wall seconds/decision; 387/28615 deviations
  (1.35%); 294/1000 fights (29.4%).
M5 most frequent switch: ACTION_SURGE -> END_TURN, n=107, estimated gain +8.91 pp.
M6 most frequent switch: ACTION_SURGE -> END_TURN, n=116, estimated gain +10.38 pp.
Fresh 1,024-pair checks, M5: 12 states, 10 positive; mean local gain +4.96 pp
  (conditional Monte Carlo SE 0.45 pp; this is a sampled-state check).
Fresh 1,024-pair checks, M6: 12 states, 9 positive; mean local gain +3.17 pp
  (conditional Monte Carlo SE 0.45 pp; this is a sampled-state check).
All action-pair counts/mean estimated gains, approximate paired CIs and
raw-output references: evaluation/headroom_probe_results.json.
Pilots/tests: 220000–220007, 220010–220013; selection: 221000–221011,
221100–221111. Spare and frozen final-test ranges remain untouched.
Only 1,000 fights/stage: small 1–2 pp whole-fight gains may remain unresolved.
Limits: early futility/finite budgets miss gains; values use heuristic
continuation. Simulator tactics have not been validated in BG3.
Reproduce via evaluation.headroom_probe (committed confirmation constant).
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb, isfinite
from time import perf_counter
import numpy as np
from combat.simulation import simulation_from_observation
from combat.stages import make_env
from .heuristic_agent import make_heuristic_agent
from .rollout_agent import _choose


@lru_cache(maxsize=16384)
def paired_pvalue(positive: int, negative: int) -> float:
    """One-sided exact McNemar p-value for positive paired win advantage."""
    n = positive + negative
    if positive <= negative:
        return 1.0
    return sum(comb(n, k) for k in range(positive, n + 1)) / 2**n


@dataclass(frozen=True)
class ImprovementDiagnostics:
    anchor: int
    chosen: int
    legal: tuple[int, ...]
    scores: tuple[tuple[int, float], ...]
    paired_counts: tuple[tuple[int, int, int], ...]
    pvalues: tuple[tuple[int, float], ...]
    samples_per_action: int
    sample_counts: tuple[tuple[int, int], ...]
    simulated_episodes: int
    planning_seconds: float
    stopped: str


class ImprovedHeuristicAgent:
    """Exact paired tests with Bonferroni over alternatives and fixed looks.

    alpha controls false-positive comparisons at ONE decision, not a whole
    fight. Futility only retains the anchor and cannot inflate that bound.
    alpha=0 is infinite strictness. No live combat seed or RNG is accessed.
    """
    def __init__(self, stage: str, *, seed: int = 0, alpha: float = 0.05,
                 looks: tuple[int, ...] = (32, 128, 1024), futility_margin: float = 0.0,
                 prune_nonpositive: bool = True) -> None:
        if stage not in ("m5", "m6"):
            raise ValueError("Improved heuristic supports m5 and m6")
        if not isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        if (not looks or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in looks)
                or tuple(sorted(set(looks))) != looks):
            raise ValueError("looks must be strictly increasing positive integers")
        if not isfinite(futility_margin) or futility_margin < 0:
            raise ValueError("futility_margin must be finite and nonnegative")
        self.stage, self.alpha, self.looks = stage, alpha, looks
        self.futility_margin = futility_margin
        self.prune_nonpositive = prune_nonpositive
        self.rng = np.random.default_rng(seed)
        self.heuristic = make_heuristic_agent(stage, "tuned" if stage == "m5" else "adaptive")
        env = make_env(stage)
        self.observation_shape, self.action_count = env.observation_space.shape, env.action_space.n
        env.close()
        self.last_diagnostics: ImprovementDiagnostics | None = None

    def _outcome(self, observation: np.ndarray, action: int, seed: int) -> int:
        env = simulation_from_observation(self.stage, observation, seed=seed)
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
            while not (terminated or truncated):
                obs, reward, terminated, truncated, _ = env.step(_choose(self.heuristic, obs, env.action_masks()))
            return int(terminated and reward == 1.0)
        finally:
            env.close()

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        tick = perf_counter()
        obs, mask = np.asarray(observation).copy(), np.asarray(action_mask, dtype=np.bool_).copy()
        if obs.shape != self.observation_shape or mask.shape != (self.action_count,):
            raise ValueError("Observation/mask shape does not match stage")
        legal = tuple(int(a) for a in np.flatnonzero(mask))
        if not legal:
            raise ValueError("No legal actions")
        anchor = _choose(self.heuristic, obs, mask)
        chosen, n = anchor, 0
        scores, counts, pvalues, sample_counts = (), (), (), ()
        episodes = 0
        reason = "forced" if len(legal) == 1 else "disabled"
        if len(legal) > 1 and self.alpha > 0:
            outcomes: dict[int, list[int]] = {a: [] for a in legal}
            threshold = self.alpha / ((len(legal) - 1) * len(self.looks))
            active = set(legal)
            for budget in self.looks:
                seeds = self.rng.integers(0, 2**63 - 1, size=budget - n)
                for a in legal:
                    if a not in active:
                        continue
                    outcomes[a].extend(self._outcome(obs, a, int(seed)) for seed in seeds)
                n = budget
                h = np.asarray(outcomes[anchor], dtype=np.int8)
                diffs = {a: np.asarray(outcomes[a], dtype=np.int8) - h[:len(outcomes[a])]
                         for a in legal if a != anchor}
                scores = tuple((a, float(np.mean(outcomes[a]))) for a in legal)
                counts = tuple((a, int(np.sum(d > 0)), int(np.sum(d < 0))) for a, d in diffs.items())
                pvalues = tuple((a, paired_pvalue(pos, neg)) for a, pos, neg in counts)
                passing = [a for a, p in pvalues if a in active and diffs[a].mean() > 0 and p <= threshold]
                if passing:
                    chosen = max(passing, key=lambda a: (np.mean(outcomes[a]), -a))
                    reason = "significant"
                    break
                reason = "budget"
                if self.prune_nonpositive:
                    active = {anchor} | {a for a in active if a != anchor and float(diffs[a].mean()) > self.futility_margin}
                remaining = [float(diffs[a].mean()) for a in active if a != anchor]
                if not remaining or max(remaining) <= self.futility_margin:
                    reason = "futility"
                    break
            sample_counts = tuple((a, len(outcomes[a])) for a in legal)
            episodes = sum(len(outcomes[a]) for a in legal)
        self.last_diagnostics = ImprovementDiagnostics(anchor, chosen, legal, scores, counts, pvalues,
                                                       n, sample_counts, episodes, perf_counter() - tick, reason)
        return chosen
