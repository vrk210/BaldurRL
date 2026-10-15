"""Conservative one-step improvement with paired multiplicity-safe gates."""
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
    def __init__(self, stage: str, *, seed: int = 0, alpha: float = 0.01,
                 looks: tuple[int, ...] = (16, 64, 256), futility_margin: float = 0.0,
                 prune_nonpositive: bool = False) -> None:
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
