"""Legal-action policy priors for guided search and expert iteration."""
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .policy import Policy
from .value_functions import make_layout, value_features


class PolicyPrior(Protocol):
    def probabilities(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray: ...


class HeuristicPrior:
    """A heuristic preference with positive support for every legal action."""
    def __init__(self, policy: Policy, preference: float = 0.8) -> None:
        if not 0 <= preference < 1:
            raise ValueError('preference must be in [0, 1)')
        self.policy = policy
        self.preference = preference

    def probabilities(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError('No legal actions')
        result = np.asarray(mask, dtype=float) * ((1 - self.preference) / len(legal))
        choice = self.policy.choose_action(observation, mask)
        if choice not in legal:
            raise ValueError('Heuristic selected an illegal action')
        result[choice] += self.preference
        return result


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if logits.shape != mask.shape or logits.ndim != 2 or np.any(~mask.any(axis=1)):
        raise ValueError('Expected matching batches with at least one legal action per row')
    if not np.all(np.isfinite(logits[mask])):
        raise ValueError('Legal logits must be finite')
    scores = np.where(mask, logits, -np.inf)
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


class MLPPolicyPrior:
    """NumPy policy MLP; masks are applied both in fitting and inference."""
    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], stage: str) -> None:
        self.layout = make_layout(stage)
        self.weights = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float)) for w, b in weights]

    @classmethod
    def load(cls, path: str | Path) -> 'MLPPolicyPrior':
        with np.load(path) as data:
            meta = json.loads(str(data['meta']))
            if meta.get('kind') != 'policy_prior':
                raise ValueError('Expected a policy-prior checkpoint')
            return cls([(data[f'w{i}'], data[f'b{i}']) for i in range(meta['layers'])], meta['stage'])

    def batch_probabilities(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        x = value_features(self.layout, observations)
        for i, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
        return masked_softmax(x, masks)

    def probabilities(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.batch_probabilities(observation[None], mask[None])[0]

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return int(np.argmax(self.probabilities(observation, action_mask)))
