"""Uniform selection among the environment's currently legal actions."""

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Choose one legal action index; observation is reserved for other policies."""
        legal_indices = np.flatnonzero(action_mask)
        if legal_indices.size == 0:
            raise ValueError("No legal actions available")
        return int(self.rng.choice(legal_indices))
