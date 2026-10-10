"""Structural action-selection contract shared by planning and evaluation."""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int: ...
