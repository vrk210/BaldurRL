"""Depth-limited expectimax over exact chance outcomes, with pluggable leaf values.

The agent reconstructs the encounter from the observation, asks the stage's
exact `TransitionModel` for legal decisions and merged successor distributions,
and maximizes win probability: max over the active ally's decisions,
expectation over chance outcomes. Depth counts controlled-turn boundaries: each
`END_TURN` (which includes any following automatic enemy phase) consumes one
unit. Depth 1 therefore searches the rest of the current ally's turn plus the
transition into the next controlled turn; ongoing states reached with no depth
left are scored by the leaf value. Terminal states score exactly (victory 1,
defeat or truncation 0). `depth=None` searches to termination (exact, feasible
only for small states).

Leaf values are evaluated in one batch per search, after expansion, and a
transposition table is shared by all decisions of the same controlled turn.
With `node_budget`, a search that would expand more decision nodes than the
budget is abandoned and the next shallower depth is used instead (depth 1 is
never budgeted), so deeper search is spent only where it is affordable.
The agent contains no combat rules; all dynamics come from `combat.transitions`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from combat.stages import make_env
from combat.transitions import State, Status, TransitionModel, transition_model


LeafValue = Callable[[np.ndarray], np.ndarray]
"""Maps a batch of observations (n, fields) to win probabilities (n,)."""


@dataclass(frozen=True)
class ExpectimaxDiagnostics:
    q_values: tuple[tuple[int, float], ...]
    value: float
    expanded_nodes: int
    leaf_evaluations: int
    planning_seconds: float
    depth_used: int | None = None


class _BudgetExceeded(Exception):
    pass


@dataclass
class _SearchStats:
    expanded: int = 0
    leaves: int = 0
    depth_used: int | None = None


_TIE_TOLERANCE = 1e-12


class ExpectimaxAgent:
    """Maximize exact expected win probability to a fixed turn depth.

    `always_search=True` also searches forced (single-action) decisions so that
    diagnostics carry a root value, e.g. for value-function training targets.
    """

    def __init__(
        self,
        stage: str,
        *,
        depth: int | None = 1,
        leaf_value: LeafValue | None = None,
        always_search: bool = False,
        node_budget: int | None = None,
        max_cache_entries: int = 3_000_000,
    ) -> None:
        if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
            raise ValueError("depth must be a positive integer or None")
        if depth is not None and leaf_value is None:
            raise ValueError("A depth-limited search requires a leaf value")
        env = make_env(stage)
        self.action_count = env.action_space.n
        self.observation_shape = env.observation_space.shape
        env.close()
        self.stage = stage
        self.depth = depth
        self.leaf_value = leaf_value
        self.always_search = always_search
        if node_budget is not None and (depth is None or node_budget < 1):
            raise ValueError("node_budget requires a finite depth and a positive budget")
        self.node_budget = node_budget
        self.max_cache_entries = max_cache_entries
        self._model: TransitionModel | None = None
        self._values: dict[tuple[State, int | None], float] = {}
        self._turn_ended = False
        self.last_diagnostics = ExpectimaxDiagnostics((), float("nan"), 0, 0, 0.0)

    # ---------------------------------------------------------------- policy API
    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        started = perf_counter()
        observation = np.asarray(observation)
        mask = np.asarray(action_mask, dtype=np.bool_)
        if observation.shape != self.observation_shape or mask.shape != (self.action_count,):
            raise ValueError(f"Observation/mask shape does not match {self.stage}")
        legal = np.flatnonzero(mask)
        if not legal.size:
            raise ValueError("No legal actions available")
        model = self.model_for(observation)
        root = model.state_from_observation(observation)
        if tuple(int(a) for a in legal) != model.legal_actions(root):
            raise ValueError("Action mask disagrees with the exact transition model")
        # Entries never outlive a controlled turn: the next turn's root depth differs.
        if self._turn_ended or len(self._values) > self.max_cache_entries:
            self._values = {}
        if legal.size == 1 and not self.always_search:
            choice = int(legal[0])
            self.last_diagnostics = ExpectimaxDiagnostics((), float("nan"), 0, 0, perf_counter() - started)
        else:
            q_values, stats = self.action_values(root)
            best = max(q for _, q in q_values)
            choice = next(action for action, q in q_values if q >= best - _TIE_TOLERANCE)
            self.last_diagnostics = ExpectimaxDiagnostics(
                q_values, best, stats.expanded, stats.leaves, perf_counter() - started, stats.depth_used
            )
        self._turn_ended = model.ends_turn(choice)
        return choice

    def model_for(self, observation: np.ndarray) -> TransitionModel:
        """Reuse the bound model (and its caches) while the encounter is unchanged."""
        if self._model is None or not self._model.same_encounter(observation):
            self._model = transition_model(self.stage, observation)
            self._values = {}
        return self._model

    # ---------------------------------------------------------------- search
    def action_values(self, root: State) -> tuple[tuple[tuple[int, float], ...], _SearchStats]:
        """Exact-chance Q-values of every legal root decision at the configured depth."""
        model = self._model
        if model is None:
            raise RuntimeError("Call model_for(observation) first")
        stats = _SearchStats()
        depth = self.depth
        while True:
            budget = self.node_budget if depth is not None and depth > 1 else None
            try:
                root_actions = self._search(model, root, depth, stats, budget)
                break
            except _BudgetExceeded:
                depth -= 1
        stats.depth_used = depth
        values = self._values
        q_values = tuple(
            (action, sum(p * values[(child, child_depth)] for p, child in outcomes))
            for action, child_depth, outcomes in root_actions
        )
        return q_values, stats

    def state_value(self, state: State) -> float:
        """Value of a state at the configured depth (uses the shared cache)."""
        model = self._model
        if model is None:
            raise RuntimeError("Call model_for(observation) first")
        stats = _SearchStats()
        if model.status(state) is Status.ONGOING:
            self._search(model, state, self.depth, stats)
        else:
            self._values[(state, self.depth)] = 1.0 if model.status(state) is Status.WIN else 0.0
        return self._values[(state, self.depth)]

    def _search(
        self, model: TransitionModel, root: State, depth: int | None, stats: _SearchStats, budget: int | None = None
    ) -> list:
        values = self._values
        expanded: dict[tuple[State, int | None], list] = {}
        post_order: list[tuple[tuple[State, int | None], list]] = []
        leaves: dict[State, None] = {}

        def expand(state: State, remaining: int | None) -> None:
            key = (state, remaining)
            if key in values or key in expanded:
                return
            status = model.status(state)
            if status is not Status.ONGOING:
                values[key] = 1.0 if status is Status.WIN else 0.0
                return
            if remaining == 0:
                leaves[state] = None
                return
            actions: list = []
            expanded[key] = actions
            if budget is not None and len(expanded) > budget:
                raise _BudgetExceeded
            for action in model.legal_actions(state):
                child_depth = remaining - 1 if remaining is not None and model.ends_turn(action) else remaining
                outcomes = model.outcomes(state, action)
                for _, child in outcomes:
                    expand(child, child_depth)
                actions.append((action, child_depth, outcomes))
            post_order.append((key, actions))

        root_key = (root, depth)
        if root_key in values and root_key not in expanded:
            # Root value cached from an earlier decision; recompute its action list.
            actions = []
            for action in model.legal_actions(root):
                child_depth = depth - 1 if depth is not None and model.ends_turn(action) else depth
                outcomes = model.outcomes(root, action)
                for _, child in outcomes:
                    expand(child, child_depth)
                actions.append((action, child_depth, outcomes))
            root_actions = actions
        else:
            expand(root, depth)
            root_actions = expanded[root_key]

        if leaves:
            states = list(leaves)
            if self.leaf_value is None:
                raise RuntimeError("Leaf reached without a leaf value")
            leaf_values = np.asarray(self.leaf_value(model.observations(states)), dtype=np.float64).reshape(-1)
            if leaf_values.shape != (len(states),):
                raise ValueError("Leaf value must return one value per observation")
            for state, value in zip(states, leaf_values):
                values[(state, 0)] = float(value)
        for key, actions in post_order:
            values[key] = max(
                sum(p * values[(child, child_depth)] for p, child in outcomes)
                for _, child_depth, outcomes in actions
            )
        stats.expanded += len(post_order)
        stats.leaves += len(leaves)
        return root_actions
