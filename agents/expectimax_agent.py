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

With `afterstate_value`, an `END_TURN` that would use the last unit of depth is
not expanded through the automatic phase. Instead it is scored by the
afterstate value of the state in which the turn ends (Sutton & Barto 6.8): the
search covers the controlled turn's own decisions and dice exactly, and a
learned function scores the hand-over. This avoids enumerating long enemy
phases, which dominate cost on M5/M6.

`decision_horizon=1` limits the search to the next controlled decision: each
legal action's exact outcomes are scored by `leaf_value` (ongoing states) or
exactly (terminal states), and `END_TURN` by `afterstate_value`. On M5/M6 a
whole turn can branch into hundreds of thousands of nodes (one three-target
Cleave alone has hundreds of outcomes), so the one-decision horizon keeps the
cost bounded by the root's outcomes.
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
_AFTER = "after"
"""Depth marker for afterstate-scored END_TURN children."""


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
        afterstate_value: LeafValue | None = None,
        decision_horizon: int | None = None,
    ) -> None:
        if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
            raise ValueError("depth must be a positive integer or None")
        if depth is not None and leaf_value is None and afterstate_value is None:
            raise ValueError("A depth-limited search requires a leaf value")
        if afterstate_value is not None and depth is None:
            raise ValueError("An afterstate value requires a finite depth")
        env = make_env(stage)
        self.action_count = env.action_space.n
        self.observation_shape = env.observation_space.shape
        env.close()
        self.stage = stage
        self.depth = depth
        self.leaf_value = leaf_value
        self.always_search = always_search
        self.afterstate_value = afterstate_value
        if decision_horizon not in (None, 1):
            raise ValueError("Only decision_horizon=1 is supported")
        if decision_horizon == 1 and (leaf_value is None or afterstate_value is None or depth != 1):
            raise ValueError("decision_horizon=1 needs depth=1, a leaf value, and an afterstate value")
        self.decision_horizon = decision_horizon
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
        elif self.decision_horizon == 1:
            q_values, stats = self.one_decision_values(root)
            best = max(q for _, q in q_values)
            choice = next(action for action, q in q_values if q >= best - _TIE_TOLERANCE)
            self.last_diagnostics = ExpectimaxDiagnostics(
                q_values, best, stats.expanded, stats.leaves, perf_counter() - started, 0
            )
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

    def one_decision_values(self, root: State) -> tuple[tuple[tuple[int, float], ...], _SearchStats]:
        """Exact-chance Q-values over the next decision only (see `decision_horizon`)."""
        model = self._model
        if model is None:
            raise RuntimeError("Call model_for(observation) first")
        stats = _SearchStats(expanded=1)
        plans: list[tuple[int, object]] = []
        ongoing: dict[State, None] = {}
        for action in model.legal_actions(root):
            if model.ends_turn(action):
                plans.append((action, None))
                continue
            outcomes = model.outcomes(root, action)
            for _, child in outcomes:
                if model.status(child) is Status.ONGOING:
                    ongoing[child] = None
            plans.append((action, outcomes))
        states = list(ongoing)
        values: dict[State, float] = {}
        if states:
            leaf = np.asarray(self.leaf_value(model.observations(states)), dtype=np.float64).reshape(-1)
            values = dict(zip(states, leaf.tolist()))
        after = float(np.asarray(self.afterstate_value(model.observations([root]))).reshape(-1)[0])
        stats.leaves = len(states) + 1

        def score(child: State) -> float:
            status = model.status(child)
            if status is Status.ONGOING:
                return values[child]
            return 1.0 if status is Status.WIN else 0.0

        q_values = tuple(
            (action, after if outcomes is None else sum(p * score(child) for p, child in outcomes))
            for action, outcomes in plans
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
        afterstates: dict[State, None] = {}

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
                actions.append(child(state, remaining, action))
            post_order.append((key, actions))

        def child(state: State, remaining: int | None, action: int) -> tuple:
            if self.afterstate_value is not None and remaining == 1 and model.ends_turn(action):
                afterstates[state] = None
                return action, _AFTER, ((1.0, state),)
            child_depth = remaining - 1 if remaining is not None and model.ends_turn(action) else remaining
            outcomes = model.outcomes(state, action)
            for _, successor in outcomes:
                expand(successor, child_depth)
            return action, child_depth, outcomes

        root_key = (root, depth)
        if root_key in values and root_key not in expanded:
            # Root value cached from an earlier decision; recompute its action list.
            root_actions = [child(root, depth, action) for action in model.legal_actions(root)]
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
        pending = [state for state in afterstates if (state, _AFTER) not in values]
        if pending:
            after_values = np.asarray(
                self.afterstate_value(model.observations(pending)), dtype=np.float64
            ).reshape(-1)
            if after_values.shape != (len(pending),):
                raise ValueError("Afterstate value must return one value per observation")
            for state, value in zip(pending, after_values):
                values[(state, _AFTER)] = float(value)
        for key, actions in post_order:
            values[key] = max(
                sum(p * values[(child, child_depth)] for p, child in outcomes)
                for _, child_depth, outcomes in actions
            )
        stats.expanded += len(post_order)
        stats.leaves += len(leaves) + len(pending)
        return root_actions
