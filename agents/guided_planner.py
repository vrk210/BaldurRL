"""Bounded policy-guided search with exact chance nodes and a conservative anchor.

The horizon counts decisions (including END_TURN's exact automatic phase).
Top-k pruning and a node budget approximate decision search, never chance
probabilities. All candidate actions get the same expansion allowance.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from combat.transitions import State, Status, TransitionModel, transition_model
from .expectimax_agent import LeafValue
from .policy import Policy
from .policy_prior import HeuristicPrior, PolicyPrior


@dataclass(frozen=True)
class GuidedDiagnostics:
    q_values: tuple[tuple[int, float], ...]
    value: float
    selected_value: float
    anchor_action: int
    proposed_action: int
    expanded_nodes: int
    leaf_evaluations: int
    budget_cutoffs: int
    planning_seconds: float
    depth_used: int


class GuidedPlannerAgent:
    def __init__(self, stage: str, leaf_value: LeafValue, anchor: Policy, *,
                 prior: PolicyPrior | None = None, horizon: int = 1,
                 root_top_k: int = 0, interior_top_k: int = 2,
                 node_budget: int = 64, deviation_margin: float = 0.03) -> None:
        if horizon < 1 or node_budget < 1 or root_top_k < 0 or interior_top_k < 1:
            raise ValueError('Invalid search horizon, budget, or top-k')
        if not np.isfinite(deviation_margin) or not 0 <= deviation_margin <= 1:
            raise ValueError('deviation_margin must be in [0, 1]')
        self.stage, self.leaf_value, self.anchor = stage, leaf_value, anchor
        self.prior = prior or HeuristicPrior(anchor)
        self.horizon, self.root_top_k, self.interior_top_k = horizon, root_top_k, interior_top_k
        self.node_budget, self.deviation_margin = node_budget, deviation_margin
        self._model: TransitionModel | None = None
        self.last_diagnostics: GuidedDiagnostics | None = None

    def _candidates(self, model: TransitionModel, state: State, k: int) -> tuple[int, ...]:
        legal = model.legal_actions(state)
        if not k or len(legal) <= k:
            return legal
        observation = model.observations([state])[0]
        mask = np.zeros(model.action_count, dtype=bool)
        mask[list(legal)] = True
        probs = np.asarray(self.prior.probabilities(observation, mask), dtype=float)
        if probs.shape != mask.shape or not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs[~mask]) or not np.isclose(probs.sum(), 1):
            raise ValueError('Prior must return normalized probabilities on legal actions')
        anchor = self.anchor.choose_action(observation, mask)
        if anchor not in legal:
            raise ValueError('Anchor selected an illegal action')
        ranked = sorted(legal, key=lambda a: (-probs[a], a))[:k]
        return tuple(sorted(set(ranked) | {anchor}))

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        started = perf_counter()
        observation = np.asarray(observation)
        mask = np.asarray(action_mask, dtype=bool)
        if not mask.any():
            raise ValueError('No legal actions')
        if self._model is None or not self._model.same_encounter(observation):
            self._model = transition_model(self.stage, observation)
        model = self._model
        root = model.state_from_observation(observation)
        if tuple(np.flatnonzero(mask)) != model.legal_actions(root):
            raise ValueError('Mask disagrees with the transition model')
        anchor = self.anchor.choose_action(observation, mask)
        if anchor not in model.legal_actions(root):
            raise ValueError('Anchor selected an illegal action')
        actions = self._candidates(model, root, self.root_top_k)
        # A separate DAG for each root candidate gives each an equal budget;
        # otherwise action enumeration order would consume a shared allowance.
        graphs = []
        leaves: dict[State, None] = {}
        expanded_total = cutoffs = 0
        for action in actions:
            values: dict[tuple[State, int], float] = {}
            visited: set[tuple[State, int]] = set()
            postorder = []
            expanded = 0

            def expand(state: State, remaining: int) -> None:
                nonlocal expanded, cutoffs
                key = (state, remaining)
                if key in visited:
                    return
                visited.add(key)
                status = model.status(state)
                if status is not Status.ONGOING:
                    values[key] = float(status is Status.WIN)
                elif remaining == 0 or expanded >= self.node_budget:
                    leaves[state] = None
                    postorder.append((key, None))
                    cutoffs += int(remaining > 0)
                else:
                    expanded += 1
                    branches = [(a, model.outcomes(state, a)) for a in self._candidates(model, state, self.interior_top_k)]
                    for _, outcomes in branches:
                        for _, child in outcomes:
                            expand(child, remaining - 1)
                    postorder.append((key, branches))

            outcomes = model.outcomes(root, action)
            for _, child in outcomes:
                expand(child, self.horizon - 1)
            graphs.append((action, outcomes, values, postorder))
            expanded_total += expanded
        states = list(leaves)
        leaf_values = np.asarray(self.leaf_value(model.observations(states)), dtype=float).reshape(-1) if states else np.empty(0)
        if leaf_values.shape != (len(states),) or not np.all(np.isfinite(leaf_values)) or np.any((leaf_values < 0) | (leaf_values > 1)):
            raise ValueError('Leaf must return one finite win probability per observation')
        leaf_map = dict(zip(states, leaf_values))
        q_values = []
        for action, outcomes, values, postorder in graphs:
            for key, branches in postorder:
                state, remaining = key
                values[key] = float(leaf_map[state]) if branches is None else max(
                    sum(p * values[child, remaining - 1] for p, child in successors) for _, successors in branches)
            q_values.append((action, sum(p * values[child, self.horizon - 1] for p, child in outcomes)))
        qs = dict(q_values)
        best = max(qs.values())
        proposed = next(a for a, q in q_values if q >= best - 1e-12)
        choice = proposed if qs[proposed] > qs[anchor] + self.deviation_margin else anchor
        self.last_diagnostics = GuidedDiagnostics(tuple(q_values), best, qs[choice], anchor, proposed,
                                                  expanded_total + 1, len(leaves), cutoffs,
                                                  perf_counter() - started, self.horizon)
        return choice
