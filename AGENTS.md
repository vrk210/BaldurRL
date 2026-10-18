# BaldurRL: rules for agents

The goal is an agent that plays Baldur's Gate 3. This file says how we work.
[ARCHITECTURE.md](docs/ARCHITECTURE.md) says what the system is now. If the two
disagree, stop and ask the owner.

## How work starts

Work has one of two motives. Don't write code without one.

- **Outcome work.** The owner confirms the outcome before code is written:
  - what must be true afterwards;
  - why it matters;
  - how we will know;
  - what the code looks like when it's done.

  When a request is underspecified, propose the outcome and get it confirmed.
- **Exploration.** Exploration is just as important. It starts from a motive
  or hypothesis, even an unlikely one, and runs on an `explore/<name>` branch.
  It ends when the hypothesis is answered or abandoned, and either way it ends
  with a results entry (see *Evidence*). Its code doesn't merge. Tag the branch
  `archive/<name>`, then delete it.

## Principles

1. **Optimize for the end state.** Judge a change by the codebase it leaves,
   not by its diff.
   - Subtract before adding: delete what the requirement makes unnecessary.
   - Prefer the smallest result, counted in concepts a reader must hold, not
     in lines.
   - If the requirement had been known from the start and the codebase would
     then be smaller or clearer, redesign rather than bolt it on. Carry the
     change through every reference: types, docs, comments.
   - Redesign only for confirmed outcomes, never for exploration. Count the
     cost of re-verifying what the redesign touches.
   - Deliver a redesign in steps, but finish every migration. Never leave old
     and new paths side by side.
2. **Every line serves a stated purpose.** Each file, class and function states
   what it is for, through its name and signature and, if those aren't enough,
   a docstring. Every line inside it serves that purpose. Nothing is there just
   in case or left over. This applies to tests, comments, flags and fixtures
   too.
3. **Be concrete.** Name the file, command, seed range, number and unit.
4. **Unknown stays unknown.** Never fill in a fact about the game from D&D
   rules, a simulator or a guess. Simulator results are hypotheses about the
   game.
5. **Test in practice.** Check how something behaves when actually used, and
   check it before the PR, not after.

## Where things live

Each fact has one home. Read it before changing what it covers.

- `README.md`: what the project is, setup, and the commands that run it.
- `docs/ARCHITECTURE.md`: the system as it is now: module boundaries and
  dependency direction, the file inventory, and simulator-wide decisions with
  their reasons and revisit conditions. Read it before changing code. Its
  decisions bind you as much as this file; if one seems wrong, raise it with
  the owner instead of working around it. The owner approves every inventory
  status change.
- `docs/BG3_INTEGRATION.md`: the contracts between the game and the code
  (records, commands, transport, recovery) and the decisions about them.
- `docs/GAME_KNOWLEDGE.md`: facts and hypotheses about the game, each with its
  evidence label.
- `docs/RESULTS.md`: measured results, with their limits and open questions.
- `bg3/h0/README.md`: the Windows procedure for installing, capturing with and
  verifying the in-game collector.
- A module's docstring: that module's contract.
- A decision comment: a decision that constrains only the code beside it.
- PR bodies and Git history: how the code got here.

## Writing code

Signatures explain themselves. The name says the action or result, the
parameter names and types say what goes in, and the return type says what
comes out.

```python
def simulation_from_observation(stage: str, observation: np.ndarray, *, seed: int) -> TacticalCombatEnv: ...  # clear
def run_chunk(stage: str, spec: dict, seeds: list[int]) -> list[dict]: ...  # what is in spec and in the rows?
```

- Name the mechanism, not the hoped-for result: `GatedRolloutAgent`, not
  `ImprovedHeuristicAgent`.
- Put units in names when the type doesn't carry them: `planning_seconds`,
  `delta_pp`.
- Use one name per concept across the repo.
- Make booleans and same-type neighbours keyword-only.
- Return dataclasses, not dicts, when callers use the fields.
- **Docstrings** say why, not what happened. Include only the parts that apply:
  - the responsibility and the outcome it serves;
  - the choice a reader would question, and why it was made;
  - how it is checked;
  - when to revisit.

  One line is fine. Results tables, timings, commit hashes and the history of
  the work belong in RESULTS.md or the PR, not in code.
- **Decision comments** have a date and a reason to revisit:
  `# Decision (YYYY-MM-DD): ... Revisit if ...`. When the decision changes,
  rewrite the comment; Git keeps the old one.
- **TODOs** name a checkable condition: `# TODO(first live capture): ...`.
- **Game facts:** where code relies on one, use the evidence labels from
  GAME_KNOWLEDGE.md.
- **Validation** happens at trust boundaries: CLI input, records from the game,
  and observations from outside the process. Trust internal callers.
- **Errors:** messages name the bad value and the rule it broke. Never
  silently catch programming errors; catch broad exceptions only at process
  boundaries, and record the failure there.
- **Imports and output:** production code doesn't import `_private` names
  from other modules; CI checks this outside `tests/`, apart from imports the
  inventory records as `private:` debt. Library code doesn't print.
- **Determinism:** randomness comes from a caller-supplied
  `numpy.random.Generator`.
- **Paths** come from arguments and default to `runs/`.
- **Dependencies** are pinned.
- **Abstract only for:**
  - a boundary that already has two or more implementations, like
    `TransitionModel` or `Policy`;
  - logic that must stay in sync showing up a second time;
  - a domain concept that otherwise travels as loose values, like `ActorRef`.

  Never abstract for callers that don't exist yet.

## Tests

Test behavior in practice, not implementation:

- **Integration:** replay captured game sessions through the pipeline. Real
  captures become the fixtures, trimmed to the records the test reads.
- **Policies:** play them, and compare policies on matched encounters.
- **Unit tests:** only for logic with more cases than practice reaches, such as
  exact transition enumeration. Use an independent oracle where one exists.
- **Bugs:** each real bug gets one regression test, named after the behavior.

For each behavior, keep the fewest tests that would catch a plausible
regression. Don't test structure, mocks, constants or argument wiring. A
behavior-preserving refactor leaves the tests unchanged.

## Evidence

**Explain differences; don't just measure them.**
1. Find where two policies diverge.
2. Form a hypothesis about which decision causes the gap.
3. Change only that decision.
4. Measure the effect.

Size each run from a pilot's noise, so you can see the smallest effect that
would change the decision. Before running, write down what an inconclusive
result means. Check the method by running a policy against itself: it should
find no difference. Select policies and configurations by play, not by
proxies such as validation loss. Reserve encounter seeds in
`evaluation/seeds.py` before running; a PR that only reserves seeds skips
review. Never use held-out final-test ranges for development.

| What | Where |
| --- | --- |
| Raw output, traces, result JSON, trained models | `runs/` (git-ignored); never committed |
| Live game captures | An archive root outside the repo, with location and retention recorded; deleted only by the owner |
| Experiment results | `docs/RESULTS.md` |
| Hypotheses about the game | `docs/GAME_KNOWLEDGE.md`, labeled unverified |

**Results.** Every exploration and experiment gets one entry in RESULTS.md,
with:
- the motive or hypothesis;
- what was run (tag, commit, seeds);
- the outcome;
- what we learned;
- irregularities;
- open leads.

For other outcome work, the PR is the record. Entries carry no narrative.

## Documentation

- Adding a document is the owner's decision.
- This file holds only how we work and where things live. Everything else
  goes in its home under *Where things live*.
- Update documents in the same PR as the code.
- State each fact and caveat once, in its home, and link to it elsewhere.
- Describe only the current system.

## Review, then PR

The PR is the final step. Live testing and review happen before it.

**Live testing.** A change that touches the game is tested in the game first.
Code that can't be tested in the game yet stays on its branch, unless that test
needs it (setup, capture). A setup change is shown working in a fresh
environment. Never report a synthetic or offline check as live verification.

**Review.** Fresh reviewer agents get the outcome, the diff and the repository,
but not the author's conversation. Review should take more effort than writing
the code. One reviewer per focus, in priority order:
1. **Outcome.** Is it met, and does every part of the diff serve it?
2. **Correctness.** Break it with inputs inside the contract. Rerun every cited
   command.
3. **Alternatives.** Is there a materially different approach that leaves a
   smaller codebase?
4. **Tests.** Mutate the code and look for a stronger test.
5. **Performance.** Measure where cost matters.
6. **Readability** (weighed least).

A reviewer may mark its focus "not applicable", with a one-sentence reason.

**Findings.** A finding needs a reproduction: the failing input, the surviving
mutation, the measurement or the alternative's result. Without one it is a
question.
- **Blocking:** only a reproduced failure to meet the outcome, or a reproduced
  contract violation.
- **Advisory:** everything else. The author fixes it or declines it in one
  line.
- **To the owner:** doubts about the outcome itself, and better alternatives.

After fixes, rerun only the reviewers whose focus the fix touched. Review ends
when a round finds no new blocking finding. A disagreement that comes up twice
goes to the owner. Before an exploration's conclusion changes a decision, its
analysis gets the correctness review.

**PR.**
- **Scope:** one outcome per PR. Split off anything that could merge alone for
  a separate outcome.
- **Deletions** that the change makes unnecessary go in the same PR. Unrelated
  cleanup gets its own PR.
- **Contents:** no experiment code, and nothing without a current caller.
- **Body**, in four parts:
  - **Outcome:** what is true after this merges.
  - **Why it matters:** the goal or decision it serves.
  - **Evidence:** live or practice results with exact commands, and the review
    findings with their resolutions.
  - **Not done:** what remains or didn't work.
- **Checks:** before opening it, run what CI runs:
  `python scripts/check_repository.py`, `python -m ruff check .` and
  `python -m pytest -q`.
- **Merging and branches:** the owner merges, or an agent the owner explicitly
  asks to. Agents never force-push. They don't bypass branch rules or push to
  another agent's branch without the owner's agreement.
