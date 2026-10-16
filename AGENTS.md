# BaldurRL: working rules

## Maintained system

Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module/entry-point inventory
and dependency boundaries before editing code. Its statuses are authoritative:
`supported`, `verification-only`, or `remove`. Update the inventory in the same
PR when adding, retiring, or changing the responsibility of a module. Do not
maintain a second module table here.

The simulator is frozen at M0–M6 as a regression and research test bed. Do not
add invented stages or run open-ended policy sweeps. Changes to simulator rules
require an explicit task; BG3 facts must come from evidence, not inferred D&D
rules. The integration package is independent of simulator/ML representations.
It preserves BG3 identifiers and unknown facts. Command acceptance never proves
execution. No real BG3 action executor is implemented or authorized by cleanup.

Simulator contracts live in the owning modules' docstrings and executable
metadata; behavior tests verify them. Report a contract/implementation
disagreement before changing behavior.

## Where code, documentation, and results live

- Put maintained implementation in the module owning its responsibility in the
  inventory. Use typed, small functions/dataclasses and explicit public APIs.
  Agents choose actions; environments coordinate; mechanics implement rules.
  Never put combat rules in policies or mutate character state from an agent.
- Put local rationale, assumptions, and conditions for revisiting a decision
  beside code. Docstrings describe contracts and design choices, not experiment
  histories, result tables, prompt changes, or machine-specific provenance.
- Maintain only README.md, AGENTS.md, docs/ARCHITECTURE.md,
  docs/BG3_INTEGRATION.md, docs/GAME_KNOWLEDGE.md, docs/RESULTS.md, and
  bg3/h0/README.md. Existing documents marked `remove` in the inventory remain
  only until their specified migration. New documents require an explicit scope
  decision. Temporary handoffs stay outside the tracked tree and state a removal
  condition. Put completed plans and chronology in PR/Git history.
- Put measured experiment findings in docs/RESULTS.md: question, configuration,
  code revision, seeds/sample size, uncertainty, limitations and open questions.
  GAME_KNOWLEDGE.md holds the resulting game hypotheses with evidence limits.
  Small executable fixtures and operational limits belong beside their consumers.

## Experiment lifecycle

1. Define the question and stopping/selection criteria. Reserve encounter seeds
   in evaluation/seeds.py before running; keep training, selection and final-test
   ranges separate. Never use frozen final-test seeds for development.
2. Write raw logs, transcripts, research bundles and result JSON under ignored
   runs/, once per experiment. Do not commit generated experiment output or
   trained models. Small fixtures are allowed only with a consuming test named
   in the inventory; configuration JSON must be identified as configuration.
3. Inspect outputs before summarizing/removing them: missing/non-finite data,
   configuration/provenance inconsistencies, failures and unexpected/promising
   results. Alert the user to findings needing investigation. Preserve the
   smallest relevant evidence while a question is unresolved.
4. Summarize findings in RESULTS.md. Before deleting completed research code or
   committed evidence, create and push an annotated tag and cite it in the
   summary. Retire regenerable output when no active investigation needs it.
   Unique live captures and active training data need a recorded location and
   retention/removal condition; do not discard irreplaceable data routinely.
5. Merge experiment code only for a concrete ongoing purpose recorded in the
   inventory. Completed one-time research leaves a summary and tagged history.

## PR scope and checks

One PR has one purpose stated in one sentence. Do not merge another agent's
branch without explicitly agreed dependency/scope coordination. No unrelated
features, behavior changes or formatting sweeps in cleanup PRs.

The default budget is 1,000 added plus deleted non-test lines. Each PR states
`Size budget: N` in its body. A different budget requires an explicit scope
agreement recorded as `Budget reason: ...`. Retirement PRs may instead state
`Size budget: 300 additions` and count deletions separately; they permit source
and test deletions only, plus evidence-summary/inventory updates. Tests are
counted separately and must be justified by behavior, not implementation shape.

- Run `python scripts/check_repository.py` and the relevant pytest tests.
- Run `python -m ruff check .` with requirements-quality.txt installed. CI checks
  inventory coverage, document links/locations, artifact rules, PR size, syntax,
  undefined names, and one statement per line. It does not judge design quality.
- No new private-name imports across implementation modules. Existing imports
  explicitly marked as cleanup debt in the inventory must be removed in the
  simplification PR. Use caller-supplied
  seeded RNGs for mechanics; no global randomness or hidden live RNG access.
- Behavior-preserving refactors state `Behavior: preserve` in the PR body so CI
  compares fixed-seed semantic fingerprints against the base commit. They may
  rewrite or delete tests that checked implementation, with the reason in the PR. Exclude timings and
  filesystem locations. Keep independent verification that detects distinct
  errors. Do not change schemas, action/observation ordering, reward semantics,
  simulator mechanics or trained models during cleanup.
- Dependency/setup changes must demonstrate installation and documented checks
  in a clean environment. BG3 loading/execution requires separate live evidence;
  synthetic or offline checks must never be reported as live verification.
