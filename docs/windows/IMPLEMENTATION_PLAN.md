# Windows preparation implementation plan

Base: main at 6047eb923c5f58bc21bcc1df5b480bcf07dfe660. The harness code is identical on the current integration PR; this work does not depend on merging simulator/planner PR #9.

Owner: primary agent handles research, architecture, review, integration, and publication. Three implementation agents own disjoint modules, tests, scripts, and guides. They must read AGENTS.md and their spec before editing. No game execution, game-state mutation, dependency downloads, registry edits, game installation changes, policy training, or simulator changes are authorized by these tasks. Do not spawn further agents.

## Boundaries

- Preserve GameSnapshot/GameEvent v1, parse_record, and RecordDirectoryReader behavior.
- Reuse the existing staged completion publisher and command protocol. Do not build another transport.
- UNKNOWN stays unknown. No fabricated ability/resource IDs or simulator defaults in production.
- Static file presence and successful offline tests do not prove BG3 loading or behavior.
- No automatic command retries or exactly-once claims. No real input injection in this preparation phase.
- Runtime implementation should use the Python standard library. Optional Lua test runtime is separate from installation prerequisites.
- Each agent edits only its owned paths. Shared README, BG3_INTEGRATION, and final runbook edits belong to the primary agent.

## Deliverables

1. BOOTSTRAP_SPEC.md -> integration/windows_preflight.py, scripts/windows/*bootstrap/package*, test dependencies, tests and BOOTSTRAP.md.
2. SESSION_SPEC.md -> integration/session_runner.py, read-only probe snippets, tests and SESSIONS.md.
3. ENCOUNTER_SPEC.md -> integration/encounter_trial.py, templates, tests and ENCOUNTERS.md.

Each implementation must include CLI examples, deterministic offline tests, machine-readable reports, and explicit remaining Windows/live checks. Do not claim the resulting tools form an autonomous combat executor.

## Integration acceptance

Run new targeted tests and the full main suite. Exercise a synthetic staged session through publication, archival, health reporting and encounter analysis. Review fail-closed behavior, malformed input handling, source/destination isolation, provenance, and Windows path handling. Produce a single focused reviewable branch/PR with individual commits per implementation track where practical. Preserve pre-existing worktrees and untracked files.
