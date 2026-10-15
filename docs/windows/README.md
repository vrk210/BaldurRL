# Windows handoff and first encounter runbook

This branch prepares the Windows installation, passive capture, and bounded encounter analysis. It contains no real BG3 action executor. It preserves observation v1 and uses the existing command protocol. Source presence, static readiness, schema-valid records, and a passing synthetic trial are separate evidence levels; none establishes autonomous combat readiness.

## Work ownership and specifications

The primary agent owns research, design decisions, integration, and review. Three GPT-5.6 implementation agents work on disjoint paths from the specs below:

- [Bootstrap specification](BOOTSTRAP_SPEC.md): static doctor, explicit PowerShell bootstrap and Divine packaging, optional test dependencies.
- [Session specification](SESSION_SPEC.md): multi-session publication, provenance, archival, health and read-only probes.
- [Encounter specification](ENCOUNTER_SPEC.md): versioned trial templates and conservative offline analysis.

[Implementation plan](IMPLEMENTATION_PLAN.md) defines shared boundaries. [RL next steps](RL_NEXT_STEPS.md) records the separate, bounded simulator research sequence; it launches no training.

## Preparation before the Windows host arrives

Read [BOOTSTRAP.md](BOOTSTRAP.md), [SESSIONS.md](SESSIONS.md), and [ENCOUNTERS.md](ENCOUNTERS.md). Keep BG3SE and Divine installs explicit and record their actual versions. The Python runtime tools use the standard library; optional pytest/Lupa dependencies are for offline tests. Package the existing H0 source with Mods at the package root. Do not assume a clean clone contains trained models or git-ignored runs/ datasets: transfer selected artifacts separately with field/stage metadata and checksums if needed.

Use the original [H0 installation guide](../../bg3/h0/README.md) for activation and the actual collector path. The doctor checks file/configuration evidence; live bootstrap output and callbacks must still be verified. A not-yet-created capture namespace before the first game launch is an expected missing preflight prerequisite, not proof that a path is wrong.

Windows offline CI covers Python 3.11/3.13, real Lua with stubbed game APIs, PowerShell parsing, and packaging behavior with a fake converter. These checks do not replace a real Divine package or a loaded BG3 encounter.

## First Windows session

1. Install and activate H0, launch a disposable single-player save, and record game/SE versions. Confirm the H0 bootstrap, session GUID, listener registrations, and actual writable IO root. Keep logging enabled using documented settings. Investigate UNAVAILABLE/HALTED output.
2. Run the session runner against the explicit parent of GUID session directories. Use an archive location outside that source tree. Keep one publisher/runner per session; do not run integration.publish separately against a session already owned by the runner.
3. Play one simple in-range 1v1 fight by hand. Capture snapshots before and after a basic attack, including a miss if observed, and end the turn normally. Reload/reset to verify new session discovery and no mixed namespaces. Compare selected actor, target IDs, HP, and event order to the UI. Preserve incomplete facts as unknown.
4. Run the read-only actor probe from [bg3/probes](../../bg3/probes/README.md) for the actual selected combatant. Inspect component names first; discover raw ability/resource identifiers and levels. Do not substitute simulator IDs or fill unavailable facts with presets.
5. Fill a human_attack trial with actual observed IDs, explicit snapshot boundaries, measured resource expectations, and the operator's completion evidence. Analyze the trace. INCONCLUSIVE means required evidence is missing, not that the game failed.

At this point success is a reproducible, inspectable human encounter capture. That is enough to begin implementing a verified observation subset and to identify precise live incompatibilities.

## Execution gate after passive capture works

The next implementation is a narrow real executor plus legal-action evidence. Follow [the execution research and trial plan](../BG3_ACTION_EXECUTION.md): reproduce ordinary UI input, measure a human control, attempt one autonomous action, compare identity/costs, then test rejection when resources are exhausted. Existing read-only probes never inject keys or mutate queues.

- H0 currently emits no owned-ability/resource/candidate enumeration. Ability commands correctly remain blocked until an exact verified LEGAL decision is available.
- Shared/overlapping turns can leave active_entity_id unknown; current command checks then block. Preserve that behavior until a representation for eligible actors is designed from live observations.
- Position targets are ability targets, not walking commands. M6 ADVANCE does not specify BG3 navigation.
- Acceptance is not completion. Confirm stopped execution, then initiate a new capture and publish PostExecutionObservation. An elapsed timeout or a newer sequence alone does not establish timing.
- Unknown outcomes stop the loop. Do not replay commands automatically across crashes or delete histories to clear an uncertain slot.

Only after one attack's identity, normal resource costs, invalid-action rejection, and fresh observation boundary are established should the loop add end turn and repeat through an entire 1v1 encounter. Expand party size, targeting and movement separately. The first controller can be deterministic; no trained policy is needed to validate this interface.
