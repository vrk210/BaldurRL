# Architecture

The simulator uses **semantic state and semantic actions**, not mouse coordinates or UI behavior, so a BG3 adapter can later serve policies through the same kind of interface. Keep the object graph shallow: characters store stats, resource counts, and known ability IDs; shared ability definitions live outside characters; mechanics remain functions.

## Maintained system inventory

This is the single inventory of tracked files. CI checks that every tracked
file except the agreed documents has exactly one row and that every row names a
tracked file. `supported` means an ongoing operational or research interface.
`verification-only` means a test, mock, or independent oracle; it is not
evidence of live BG3 behavior. The checker reads four purpose prefixes:
- `CLI:` on every Python file outside `tests/` with a `__main__` block;
- `fixture: tests/<reader>.py` or `fixture: scripts/<reader>.py`, naming the
  test or script that reads it, or
  `config:` on every committed `.json`, `.jsonl`, `.npz`, `.zip` or `.log` file;
- `private:<name>` for a recorded private-name import that a later PR removes.

| Path | Status | Entry point / responsibility |
| --- | --- | --- |
| `.gitignore` | supported | Keeps generated output, virtual environments and editor settings out of Git. |
| `.github/workflows/repository.yml` | supported | Inventory/lint checks and the full Windows test suite. |
| `.github/workflows/windows-harness.yml` | supported | Offline Windows harness verification; no game execution. |
| `agents/__init__.py` | supported | Public baseline-policy re-exports; no CLI. |
| `agents/expectimax_agent.py` | supported | Exact chance search with bounded turn/decision horizons. |
| `agents/heuristic_agent.py` | supported | Supported stage heuristics and threat-aware baseline. Cleanup debt until simplification: private:_new_fighter. |
| `agents/policy.py` | supported | Model-independent policy interface. |
| `agents/random_agent.py` | supported | Seeded random legal-action baseline. |
| `agents/rollout_agent.py` | verification-only | Sampling planner reference and reconstruction/isolation baseline; exposed by evaluation.evaluate. |
| `agents/tactical_heuristics.py` | supported | M5/M6 named heuristic variants. |
| `agents/value_functions.py` | supported | Heuristic/learned frontier values; no combat mechanics. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Config.json` | supported | config: Script Extender module settings. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/BootstrapClient.lua` | supported | Client diagnostic bootstrap; no action execution. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/BootstrapServer.lua` | supported | Server lifecycle and console entry points: !baldurrl_snapshot, !baldurrl_inspect; passive only. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/Emitter.lua` | supported | Session sequence allocation and immutable record staging. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/Inspect.lua` | supported | Read-only entity/component probe; diagnostic output stays outside records. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/Observation.lua` | supported | Best-effort passive BG3 facts; absent fields remain unknown. |
| `bg3/h0/Mods/BaldurRL_H0/ScriptExtender/Lua/Wire.lua` | supported | Versioned JSON wire representation and ready-marker publication. |
| `bg3/h0/Mods/BaldurRL_H0/meta.lsx` | supported | config: BG3 module identity. |
| `combat/__init__.py` | supported | Package marker; no entry point. |
| `combat/abilities.py` | supported | Immutable ability costs and target kinds. |
| `combat/actions.py` | supported | Semantic action identities; no execution. |
| `combat/actors.py` | supported | Stable actor addresses and roster sides. |
| `combat/characters.py` | supported | Character state and presets; no turn orchestration. |
| `combat/damage.py` | supported | Immutable damage specifications. |
| `combat/distributions.py` | supported | Exact outcome distributions using mechanics primitives. |
| `combat/env.py` | supported | M0 Gymnasium coordinator and stable observation/action contract. |
| `combat/legality.py` | supported | Candidate decisions and per-actor target/resource legality. |
| `combat/mechanics.py` | supported | Shared attack, damage, healing and resource effects. |
| `combat/resources.py` | supported | Nonnegative resource pools and atomic affordability/spending. |
| `combat/rolls.py` | supported | Advantage/disadvantage roll mechanics. |
| `combat/simulation.py` | supported | Validated reconstruction without access to live RNG. Cleanup debt until simplification: private:_enemy_profile. |
| `combat/stages.py` | supported | M1A–M4 coordinators and M0–M6 factory/metadata. Cleanup debt until simplification: private:_advance_to_next_turn, private:_new_fighter. |
| `combat/table_tactical_transitions.py` | supported | Default M5/M6 exact planner dynamics; checked against replay. Cleanup debt until simplification: private:_dice_sum_distribution. |
| `combat/tactical_env.py` | supported | M5/M6 sampling, turns, observations and episode outcomes. Cleanup debt until simplification: private:_new_fighter. |
| `combat/tactical_transitions.py` | supported | Validated observation decoding used by the production table model; independent replay dynamics retained for verification. |
| `combat/tactics.py` | supported | M5/M6 conditions, reach, enemy policies and turn lifecycle. |
| `combat/transitions.py` | supported | Planner protocol, registry and M0–M4 exact dynamics. |
| `combat/turns.py` | supported | Deterministic turn order, dead-actor skipping and rounds. |
| `evaluation/__init__.py` | supported | Existing evaluation API re-exports; no CLI. |
| `evaluation/compare.py` | supported | CLI: python -m evaluation.compare; outcome/trace comparison. |
| `evaluation/evaluate.py` | supported | CLI: python -m evaluation.evaluate; seeded episodes, run cards and traces. |
| `evaluation/paired.py` | supported | CLI: python -m evaluation.paired; paired run-card statistics. |
| `evaluation/planning_eval.py` | supported | CLI: python -m evaluation.planning_eval; parallel planner evaluation. |
| `evaluation/seeds.py` | supported | CLI: python -m evaluation.seeds; seed reservations and frozen-set enforcement. |
| `evaluation/tactical_behavior.py` | supported | CLI: python -m evaluation.tactical_behavior; trace behavior summaries. |
| `evaluation/tactical_transition_audit.py` | verification-only | CLI: python -m evaluation.tactical_transition_audit; adversarial oracle comparison. |
| `evaluation/tactical_transition_benchmark.py` | verification-only | CLI: python -m evaluation.tactical_transition_benchmark; exactness and timing. |
| `integration/__init__.py` | supported | Package marker; no entry point. |
| `integration/command_transport.py` | supported | Atomic session command publication, freshness and feedback validation. |
| `integration/commands.py` | supported | Model-independent commands, targets, legality evidence and result states. |
| `integration/encounter_trial.py` | supported | CLI: python -m integration.encounter_trial; evidence analysis; missing live facts remain inconclusive. |
| `integration/inspect.py` | supported | CLI: python -m integration.inspect; passive record inspection. |
| `integration/mock_executor.py` | verification-only | CLI: python -m integration.mock_executor; synthetic command acknowledgements. Cleanup debt until simplification: private:_publish, private:_read. |
| `integration/mock_h0.py` | verification-only | CLI: python -m integration.mock_h0; synthetic passive collector. |
| `integration/publish.py` | supported | CLI: python -m integration.publish; ready-marker record publication. |
| `integration/reader.py` | supported | Ordered passive-record reader; preserve compatibility. |
| `integration/schema.py` | supported | Versioned passive GameSnapshot/GameEvent records; preserve compatibility. |
| `integration/serialization.py` | supported | Strict passive-record JSON decoding/encoding. |
| `integration/session_runner.py` | supported | CLI: python -m integration.session_runner; isolated capture archival and health. |
| `integration/trace.py` | supported | Candidate decisions from passive observations; UNKNOWN stays unknown. |
| `integration/windows_preflight.py` | supported | CLI: python -m integration.windows_preflight; read-only installation checks. |
| `pyproject.toml` | supported | Pinned-tool lint configuration; no packaging framework. |
| `pytest.ini` | supported | Puts the repository root on the test import path. |
| `requirements.txt` | supported | Pinned Python 3.11 simulator/development dependency graph; includes harness and quality manifests. |
| `requirements-harness-test.txt` | supported | Pinned minimal dependencies for harness capture and its offline tests. |
| `requirements-quality.txt` | supported | Pinned lint tool used by repository CI. |
| `scripts/check_repository.py` | supported | CLI: python scripts/check_repository.py; inventory, docs and artifact checks. |
| `scripts/windows/bootstrap-harness.ps1` | supported | CLI: scripts/windows/bootstrap-harness.ps1; isolated harness environment. |
| `scripts/windows/package-h0.ps1` | supported | CLI: scripts/windows/package-h0.ps1; Divine packaging and source manifest. |
| `tests/m2_oracle.py` | verification-only | pytest verification: m2 oracle. |
| `tests/test_bg3_h0_lua.py` | verification-only | pytest verification: bg3 h0 lua. |
| `tests/test_combat_regression.py` | verification-only | pytest verification: combat regression. |
| `tests/test_encounter_trial.py` | verification-only | pytest verification: encounter trial. |
| `tests/test_env.py` | verification-only | pytest verification: env. |
| `tests/test_evaluation.py` | verification-only | pytest verification: evaluation. |
| `tests/test_expectimax_agent.py` | verification-only | pytest verification: expectimax agent. |
| `tests/test_heuristic.py` | verification-only | pytest verification: heuristic. |
| `tests/test_integration_command_review_regressions.py` | verification-only | pytest verification: integration command review regressions. |
| `tests/test_integration_command_transport.py` | verification-only | pytest verification: integration command transport. |
| `tests/test_integration_commands.py` | verification-only | pytest verification: integration commands. |
| `tests/test_integration_publish.py` | verification-only | pytest verification: integration publish. |
| `tests/test_integration_reader.py` | verification-only | pytest verification: integration reader. |
| `tests/test_integration_schema.py` | verification-only | pytest verification: integration schema. |
| `tests/test_integration_serialization.py` | verification-only | pytest verification: integration serialization. |
| `tests/test_integration_trace.py` | verification-only | pytest verification: integration trace. |
| `tests/test_legality.py` | verification-only | pytest verification: legality. |
| `tests/test_m3.py` | verification-only | pytest verification: m3. |
| `tests/test_m4.py` | verification-only | pytest verification: m4. |
| `tests/test_m5_m6.py` | verification-only | pytest verification: m5 m6. |
| `tests/test_mechanics.py` | verification-only | pytest verification: mechanics. |
| `tests/test_paired.py` | verification-only | pytest verification: paired. |
| `tests/test_repository_policy.py` | verification-only | pytest verification: repository policy. |
| `tests/test_rollout_agent.py` | verification-only | pytest verification: sampling planner and independent continuation isolation. |
| `tests/test_seeds.py` | verification-only | pytest verification: seeds. |
| `tests/test_session_runner.py` | verification-only | pytest verification: session runner. |
| `tests/test_simulation.py` | verification-only | pytest verification: simulation. |
| `tests/test_stage_evaluation.py` | verification-only | pytest verification: stage evaluation. |
| `tests/test_stages.py` | verification-only | pytest verification: stages. |
| `tests/test_table_tactical_transitions.py` | verification-only | pytest verification: table tactical transitions. |
| `tests/test_tactical_heuristics.py` | verification-only | pytest verification: tactical heuristics. |
| `tests/test_tactical_simulation.py` | verification-only | pytest verification: tactical simulation. |
| `tests/test_tactical_transitions.py` | verification-only | pytest verification: tactical transitions. |
| `tests/test_tactics.py` | verification-only | pytest verification: tactics. |
| `tests/test_training.py` | verification-only | pytest verification: training. |
| `tests/test_transitions.py` | verification-only | pytest verification: transitions. |
| `tests/test_turns.py` | verification-only | pytest verification: turns. |
| `tests/test_value_functions.py` | verification-only | pytest verification: value functions. |
| `tests/test_windows_preflight.py` | verification-only | pytest verification: windows preflight. |
| `tests/transition_harness.py` | verification-only | pytest verification: transition harness. |
| `train.py` | supported | CLI: python train.py; masked PPO training; no combat rules. |
| `value_training.py` | supported | CLI: python value_training.py; collect/fit model-independent planner values. |

## Stage progression

M0 is the fixed 1v1 regression baseline. M1A–M4 add randomized opponents, target
choice, Cleave and two controlled allies. M5/M6 add roles, conditions and an
abstract two-rank positioning model. These are simulator choices, not verified
BG3 rules. Behavioral contracts live beside their implementations in
`combat/env.py`, `combat/stages.py`, `combat/tactical_env.py` and `combat/tactics.py`.
Action/observation metadata defines stable indices; changing it requires explicit
compatibility work. Tests exercise behavior and independent reconstruction.

The simulator is frozen at M0–M6. It remains a regression/research test bed and a
stand-in game for integration protocols. Its trained policies do not directly
transfer to BG3: their vectors/action indices are simulator-specific. Revisit
simulator scope only when logged BG3 encounters identify missing mechanics that
matter. Findings and uncertainty live in [RESULTS.md](RESULTS.md).

## Dependency direction and verification

`actions/resources/damage → abilities/characters → mechanics → environments → policies/evaluation`

Characters store state and refresh mappings; immutable ability catalogs describe
costs and targets. Mechanics resolve effects using supplied RNGs. Environments
own turns, legality gating, rewards and termination. Policies only select actions.
Legality is turn-agnostic; environments expose masks only for controlled turns.
Models reconstruct observations without reading the live simulator's RNG.

M0–M4 exact transitions share rule/distribution primitives with sampling. M5/M6
table transitions are the fast default and retain the independent environment
replay oracle for verification. Oracle agreement detects implementation drift,
not an incorrect shared rule or a mismatch with BG3. The transition audit and
benchmark remain verification tools; their commands are in module --help.

Expectimax depends on the TransitionModel protocol and a pluggable frontier
value. It contains no combat rules. The sampling planner remains a verification
reference for independent continuation/reconstruction; completed experiment
bundles are retained through tags cited in RESULTS.md.

## BG3 integration boundary

`BG3 collector → snapshots/events → candidate decisions → semantic command → executor feedback`

The integration package is independent of combat, Gymnasium and ML models.
Snapshots preserve original game IDs and unknown facts; a future adapter chooses
policy features. Commands require current evidence, and acceptance is distinct
from execution and post-execution observation. The passive H0 collector and
Windows capture tools exist with offline tests. No real action executor or
BG3-backed policy environment is implemented; game loading and IO require live
verification. The [integration contract](BG3_INTEGRATION.md) owns wire/recovery
semantics; [the H0 runbook](../bg3/h0/README.md#first-windows-session) is the sole
Windows installation/capture/verification procedure.
