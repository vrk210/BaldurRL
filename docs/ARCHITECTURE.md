# Architecture

The simulator uses **semantic state and semantic actions**, not mouse coordinates or UI behavior, so a BG3 adapter can later serve policies through the same kind of interface. Keep the object graph shallow: characters store stats, resource counts, and known ability IDs; shared ability definitions live outside characters; mechanics remain functions.

## Maintained system inventory

This is the single inventory of tracked files. CI checks that every tracked
file except the agreed documents has exactly one row and that every row names a
tracked file. `supported` means an ongoing operational or research interface.
`verification-only` means a test, mock, or independent oracle; it is not
evidence of live BG3 behavior. `remove` means retirement is decided, with the
prerequisite stated below; the checker skips the agreed-document, prefix and
import rules for those files until they are gone. The checker reads four purpose prefixes:
- `CLI:` on every Python file outside `tests/` with a `__main__` block;
- `fixture: tests/<reader>.py` or `fixture: scripts/<reader>.py`, naming the
  test or script that reads it, or
  `config:` on every committed `.json`, `.jsonl`, `.npz`, `.zip` or `.log` file;
- `private:<name>` for a recorded private-name import that a later PR removes.

| Path | Status | Entry point / responsibility |
| --- | --- | --- |
| `.gitignore` | supported | Keeps generated output, virtual environments and editor settings out of Git. |
| `.github/workflows/repository.yml` | supported | Inventory/lint checks. |
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
| `docs/BG3_ACTION_EXECUTION.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
| `docs/COMBAT_TURN_MODEL.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
| `docs/EXPECTIMAX_PLANNING.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
| `docs/M0_SPEC.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
| `docs/ROLLOUT_PLANNING.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
| `docs/STAGES_SPEC.md` | remove | Relocate durable contracts/references to maintained code/docs, then delete in docs PR. |
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

M0 remains the fixed 1v1 regression environment specified in `M0_SPEC.md`. The later stages are specified in `STAGES_SPEC.md`: M1A varies one opponent's stats; M1B has two fixed melee enemy slots and explicit target decisions; M2 adds one-use Cleave; M3 controls two allies against two enemies with an active-actor observation; M4 extends that to three scaled enemies. M5 and M6 (`combat/tactical_env.py`) are decision-rich stages: role-typed enemies (Brute, Archer, Healer) with observed targeting policies, advantage/disadvantage, ally Trip/Prone and Dodge, and in M6 a two-rank front/back abstraction with Advance, Disengage, and opportunity attacks. Their rules live in `combat/tactics.py`; per-stage tuning data lives in `STAGE_RULES`. Stage-specific character state (`TacticalFighter`, `TacticalEnemy`) adds conditions, positions, and last-attacker memory to `Character`; all of it is observed, so `simulation_from_observation` restores M5/M6 exactly. A small stage factory selects an environment for training and evaluation. Each environment publishes action labels and observation field names in index order so traces and run cards do not encode M0 assumptions.

Gym action indices represent decisions. A decision carries a semantic `Action` and, for a targeted attack, an enemy slot index. Target selection is separate from the shared ability identity. The environment owns target legality, turn order, outcomes, and reward selection; mechanics still own attack and ability effects. Character state holds resources and known abilities. Agents receive only the flat observation and action mask and return an index.

All three later stages use terminal win/loss reward for their main comparison. M2 also supports an isolated bounded damage-shaping training experiment. External evaluation always measures combat outcomes and behavior on seeded episodes, independent of the training reward scheme.

**Simulator role (decided 2026-10-15).** No new abstract stages are planned. M5 and M6 already separate good policies from weak ones. Hand heuristics win about 79–81%. MaskablePPO with default settings plateaus 4–5 points below them, and on M2 it comes within about 2 points of the exact optimum. More invented difficulty would not move the project toward BG3. The simulator stays as a regression and research test bed and as a stand-in game behind the integration protocols. Trained simulator policies do not transfer, because their observation vectors and action indices don't exist in BG3. Revisit this once logged BG3 fights show which mechanics drive outcomes; the next stage should then be calibrated to those logs, not invented. Measured results are in [RESULTS.md](RESULTS.md).

## Dependency direction

```text
actions + resources → abilities
actions + resources + damage → characters
abilities + characters → mechanics → environment → agents/evaluation
```

Avoid circular dependencies. Characters must not import ability execution or the RL environment. Mechanics must not import trained agents. Agents must not implement combat formulas. The environment must call mechanics rather than duplicate their rules. Generic resource affordability uses `ResourcePool.has`; living-target and missing-HP rules stay with the corresponding mechanics. `can_use_ability` checks known IDs and costs, not full effect-specific legality. `END_TURN` stays in environment turn control.

For M0, `BaldurCombatEnv.step()` handles exactly one Fighter decision. It dispatches mechanics and returns immediately for Fighter abilities; `END_TURN` alone triggers the automatic Goblin attack and round advance. Legal-action masks stay separate from the seven-field `float32` Box observation. The action space is an explicit four-index Discrete space. Rewards use an injectable callable receiving an immutable before snapshot, the selected semantic `Action`, an immutable after snapshot, and termination/truncation flags; the default is terminal-only. The environment alone owns episode completion and the round-50 truncation limit.

`RandomAgent` is the first baseline policy. It chooses uniformly from the legal-action mask using its own seeded NumPy generator; the environment's separate seeded generator handles combat rolls. The evaluation harness runs policies over explicit environment episode seeds (by default `0..9999`) and reports win rate as the primary baseline metric, along with outcomes, final HP, rounds, and action usage. Run it with `python -m evaluation.evaluate`.

`RolloutAgent` evaluates every legal first action by following independent fixed
continuation policies through terminal-reward simulators restored from the
observation. Combat code owns reconstruction; the agent only chooses and scores
actions through `step()`. A small `agents.policy.Policy` structural protocol is
shared with evaluation. See [rollout planning](ROLLOUT_PLANNING.md) for factory
isolation, budgets, diagnostics, and CLI usage.

For decision-level inspection, evaluation can write one JSONL record per episode. Each record contains the outcome and every Fighter decision with the observation, legal-action mask, chosen action, reward, next observation, termination flags, and diagnostic `info` (including attack rolls). A run card records the policy or model, exact episode seeds, field and action names, aggregate metrics, and loss seeds. This is evaluation data only; it does not change the policy observation or combat transitions. Training also saves traces and a run card for its final evaluation. Use a separate seed range for final comparison because periodic model selection already uses the training run's evaluation seeds.
`evaluation.compare` compares two run cards evaluated on the same seeds and reports losses and the first trace divergence. Later dice rolls can differ after policies choose different actions, so the report does not attribute outcome differences solely to that first decision.

`ExpectimaxAgent` instead enumerates exact chance outcomes through
`combat.transitions` and maximizes expected win probability to a fixed number of
turn boundaries, scoring the frontier with a pluggable leaf value (exact terminal
values, a heuristic race estimate, or a learned MLP). New stages plug in by
registering their own transition model. See [expectimax planning](EXPECTIMAX_PLANNING.md).

## Future BG3 integration

The platform-neutral [integration contract](BG3_INTEGRATION.md) separates rich BG3 facts from policy observations:

```text
BG3 collection → integration snapshots/events → observation adapter → policy vector
```

The collection harness preserves BG3 entity, ability, and resource IDs. The adapter later chooses fields and maps IDs for M0, M1, M2, or richer policies. The harness must not emit the current PPO vector directly. The `integration/` package is independent of `combat/`, Gymnasium, and training code; `combat/` does not import it. The current simulator evaluation JSONL traces remain a separate format containing already encoded observations and masks. The Python semantic command protocol and a synthetic mock executor are implemented separately from passive observations. Commands preserve original BG3 ability IDs and require current legality evidence; acceptance does not confirm execution. The passive H0 collector and Windows capture tools exist but are not live-verified. Observation adaptation to a game-playing policy and real action execution remain unimplemented. See the integration contract for session transport and recovery limits.
