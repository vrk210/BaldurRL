# BG3 observation and semantic command integration contract

## Boundary and data flow

```text
BG3
 │
 ├── Osiris events / queries
 ├── Script Extender / ECS
 └── Script Extender IO
        │
        ▼
 Observation Harness
        │
        ├── GameSnapshot stream
        └── GameEvent stream
        │
        ▼
 Python integration layer
        │
        ▼
 Observation Adapter
        │
        ├── M0 vector
        ├── M1 vector
        └── future richer vectors
        │
        ▼
 Agent
```

The harness captures rich semantic facts from the real game. It does **not** emit M0's seven-value PPO observation or infer simulator resources. An observation adapter will later choose what each policy sees and map raw BG3 IDs to policy concepts. Preserve entity, ability, and resource IDs supplied by BG3 until that boundary. Unknown facts remain `None`, an empty tuple when enumeration is incomplete, or `Legality.UNKNOWN`; do not fill them with simulator presets. The current `evaluation/evaluate.py` JSONL trace is a separate simulator format with flat vectors, masks, and diagnostic `info`.

`integration/schema.py` is a platform-neutral contract, independent of Gymnasium and the simulator. `GameSnapshot` answers **what is true now**; `GameEvent` answers **what just happened**. A snapshot contains no event history. Both carry `schema_version = 1` and a collector-assigned `sequence`. The collector increases sequence across its emitted records within one session directory; timestamp milliseconds are optional. `to_dict()` emits JSON-compatible primitives and arrays, with enum values as lowercase strings and unknown values as JSON `null`.

`CombatSnapshot.participant_ids` may be empty until participant enumeration works. `EntitySnapshot.armor_class`, `alive`, and `position` may remain unknown if unavailable at capture time. `AbilitySnapshot.visible` does not assert usability. A candidate decision describes ability and target intent; `LEGAL` means a source establishes current legality, `ILLEGAL` means a source establishes rejection, and `UNKNOWN` means available evidence is insufficient. `legality_source` and `legality_reason` preserve that evidence. Ability ownership plus apparent resources do not reconstruct BG3 UI legality in general: target validity, range, line of sight, status, and other conditions can matter. Do not label such a candidate legal solely from ownership and resources.

The schema is designed so a future demonstration record can be assembled as `snapshot_before + candidate/legal decisions + human chosen action + outcome events + snapshot_after`. This can support behavior cloning, DAgger-style human correction, and simulator/BG3 validation later. The Python tools for the [passive H0 collector](../bg3/h0/README.md) do not assemble demonstrations or verify live event semantics.

## Simulator concept to integration field

`VERIFIED_API` means the linked API is documented, not that the proposed collector mapping has been observed live. `LIVE_VERIFY_*` and `TODO_VERIFY` retain the unresolved work.

| Current BaldurRL concept | Integration-schema field | Potential BG3 source | Status |
| --- | --- | --- | --- |
| `Character.hp`, `Character.max_hp` | `EntitySnapshot.hp`, `EntitySnapshot.max_hp` | [GetHitpoints](https://docs.baldursgate3.game/index.php?title=GetHitpoints), [GetMaxHitpoints](https://docs.baldursgate3.game/index.php?title=GetMaxHitpoints) | VERIFIED_API |
| Environment combat/turn state | `CombatSnapshot.combat_id`, `round_number`, `active_entity_id` | [CombatStarted](https://docs.baldursgate3.game/index.php?title=CombatStarted), [CombatEnded](https://docs.baldursgate3.game/index.php?title=CombatEnded), [EnteredCombat](https://docs.baldursgate3.game/index.php?title=EnteredCombat), [TurnStarted](https://docs.baldursgate3.game/index.php?title=TurnStarted), [TurnEnded](https://docs.baldursgate3.game/index.php?title=TurnEnded), [CombatRoundStarted](https://docs.baldursgate3.game/index.php?title=CombatRoundStarted), [CombatGetActiveEntity](https://docs.baldursgate3.game/index.php?title=CombatGetActiveEntity) | VERIFIED_API |
| Fixed melee range (simulator) | `EntitySnapshot.position` | [GetPosition](https://docs.baldursgate3.game/index.php?title=GetPosition), [GetDistanceTo](https://docs.baldursgate3.game/index.php?title=GetDistanceTo) | VERIFIED_API |
| `ResourcePool` counts | `ResourceValue` | [GetActionResourceValuePersonal](https://docs.baldursgate3.game/index.php?title=GetActionResourceValuePersonal), [Script Extender ActionResources component](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md) | VERIFIED_API / LIVE_VERIFY_IDENTIFIERS |
| `Character.known_abilities` | `EntitySnapshot.abilities` | [HasSpell](https://docs.baldursgate3.game/index.php?title=HasSpell), [CanShowSpellForCharacter](https://docs.baldursgate3.game/index.php?title=CanShowSpellForCharacter), [Script Extender SpellBook.Spells](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md) | VERIFIED_API / LIVE_VERIFY_STRUCTURE |
| Human chosen action | `GameEvent` `ABILITY_USED*` | [UsingSpell](https://docs.baldursgate3.game/index.php?title=UsingSpell), [UsingSpellOnTarget](https://docs.baldursgate3.game/index.php?title=UsingSpellOnTarget), [UsingSpellAtPosition](https://docs.baldursgate3.game/index.php?title=UsingSpellAtPosition) | VERIFIED_API |
| Attack result | `GameEvent` `DAMAGE` / `MISS` / `CRITICAL_HIT` | [AttackedBy](https://docs.baldursgate3.game/index.php?title=AttackedBy), [MissedBy](https://docs.baldursgate3.game/index.php?title=MissedBy), [CriticalHitBy](https://docs.baldursgate3.game/index.php?title=CriticalHitBy), [Died](https://docs.baldursgate3.game/index.php?title=Died) | VERIFIED_API |
| Decision to outcome association | `GameEvent.story_action_id` | StoryActionID exposed by [spell-use](https://docs.baldursgate3.game/index.php?title=UsingSpellOnTarget) and [combat-result](https://docs.baldursgate3.game/index.php?title=AttackedBy) events | VERIFIED_API_FIELD / LIVE_VERIFY_CORRELATION_RELIABILITY |
| `Character.armor_class` | `EntitySnapshot.armor_class` | Script Extender ECS/stats | TODO_VERIFY |
| Environment action mask | `DecisionCandidate.legality` | Known ability, resource availability, [CanCastSpellOnEnemyInSameCombat](https://docs.baldursgate3.game/index.php?title=CanCastSpellOnEnemyInSameCombat), future target/range/LOS checks | PARTIAL / TODO_VERIFY |

The APIs expose StoryActionID, but actual event ordering and correlation in our encounters still require live verification. No exact Action, Bonus Action, Second Wind, Action Surge, or basic attack identifier is assumed here. Do not claim an armor-class ECS property before inspecting a live entity.

## First file bridge

```text
BG3 Script Extender Lua (OFFLINE-TESTED / LIVE-UNVERIFIED)
        │
        │ Ext.IO.SaveFile (.pending JSON + .ready byte count)
        ▼
integration.publish (validate + atomic publication)
        │
        ▼
JSON snapshot/event files
        │
        ▼
RecordDirectoryReader (IMPLEMENTED)
        │
        ▼
GameSnapshot / GameEvent stream
        │
        ▼
SnapshotInterval reconstruction
        │
        ▼
Demonstration assembly (NOT IMPLEMENTED)
```

[Script Extender documents `Ext.IO.SaveFile` and `Ext.IO.LoadFile`](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md), but no atomic rename API. The Lua collector stages immutable payloads and completion markers; the Python publisher validates them and uses `os.replace` to expose complete `.json` files. This protocol is tested offline. Output-root resolution, readback behavior, and atomic replacement on Windows remain `TODO_VERIFY`. The optional publisher watch loop polls files every 0.5 seconds. The session runner polls the record directory with `read_all()`; nothing watches the filesystem asynchronously. Lua collection is event-driven, without per-frame polling.

### Version 1 wire record

Each `.json` file contains one top-level record. The existing v1 `GameSnapshot.to_dict()` and `GameEvent.to_dict()` now include a stable `record_type` discriminator; this is a v1 clarification made before any live producer exists. The rest of the fields remain at the top level, without an envelope:

```json
{"record_type": "snapshot", "schema_version": 1, "sequence": 1, "timestamp_ms": null, "controlled_entity_id": "player-1", "combat": null, "entities": [], "candidate_decisions": []}
```

```json
{"record_type": "event", "schema_version": 1, "sequence": 2, "timestamp_ms": null, "kind": "ability_used_on_target", "combat_id": null, "round_number": null, "actor_id": "player-1", "target_id": "enemy-1", "ability_id": "ability-basic-attack", "target_position": null, "story_action_id": null, "damage": null, "damage_type": null, "critical": null}
```

The IDs above are **synthetic test IDs**, not discovered BG3 IDs. `integration.serialization.parse_record()` rejects unsupported versions, unknown types and fields, and malformed nested values. Optional facts may be omitted or `null`; omitted collections default to empty arrays. Required fields are `record_type`, `schema_version`, and positive integer `sequence`, plus `kind` for events. Nested entities require `entity_id`; resources require `resource_id` and `amount`; abilities require `ability_id`; candidates require `ability_id`, `target_kind`, and `legality`. Enum values are lowercase strings. Schema v1 uses no simulator defaults.

The passive Lua producer in [bg3/h0](../bg3/h0/README.md) chooses increasing sequences, populates `schema_version` and `record_type`, and stages one JSON document per record. `python -m integration.publish` validates completion markers and atomically publishes `.json` files for this reader. Each collector session has its own directory and sequence namespace. It does not need PPO, Gymnasium, reward, or observation encoding.

### Python use

```text
python -m integration.mock_h0 /tmp/baldurrl-h0
python -m integration.inspect_records /tmp/baldurrl-h0
```

One record directory represents one collector session / sequence namespace. For H0, a restarted collector should start a new output directory rather than reuse sequence IDs in an existing directory. `RecordDirectoryReader(directory).read_all()` returns all available records sorted by embedded sequence, and `validate_all()` reports gaps from sequence 1. Duplicate IDs and invalid JSON or schema records raise errors naming the offending file. Record files are treated as immutable after a successful read.

`reconstruct_intervals()` creates a `SnapshotInterval(before, events, after)` between consecutive snapshots. An interval is **not necessarily one decision**. Trailing events after the last snapshot remain in the raw stream and are not placed in a completed interval. `controlled_action()` checks only ability-use events whose `actor_id` matches the requested entity. Multiple events with the same non-null `story_action_id` are treated as one attempted action; a targeted event is preferred over a position event, which is preferred over a generic event. Distinct IDs, multiple events without an ID, or conflicting target and position events raise `AmbiguousActionError`; no match returns `None`. Raw interval events are preserved. This correlation rule needs live verification. It does not infer rewards, legal actions, or policy labels.

**IMPLEMENTED (offline H0):** schema, serialization/parser, sequence-aware directory reader, interval reconstruction, inspection CLI, synthetic fixture, passive Script Extender Lua source mod, documented Osiris subscriptions, conservative snapshot probes, and completion-aware file publisher. **LIVE-UNVERIFIED / WINDOWS:** mod packaging/loading, actual API signatures/availability, callback timing and event ordering, output paths and Windows IO, and snapshot completeness. Resource/owned-ability/participant enumeration, legality, and action execution remain unimplemented. Spell IDs are captured only from callbacks.

## Python semantic command protocol

The Python side of a future command executor is implemented in `integration/commands.py`,
`integration/command_transport.py`, and `integration/mock_executor.py`. It has no
imports from `combat/`, Gymnasium, trained agents, or ML libraries. It does not
execute actions in BG3 or simulate BG3 mechanics. Passive `GameSnapshot`,
`GameEvent`, `parse_record()`, and `RecordDirectoryReader` retain their v1 contract.
Commands and acknowledgements have their own v1 schema and parsers; do not place
these records in the observation directory or feed them to `parse_record()`.

### Session layout and wire format

Each session has its own directory tree and independent ID/sequence namespace:

```text
sessions/<session-id>/
    records/             # existing snapshot/event JSON, collector sequence
    commands/            # <command_id>.json, immutable semantic intent
    acknowledgements/    # <command_id>.1.json, .2.json, and .observation.json
```

`CommandSession(session_directory)` creates these three directories. Use a fresh
session when restarting a collector rather than reusing its sequence IDs.
Observation sequences continue to include events; command IDs and per-command
acknowledgement sequences do not consume collector sequence numbers.

```json
{"record_type":"command","schema_version":1,"command_id":"demo-001","snapshot_sequence":1,"expected_combat_id":"synthetic-combat","actor_entity_id":"synthetic-player","action_type":"use_ability","ability_id":"synthetic-ability","target_kind":"entity","target_entity_id":"synthetic-enemy","target_position":null}
```

`ActionCommand` requires a unique `command_id`, positive `snapshot_sequence`,
`actor_entity_id`, `action_type`, and `target_kind`, plus `schema_version=1`.
Generate IDs with `str(uuid.uuid4())`, or supply an ID unique within this session.
IDs must contain 1–128 filename-safe ASCII characters: the first is alphanumeric,
and the remainder may also contain `_`, `-`, or `.`. BG3 entity, combat, and
ability IDs remain opaque nonempty strings and are never mapped to PPO indices.

| Action/target | Required intent | Forbidden intent |
| --- | --- | --- |
| `use_ability` | Original BG3 `ability_id` | Missing/empty ability ID |
| `end_turn` | `target_kind=none` | Ability ID or target payload |
| `none` | No target payload | Entity ID or position |
| `self` | Actor is the implicit target | Redundant entity ID or position |
| `entity` | `target_entity_id` | Position |
| `position` | `target_position={"x":number,"y":number,"z":number}` | Entity ID |

Coordinates must be finite JSON numbers, excluding booleans. Optional IDs and
positions may be omitted or `null`; the serializer emits explicit `null` values.
`parse_command()`, `parse_acknowledgement()`, and
`parse_post_execution_observation()` reject unknown fields, missing
required fields, unsupported versions, wrong types, unknown enum values, and
inconsistent targets. Transport JSON loading additionally rejects duplicate
object keys and nonstandard `NaN`/`Infinity` constants. File errors name the path.
Malformed inbox records fail closed with a validation error, rather than emitting
an acknowledgement for an untrusted or unparseable command ID.

### Freshness and legality

`publish_command()` checks the current observation stream and writes only valid
commands. The mock independently repeats those checks when receiving a command,
so a command published before a new snapshot can still be rejected. A command
must refer to exactly the latest snapshot sequence; older and future sequences
are rejected. Observation gaps, an empty stream, or events following the latest
snapshot require a fresh snapshot before issuing another decision. Files are
immutable for the session lifetime. Validation and latest-snapshot selection use
one `read_all()` collection; another publication cannot switch the selected
snapshot to an unvalidated collection within the same check. This does not lock
an independent observation publisher or freeze BG3.

Completion does not make the pre-action facts fresh. Both command publication
and executor receipt use `validate_command_for_execution()`. After each
`resolved` or `failed` result, another command requires an explicit
`PostExecutionObservation` confirmation for that completed command. A genuinely
`rejected` intent was never executed and imposes no such boundary, so a new
command ID can retry from the unchanged snapshot if it is still current/legal.
`uncertain` continues to hold the in-flight slot and cannot be cleared by a
snapshot confirmation.

The completion/capture handshake is:

1. The executor establishes completion or a known, stopped failure. It publishes
   the terminal result with `observation_boundary_sequence`, at least the highest
   observation sequence already published and the command's decision sequence.
   Python `publish_acknowledgement()` fills this boundary when omitted; its
   returned file contains the persisted value. External result producers must
   supply the boundary themselves. Missing boundaries in older result files
   fail closed and require external recovery.
2. After observing that terminal acknowledgement, the trusted collector initiates
   a fresh capture. It must not relabel a cached snapshot or a delayed snapshot
   captured before completion. It publishes the snapshot in the same collector
   sequence namespace, then publishes a `PostExecutionObservation` record through
   `publish_post_execution_observation()` (or the equivalent future producer).
3. The confirmation references the completed command ID and a snapshot strictly
   after the persisted boundary. Its `reason` describes the capture evidence.
   The gate checks the result status, boundary, snapshot existence, continuity,
   and uniqueness. The next command must use the current latest snapshot at or
   after that confirmation. Every previous executed command requires a valid
   confirmation; restart does not discard these requirements.

```json
{"record_type":"post_execution_observation","schema_version":1,"command_id":"demo-001","snapshot_sequence":2,"reason":"Synthetic capture initiated after observing the terminal result"}
```

This confirmation lives in `acknowledgements/<command_id>.observation.json`, not
in `records/`; passive observation schemas and readers are unchanged. The
collector attestation explicitly establishes the capture-after-completion
relation. **A higher sequence, a later file publication, or the result's optional
`observation_sequence` alone does not prove post-action timing.** Python trusts
this producer assertion just as it trusts a collector's `LEGAL` assertion; the
files do not independently authenticate game timing. A real producer must
coordinate capture with the terminal result and preserve capture/sequence order,
including draining or discarding old queued captures. Verifying that producer
behavior in BG3 remains future work. If effects may still be ongoing after a
failure, the executor must report `uncertain` rather than assert a stopped
failure and fresh capture.

The actor must match `controlled_entity_id`, which must be known. When combat
exists, it must also match the known `active_entity_id`. `expected_combat_id`
must equal the observed combat ID, including `null` when unknown: omitting a
known combat ID does not bypass the check.

Ability commands must exactly match one current `DecisionCandidate` by original
ability ID, target kind, entity ID, and position, with `legality=LEGAL`. The
collector's explicit `LEGAL` assertion is trusted; this layer does not invent
legality from ownership, visibility, or resources. Missing, duplicated,
`ILLEGAL`, and `UNKNOWN` candidates are rejected. An empty candidate list does
not disable legality checks. Positions compare exactly, without rounding or
range inference. Self-target candidates use the same implicit-actor form.

The existing observation candidate schema describes abilities only; it cannot
represent turn control. For `end_turn`, legality checking is limited to verifying
the current controlled actor's active combat turn. It is rejected outside combat
or when the active actor is unknown. No fake BG3 end-turn ability ID or change to
the observation schema is introduced. Richer BG3 end-turn restrictions still
require future verified evidence and live implementation.

### Acknowledgement semantics

```json
{"record_type":"acknowledgement","schema_version":1,"command_id":"demo-001","acknowledgement_sequence":1,"status":"accepted","reason":null,"observation_sequence":null,"observation_boundary_sequence":null}
```

```json
{"record_type":"acknowledgement","schema_version":1,"command_id":"demo-001","acknowledgement_sequence":2,"status":"resolved","reason":"Synthetic fixture confirmation; no BG3 execution","observation_sequence":null,"observation_boundary_sequence":1}
```

| Status | Sequence | Meaning | Releases the command slot |
| --- | --- | --- | --- |
| `accepted` | 1 | Validated and accepted for execution; no completion evidence | No |
| `rejected` | 1 | Refused before execution, with a reason | Yes |
| `resolved` | 2 | Execution confirmed completed, with evidence described in `reason` | Yes; next decision awaits confirmed post-execution capture |
| `failed` | 2 | Execution reported a known, stopped failure, with a reason | Yes; next decision awaits confirmed post-execution capture |
| `uncertain` | 2 | Execution outcome cannot be established, with a reason | No |

Sequence 2 requires a preceding `accepted` record. Results after rejection,
unknown command IDs, duplicate acknowledgement sequences, and repeated or
conflicting results are errors. `reason` is required for every status except
`accepted`. Optional positive `observation_sequence` may identify supporting
observations, but does not establish correlation by itself. The separate
`observation_boundary_sequence` is a positive sequence fence for resolved/failed
results, not completion or snapshot-timing evidence by itself. A future real
executor must supply verified completion evidence before reporting `resolved`;
acceptance or an elapsed timeout is insufficient. Resolution means the attempted
action completed, not that an attack hit or achieved a desired gameplay effect.

### Publication, duplicates, and recovery limits

Python publication writes and fsyncs a temporary file on the destination
filesystem, then atomically hard-links it to the final name without overwriting
an existing record. Pollers ignore temporary files. This requires a filesystem
with atomic hard-link creation; unsupported filesystems raise an error without
a partial final file. Windows/Script Extender producer behavior is still
unverified. There is no networking or filesystem watcher.

A short-lived `.writer-lock/` directory serializes cooperating Python writers.
`publish_command()` allows at most one pending, accepted, or uncertain command
per session. A released execution slot still requires post-execution snapshot
confirmation before another command can be published or accepted. External
conflicting inbox commands are rejected by the mock.
Duplicate command IDs are errors, including identical payloads and IDs already
completed. Duplicate inbox files fail closed; the original history is never
replaced with a new rejection. Repeated mock polling skips acknowledged IDs.

**This is not exactly-once execution across crashes.** The lock and immutable
history prevent ordinary cooperating writers from overwriting or replaying IDs;
they cannot atomically commit a game effect and its acknowledgement. A crash
between those operations can leave a pending or accepted command whose real
outcome is unknown. Missing results never trigger automatic retries. On reopen,
the mock leaves accepted-only histories in flight. Completed histories without
post-execution confirmation continue to block new decisions. `uncertain` also
blocks new commands; this minimal protocol intentionally has no retry/reset API.

Keep session files intact. Before recovery, stop all writers, inspect any stale
lock/temp files, and reconcile the game state externally. Remove a stale lock
only after establishing that its owner is gone. Do not blindly replay commands
or delete acknowledgement history. After reconciliation, a fresh session and
fresh observations provide a new namespace; they do not prove that a previous
effect happened only once. A known `failed` result likewise does not roll back
partial effects. File fsync does not guarantee directory-entry durability after
power loss. The future collector must coordinate a stable snapshot with command
intake; this Python lock does not freeze BG3 or cover an independent Lua writer.

### Synthetic CLI demonstration

```text
python -m integration.mock_executor /tmp/baldurrl-command-session-001
python -m integration.inspect_records /tmp/baldurrl-command-session-001/records
```

Use a fresh empty session tree for each demo. The CLI writes a synthetic
observation with `LEGAL` and `UNKNOWN` candidates, reads it using the existing
reader, selects the legal intent, publishes a command, and prints the mock's
`accepted` acknowledgement followed by a separate `resolved` fixture result.
It then initiates a new synthetic capture and prints the persisted post-execution
observation confirmation; fixture state stays unchanged because no mechanics run.
`MockExecutor.receive()` only validates and accepts/rejects; `finish()` explicitly
reports a synthetic `resolved`, `failed`, or `uncertain` outcome. It never mutates
HP/resources/turns, rolls dice, invokes BG3, or derives rewards. Fixture
confirmation is clearly labeled and is not evidence of actual game execution.

## Windows verification and future execution path

[The H0 runbook](../bg3/h0/README.md#first-windows-session) is the Windows
procedure: installation, static preflight, session capture and archival,
read-only actor probes, bounded encounter-trial analysis, and the live checklist
of entity, resource, ability, ordering, legality and file-bridge checks. These
tools keep to this v1 contract; they don't establish live legality or execute
actions. Unknown facts stay unknown until those checks establish their meaning,
and offline success doesn't establish game behavior. Real traces should become
conformance fixtures for the simulator mechanics they cover.

Decision (2026-10-09): the first execution experiment uses the normal client
UI. Script Extender client keyboard injection is used where it reproduces the
human interaction, and Windows keyboard/mouse input is the fallback for
targeting and confirmation. Neither path has been demonstrated, and no
executor exists. Revisit if a live input trial or a documented,
rule-respecting public submission API supplies new evidence.

- [Osi.UseSpell][exec-use-spell] bypasses player preconditions, including spell
  ownership/resource availability. It is unsuitable as the ordinary-action
  executor. This does not prove all casts are free or all targets are accepted.
- Internal cast-request arrays are exposed and [Brawl][exec-brawl] shows queue
  writes; that is not a supported, validated public action API. Do not fabricate
  requests or infer safe lifetimes/validation from mapped structure names.
- [SE client input][exec-input] exposes key injection; focus, bindings, UI routing,
  target selection and observed completion still require live tests.
  [Windows SendInput][exec-windows] is the fallback, with the same evidence needs.
- Noesis UI command wrappers and native AI delegation remain research leads;
  no verified semantic ability/target executor was identified.

The inspected SE revision was `91abbf230d02faa2b093d0a458d9caf6d1494901`.
Record the installed game/SE versions; pinned source is not evidence about a
particular binary. Before starting execution work, read
`docs/BG3_ACTION_EXECUTION.md` in tag `archive/simulator-research-2026-10-15`.
It holds:

- the seven-step live trial plan and its pass/fail criteria: one valid
  autonomous attack, the same resource cost as a human attack, rejection of an
  invalid attack, and no blind retries;
- read-only Lua console snippets for tracing cast requests on the server and
  input bindings on the client, which `!baldurrl_inspect` doesn't cover;
- the open questions only the game can answer;
- the request-layout analysis and the reference index.

[exec-use-spell]: https://docs.baldursgate3.game/index.php?title=UseSpell&oldid=2261

[exec-brawl]: https://github.com/tinybike/Brawl/blob/1ac3c7c1a6bc01cdc21cb9ec8a0f857210f0ef31/Brawl/Mods/Brawl/ScriptExtender/Lua/Server/Actions.lua#L218-L315

[exec-input]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/ClientInput.inl#L6-L53

[exec-windows]: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
