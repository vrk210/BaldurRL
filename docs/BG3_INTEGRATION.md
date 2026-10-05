# BG3 observation integration contract

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

`integration/schema.py` is a platform-neutral contract, independent of Gymnasium and the simulator. `GameSnapshot` answers **what is true now**; `GameEvent` answers **what just happened**. A snapshot contains no event history. Both carry `schema_version = 1` and a collector-assigned `sequence`. The future collector must increase sequence across its emitted records to preserve order; timestamp milliseconds are optional. `to_dict()` emits JSON-compatible primitives and arrays, with enum values as lowercase strings and unknown values as JSON `null`. Python serialization, reading, inspection, and interval reconstruction are implemented; collection, policy adaptation, and action execution are future work.

`CombatSnapshot.participant_ids` may be empty until participant enumeration works. `EntitySnapshot.armor_class`, `alive`, and `position` may remain unknown if unavailable at capture time. `AbilitySnapshot.visible` does not assert usability. A candidate decision describes ability and target intent; `LEGAL` means a source establishes current legality, `ILLEGAL` means a source establishes rejection, and `UNKNOWN` means available evidence is insufficient. `legality_source` and `legality_reason` preserve that evidence. Ability ownership plus apparent resources do not reconstruct BG3 UI legality in general: target validity, range, line of sight, status, and other conditions can matter. Do not label such a candidate legal solely from ownership and resources.

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
BG3 Script Extender Lua (NOT IMPLEMENTED)
        │
        │ Ext.IO.SaveFile
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
Later H1 demonstration assembly (NOT IMPLEMENTED)
```

[Script Extender documents `Ext.IO.SaveFile` and `Ext.IO.LoadFile`](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md). File location, safe read behavior, naming, and polling cadence require live tests. Temp-file/rename behavior remains `TODO_VERIFY` on Windows. The Python reader uses simple polling through repeated `read_new()` calls; it does not watch the filesystem asynchronously.

### Version 1 wire record

Each `.json` file contains one top-level record. The existing v1 `GameSnapshot.to_dict()` and `GameEvent.to_dict()` now include a stable `record_type` discriminator; this is a v1 clarification made before any live producer exists. The rest of the fields remain at the top level, without an envelope:

```json
{"record_type": "snapshot", "schema_version": 1, "sequence": 1, "timestamp_ms": null, "controlled_entity_id": "player-1", "combat": null, "entities": [], "candidate_decisions": []}
```

```json
{"record_type": "event", "schema_version": 1, "sequence": 2, "timestamp_ms": null, "kind": "ability_used_on_target", "combat_id": null, "round_number": null, "actor_id": "player-1", "target_id": "enemy-1", "ability_id": "ability-basic-attack", "target_position": null, "story_action_id": null, "damage": null, "damage_type": null, "critical": null}
```

The IDs above are **synthetic test IDs**, not discovered BG3 IDs. `integration.serialization.parse_record()` rejects unsupported versions, unknown types and fields, and malformed nested values. Optional facts may be omitted or `null`; omitted collections default to empty arrays. Required fields are `record_type`, `schema_version`, and positive integer `sequence`, plus `kind` for events. Nested entities require `entity_id`; resources require `resource_id` and `amount`; abilities require `ability_id`; candidates require `ability_id`, `target_kind`, and `legality`. Enum values are lowercase strings. Schema v1 uses no simulator defaults.

The eventual Lua producer only needs to choose the next monotonically increasing sequence, populate `schema_version` and `record_type`, serialize one snapshot or event, and write one JSON file into the record directory. It does not need PPO, Gymnasium, reward, or observation encoding.

### Python use

```text
python -m integration.mock_h0 /tmp/baldurrl-h0
python -m integration.inspect /tmp/baldurrl-h0
```

`RecordDirectoryReader(directory, start_sequence=1).read_new()` emits each record once and stops at a missing sequence, retaining later records until another poll. Use `start_sequence` when attaching to a stream that starts after 1. `read_all()` returns all available records sorted by embedded sequence; `validate_all()` reports gaps for offline inspection. Duplicate IDs and invalid JSON or schema records raise errors naming the offending file. Record files are treated as immutable after a successful read.

`reconstruct_intervals()` creates a `SnapshotInterval(before, events, after)` between consecutive snapshots. An interval is **not necessarily one decision**. Trailing events after the last snapshot remain in the raw stream and are not placed in a completed interval. `controlled_action()` checks only ability-use events whose `actor_id` matches the requested entity; it returns `None` for no match and raises `AmbiguousActionError` for multiple matches. It does not infer rewards, legal actions, or policy labels.

**IMPLEMENTED (Python H0):** schema, serialization/parser, sequence-aware directory reader, interval reconstruction, inspection CLI, synthetic fixture. **NOT IMPLEMENTED / WINDOWS:** Script Extender producer, Osiris subscriptions, real BG3 resource and spell IDs, participant enumeration, legal-action reconstruction, action execution. All live-verification checklist items below remain open.

## Windows live-verification checklist

### Entity/state

- [ ] Dump host character component names.
- [ ] Dump one Goblin/enemy component names.
- [ ] Compare Osiris HP/max HP with ECS values.
- [ ] Locate runtime armor class.
- [ ] Verify alive/dead/downed representation.

### Resources

- [ ] Discover exact Action resource identifier.
- [ ] Discover exact Bonus Action identifier.
- [ ] Inspect movement resource.
- [ ] Inspect spell-slot representation.
- [ ] Determine how Second Wind availability is represented.
- [ ] Determine how Action Surge availability is represented.

### Abilities

- [ ] Dump Fighter SpellBook.
- [ ] Identify Basic Attack ability ID.
- [ ] Identify Second Wind ability ID.
- [ ] Identify Action Surge ability ID.
- [ ] Compare HasSpell / SpellBook / visible UI abilities.

### Combat/event ordering

- [ ] Record CombatStarted.
- [ ] Record EnteredCombat.
- [ ] Record TurnStarted / TurnEnded.
- [ ] Record CombatRoundStarted.
- [ ] Record UsingSpell.
- [ ] Record UsingSpellOnTarget.
- [ ] Record AttackedBy.
- [ ] Record MissedBy.
- [ ] Record CriticalHitBy.
- [ ] Record Died.
- [ ] Record CombatEnded.
- [ ] Inspect event ordering.
- [ ] Inspect StoryActionID correlation.

### Legality

- [ ] Compare candidate actions to actual enabled UI actions.
- [ ] Test CanCastSpellOnEnemyInSameCombat.
- [ ] Identify range restrictions.
- [ ] Identify line-of-sight restrictions.
- [ ] Identify status restrictions.

### File bridge

- [ ] Write a test JSON file with Ext.IO.SaveFile.
- [ ] Read it from Python.
- [ ] Determine actual Script Extender output path.
- [ ] Test repeated writes.
- [ ] Test sequence numbering.
- [ ] Test whether temp-file + rename is needed for safe reads.
- [ ] Measure reasonable polling latency for a turn-based game.

## Harness milestones

### H0 — Passive combat logger

H0 only records; it does not control the game. Success means:

1. Load a local BG3 save.
2. Enter a controlled 1v1 combat.
3. Detect combat start.
4. Emit a `GameSnapshot` at the Fighter turn start.
5. Record an ability-use event when the human attacks.
6. Record hit/miss/critical/damage outcome events as applicable.
7. Emit another snapshot after resolution.
8. Record Goblin turn/events.
9. Record death/combat end.
10. Read emitted JSON in Python and reconstruct an ordered event/snapshot trace (Python side implemented with synthetic records; live verification pending).

Later stages, described here only:

- **H1 — Reliable human-decision recorder:** snapshot before, candidate/legal decisions, human action, outcome events, snapshot after.
- **H2 — Validated legal-action extraction.**
- **H3 — Python → BG3 command execution.**
- **H4 — BG3-backed Gymnasium adapter.**

The schema is designed so a future demonstration record can be assembled as `snapshot_before + candidate/legal decisions + human chosen action + outcome events + snapshot_after`. This can support behavior cloning, DAgger-style human correction, and simulator/BG3 validation later. The Python H0 tools do not assemble demonstrations or verify live event semantics.
