# BaldurRL passive BG3 H0 collector

**Implemented and verified offline; never loaded in BG3/Windows.** This source
mod observes human gameplay on the host server. No game-state writes, agent,
input injection, training, or model calls. The only writes are collector files.
Loader behavior, API availability, callback timing, and Windows IO remain live
verification blockers; this is a source mod, not a tested release `.pak`.

## Layout and installation

The mod uses [Norbyte's loader conventions](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md#bootstrap-scripts)
and [LaughingLeader's community setup example](https://github.com/LaughingLeader/BG3ModdingTools/wiki/Script-Extender-Lua-Setup).
Metadata follows the community [bg3modmeta snippet](https://github.com/LaughingLeader/BG3ModdingTools/blob/master/.vscode/BaldursGate3.code-snippets),
with no gameplay dependencies or story scripts. The UUID below identifies this
mod, not a BG3 resource or spell.

```text
bg3/h0/
  Mods/BaldurRL_H0/
    meta.lsx                    # UUID d860477b-c599-4f12-9e95-5b8ec7235860
    ScriptExtender/
      Config.json               # Lua only; targets SE >=30 (live-unverified)
      Lua/
        BootstrapServer.lua     # subscriptions / lifecycle
        BootstrapClient.lua     # intentionally empty
        Observation.lua         # read-only Osiris probes / event state
        Wire.lua                # pure Lua JSON / v1 field mapping
        Emitter.lua             # sequence and completion protocol
```

1. On Windows, install the [Script Extender updater](https://github.com/Norbyte/bg3se/releases)
   by placing `DWrite.dll` in the game's `bin` directory. Launch once to download
   the extender. Check the reported extender/game versions.
2. Merge `{"CreateConsole": true, "LogRuntime": true}` into
   `bin/ScriptExtenderSettings.json` (valid JSON, preserving existing settings).
3. Package **the contents of `bg3/h0`**, with `Mods` at the package root, as
   `BaldurRL_H0.pak` using [LSLib/ConverterApp](https://github.com/Norbyte/lslib).
   Import into [BG3 Mod Manager](https://github.com/LaughingLeader/BG3ModManager),
   activate, and export the load order. Check the mod UUID in that order.
   For development, the community example also supports loose files under
   `<BG3>/Data/Mods/BaldurRL_H0`; the module must still be activated in the load
   order. Do not install both copies. Packaging and activation are untested here.
4. Load a local save. Confirm the server bootstrap and `[BaldurRL H0] New session:`
   message. `UNAVAILABLE` or `HALTED` output requires investigation before treating
   a capture as complete. With no Windows installation we cannot validate these steps.

## Running and inspecting one session

The collector starts on `SessionLoaded` and on Lua `ResetCompleted`. Each start
uses `Ext.Utils.GenerateGuid()` and creates a separate sequence namespace, starting
at 1. This API export is verified in [upstream Utils.inl](https://github.com/Norbyte/bg3se/blob/main/BG3Extender/Lua/Libs/Utils.inl);
its availability and GUID representation on the installed extender must be checked live.
No counters are saved in a BG3 savegame. Existing session directories are refused.

The logged relative staging path is `BaldurRL/H0/<session-guid>/staging`. Per
[the IO documentation](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md#io),
the usual absolute root is:

```text
%LOCALAPPDATA%\Larian Studios\Baldur's Gate 3\Script Extender\
```

**Verify this root on the actual installation.** It is not the updater cache or
the `Script Extender Logs` directory. A sequence produces one JSON document in
`record-00000001.pending`, then `record-00000001.ready` containing its UTF-8 byte
count and newline. Lua constructs JSON in memory, checks the SaveFile result and
exact LoadFile readback, and only then writes the marker. It stops on a write or
serialization failure; there are no retries that reuse sequence numbers.

SE has no documented atomic rename operation ([exported IO functions](https://github.com/Norbyte/bg3se/blob/main/BG3Extender/Lua/Libs/IO.inl)).
Therefore a small Python publisher is required: it validates marked bytes with
`integration.serialization.parse_record`, writes a temporary file, and uses
`os.replace` in the destination directory. A partial/unmarked payload never
appears as a completed `.json` record to the unchanged Python reader. Staging is
retained for audit. This is a completion protocol, not a promise of disk durability
through power loss; verify atomic replacement and IO on a Windows local filesystem.

From the repository root in PowerShell, replace `<session-guid>` with the value
printed by the collector:

```powershell
$capture = Join-Path $env:LOCALAPPDATA "Larian Studios\Baldur's Gate 3\Script Extender\BaldurRL\H0\<session-guid>"
python -m integration.publish "$capture\staging" "$capture\records" --watch
```

Enter a small combat and perform actions manually. In the SE console, select
`server`; `!baldurrl_snapshot` takes an extra snapshot, `!baldurrl_stop` stops
collection, and `!baldurrl_start` starts a **new** namespace. These commands only
control observation. Save reloads and `reset` also start new directories; switch
the publisher to the new logged session. Commands must run after a save loads.

Stop collection, then stop the publisher with Ctrl-C and perform a final sweep:

```powershell
python -m integration.publish "$capture\staging" "$capture\records"
python -m integration.inspect "$capture\records"
```

The general inspection command is exactly:

```text
python -m integration.inspect <directory>
```

Pass the **records directory of one session**, never the staging directory or
parent containing several sessions. Keep one publisher per session and a fresh
destination on a local filesystem. Repeating publication is idempotent; differing
existing records cause an error. Do not reuse an output directory for another
session. Inspection reports sequences, gaps, event counts and snapshot intervals.
Use the raw `.json` files to examine IDs, HP, StoryActionID, and order.

## Observed facts and limitations

Snapshots occur at session start, combat/entry/round/turn boundaries, damage,
miss, critical hit, and death, plus explicit console requests. They run after
the corresponding Osiris callback; they are **not** guaranteed to be an animation
completion or a human decision boundary. No per-frame polling or background Lua
timer is installed. The Python publisher polls files every 0.5 seconds only when
`--watch` is supplied.

| Snapshot fact | Read-only source | Conservative behavior |
| --- | --- | --- |
| Controlled entity | [GetHostCharacter](https://docs.baldursgate3.game/index.php?title=GetHostCharacter) → [GetReservedUserID](https://docs.baldursgate3.game/index.php?title=GetReservedUserID) → [GetCurrentCharacter](https://docs.baldursgate3.game/index.php?title=GetCurrentCharacter) | Re-queried each snapshot; no fixed user ID or avatar-as-selected assumption; null on failure/unassigned user |
| Entity IDs | Controlled entity and callback actor/target/attacker owner | Session-observed subset; not a participant roster; may include items, departed entities, or other combats |
| HP/max HP | [GetHitpoints](https://docs.baldursgate3.game/index.php?title=GetHitpoints), [GetMaxHitpoints](https://docs.baldursgate3.game/index.php?title=GetMaxHitpoints) | Integer values when queries succeed; null otherwise |
| Controlled entity's combat | [CombatGetGuidFor](https://docs.baldursgate3.game/index.php?title=CombatGetGuidFor) | null without a successful query or after a recorded end |
| Round number | Most recent CombatRoundStarted for that combat | null if collector missed it; never default to round 1 |
| Active entity | TurnStarted / TurnEnded ledger for that combat | Only the sole observed unended turn; null for none/shared turns; completeness requires live verification |

`CombatGetActiveEntity` is deliberately not used: its documented meaning is
[the first active, non-dead entity](https://docs.baldursgate3.game/index.php?title=CombatGetActiveEntity),
which does not establish the current turn owner. Resets mid-combat cannot recover
missed round/turn facts. Combat merges/shared initiative/selection changes require
live validation and can leave facts unknown. Unknown queries are logged once per
query per session. A query failure never substitutes simulator values.

Armor class, name, position, alive/downed state remain null. HP=0 is not used to
infer death. Resources, owned/visible abilities, participants, and legal candidate
decisions remain empty arrays. Spell IDs are copied from real callbacks only;
none are hardcoded. Timestamps are null (sequence orders collector callbacks).

## Event signatures and mapping

Every listener uses `Ext.Osiris.RegisterListener(name, arity, "after", handler)`.
Signatures below are verified against the linked API definitions, not the game
runtime. Registration failures print `UNAVAILABLE`; handler/write failures halt
the session visibly. `UsingSpell` announces the start of casting, not successful
execution; both generic and targeted callbacks are retained.

| Documented callback (argument order) | Arity | v1 event |
| --- | --- | --- |
| [CombatStarted](https://docs.baldursgate3.game/index.php?title=CombatStarted)(combat) | 1 | combat_started |
| [CombatEnded](https://docs.baldursgate3.game/index.php?title=CombatEnded)(combat) | 1 | combat_ended |
| [EnteredCombat](https://docs.baldursgate3.game/index.php?title=EnteredCombat)(entity, combat) | 2 | entered_combat; actor=entity |
| [CombatRoundStarted](https://docs.baldursgate3.game/index.php?title=CombatRoundStarted)(combat, round) | 2 | round_started |
| [TurnStarted](https://docs.baldursgate3.game/index.php?title=TurnStarted)(entity) | 1 | turn_started; actor=entity |
| [TurnEnded](https://docs.baldursgate3.game/index.php?title=TurnEnded)(entity) | 1 | turn_ended; actor=entity |
| [UsingSpell](https://docs.baldursgate3.game/index.php?title=UsingSpell)(caster, spell, type, element, StoryActionID) | 5 | ability_used; actor=caster; ability=spell |
| [UsingSpellOnTarget](https://docs.baldursgate3.game/index.php?title=UsingSpellOnTarget)(caster, target, spell, type, element, StoryActionID) | 6 | ability_used_on_target |
| [AttackedBy](https://docs.baldursgate3.game/index.php?title=AttackedBy)(defender, owner, attacker, damageType, amount, cause, StoryActionID) | 7 | damage; actor=direct attacker; target=defender; damage=amount |
| [MissedBy](https://docs.baldursgate3.game/index.php?title=MissedBy)(defender, owner, attacker, StoryActionID) | 4 | miss; actor=direct attacker; target=defender |
| [CriticalHitBy](https://docs.baldursgate3.game/index.php?title=CriticalHitBy)(defender, owner, attacker, StoryActionID) | 4 | critical_hit; critical=true |
| [Died](https://docs.baldursgate3.game/index.php?title=Died)(entity) | 1 | died; actor=deceased; no inferred attacker/StoryActionID |

All exposed StoryActionID values are copied, including zero; no event-time ID
association is guessed. Event combat/round context comes from actor or target
combat queries and previously observed rounds, and can be null after departure.
AttackedBy can report zero damage and status/environmental effects; it does not
establish a basic attack, critical flag, or spell ID. The v1 contract cannot store
attacker ownership, spell type/element, or damage cause; these arguments are not
repurposed into unrelated fields. Owner entity IDs are observed for snapshots.
UsingSpellAtPosition is outside this minimal H0 subscription set.

## Offline validation

```text
python -m pip install pytest lupa
python -m pytest tests/test_bg3_h0_lua.py tests/test_integration_publish.py tests/test_integration_*.py
```

`test_bg3_h0_lua.py` uses the **real Lua 5.4 runtime** in `lupa.lua54`, loading the
actual mod scripts with stubbed Ext/Osiris calls and a real temporary filesystem.
It checks the event trace through `parse_record`, round-trips all produced records,
inspects it with the existing reader, checks session resets, unknown facts,
shared turns, JSON escaping, and failed writes. These tests explicitly skip if
Lupa is unavailable; a skip is not Lua validation. Its metadata/layout test and
`test_integration_publish.py` are **Python/static-only** tests; they do not execute
Lua. No tests emulate engine timing or prove an installed SE version works.

## Windows live-verification checklist (all pending)

- [ ] Record BG3/SE versions; confirm updater, console, package activation, server
  bootstrap, all 12 listener registrations, and session GUID generation.
- [ ] Confirm exact IO root and UTF-8 bytes; test `.pending` readback, `.ready`
  completion, publisher atomic replacement, immutable `.json` records, and
  `python -m integration.inspect <directory>`. Interrupt a write: no partial
  completed JSON; an interrupted capture may require a fresh session.
- [ ] Confirm every arity and argument position above using an actual 1v1 fight:
  combat start/end, entry, round/turn start/end, manual ability use/target, damage,
  miss, critical, and death. Investigate any `UNAVAILABLE`/`HALTED` message.
- [ ] Compare selected character (switch companions), HP/max HP, combat GUID,
  round and sole turn owner against UI; note unknown fields and incomplete entity
  coverage. Test load/reset mid-combat and shared turns; never infer participants.
- [ ] Inspect callback order and generic/target spell pairs, compare outcome
  StoryActionIDs (including multi-target spells), and check when HP updates become
  visible. Immediate post-callback snapshots may need deferred capture after live
  evidence. Do not assume one interval equals one decision.
- [ ] Reload save, `reset`, stop/start; confirm new directories begin at sequence 1,
  no overwrites, contiguous sequences within each session, and no mixed sessions.

Unresolved blockers: no Windows game installation, untested mod packaging/loading,
actual extender API/signature availability, output-root/IO behavior, and real event
ordering/snapshot completeness. Richer resource/ability/participant/legality facts
remain deferred to later verified observation work.
