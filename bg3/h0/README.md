# BaldurRL passive BG3 H0 collector

**Verified offline only; never loaded in BG3.** The
[live-verification checks](#live-verification-acceptance-checks-all-pending)
below establish what the game actually does. This source mod observes human gameplay on the
host server. No game-state writes, agent, input injection, training, or model
calls. The only writes are collector files. It is a source mod, not a tested
release `.pak`.

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
        Inspect.lua             # explicit actor diagnostic; !baldurrl_inspect
```

## First Windows session

This is the sole Windows setup/capture/live-verification runbook. Start from a
clean checkout and a disposable local save. Runtime capture tools use only the
Python standard library; do not install simulator/ML dependencies for capture.
The complete developer environment is documented in the root README.

1. Install Python 3.11 and the [Script Extender updater](https://github.com/Norbyte/bg3se/releases)
   (`DWrite.dll` in the game's `bin` directory); launch once to download SE.
   Merge `{"CreateConsole": true, "LogRuntime": true}` into
   `bin/ScriptExtenderSettings.json`, preserving existing settings. Record the
   game/SE versions. File presence does not establish that the extender loaded.
2. From the repository root in PowerShell, create the minimal environment and
   run the static preflight. Replace the example game path with the actual one:

   ```powershell
   py -3.11 -m venv .venv-harness
   $python = (Resolve-Path .\.venv-harness\Scripts\python.exe).Path
   .\scripts\windows\bootstrap-harness.ps1 -GameDir "D:\SteamLibrary\steamapps\common\Baldurs Gate 3" -InstallTestDependencies
   & $python -m pytest -q tests/test_bg3_h0_lua.py tests/test_integration_publish.py
   ```

   Preflight never installs the game/extender, changes load order or launches
   BG3. It may report missing setup before the remaining steps. Modern
   `Mods/ModuleShortDesc` and legacy `ModOrder` activation layouts are recognized;
   a listed UUID is static evidence only. The Python path above is used explicitly
   below, so activation/PATH changes are unnecessary.
3. Install [LSLib/Divine](https://github.com/Norbyte/lslib) separately. Package the
   contents of bg3/h0, with Mods at the package root:

   ```powershell
   .\scripts\windows\package-h0.ps1 -DivinePath "C:\Tools\LSLib\Divine.exe" -OutputPath "D:\BaldurRL\BaldurRL_H0.pak" -DryRun
   ```

   Remove `-DryRun` to build. A successful build writes a source/output SHA-256
   manifest; `-Force` explicitly permits replacing an existing package. Import
   it into [BG3 Mod Manager](https://github.com/LaughingLeader/BG3ModManager),
   activate and export load order. UUID: `d860477b-c599-4f12-9e95-5b8ec7235860`.
   Get-Help describes script options.

   **Fallback, unverified:** if packaging fails, copy `bg3/h0/Mods/BaldurRL_H0`
   to `<BG3>/Data/Mods/BaldurRL_H0` as loose files, a layout the
   [community setup example](https://github.com/LaughingLeader/BG3ModdingTools/wiki/Script-Extender-Lua-Setup)
   supports. The module must still be activated in the load order. Use the
   package or the loose copy, never both. Prefer the package: it is what
   `package-h0.ps1` builds and records in its manifest.
4. Load the save and confirm the server bootstrap, listener output and
   `[BaldurRL H0] New session:` GUID/path. Investigate UNAVAILABLE or HALTED.
   H0 starts on SessionLoaded/ResetCompleted; stop/start and reload create new
   namespaces beginning at sequence 1. Existing session directories are refused.
   Verify the printed IO root on this installation, then archive outside it:

   ```powershell
   $captures = Join-Path $env:LOCALAPPDATA "Larian Studios\Baldur's Gate 3\Script Extender\BaldurRL\H0"
   $archive = "D:\BaldurRL-captures"
   & $python -m integration.session_runner --capture-root "$captures" --archive-root "$archive"
   ```

   The usual SE root is under LOCALAPPDATA, not the updater cache or Script
   Extender Logs directory. Keep one runner/publisher per session. Failed
   sessions are quarantined with health/manifest/source hashes; healthy sessions
   continue. Investigate stale locks manually. --once processes all sessions and
   exits nonzero on failure. Raw staging is retained; recovery is not exactly-once.
5. Play a simple in-range encounter by hand. Compare selected actor, HP, IDs,
   rounds, turn owner and event order to the UI. In the SE server console use
   `!baldurrl_snapshot` for an extra snapshot and
   `!baldurrl_inspect <selected-actor-uuid>` for component diagnostics. Omit the
   UUID for help. The packaged probe lists component names and dumps SpellBook,
   SpellBookPrepares, ActionResources and TurnBased when present; dump_json is
   the raw JSON string from Ext.DumpExport. Missing facts remain unknown.
6. Record a bounded human trial using observed IDs, snapshot boundaries,
   measured resource deltas and explicit completion evidence:

   ```powershell
   & $python -m integration.encounter_trial init runs/trial.json --kind human_attack
   & $python -m integration.encounter_trial analyze runs/trial.json --records "$archive\<session-guid>\records"
   ```

   H0 omits resources/owned abilities; live resource checks remain INCONCLUSIVE
   until verified enrichment exists. Preserve the human trace and probe output.
   A miss can still be execution; command acceptance alone proves no execution.
7. Use `!baldurrl_stop`, stop the runner with Ctrl-C, then make a final sweep:

   ```powershell
   & $python -m integration.session_runner --capture-root "$captures" --archive-root "$archive" --once
   & $python -m integration.inspect_records "$archive\<session-guid>\records"
   ```

   Inspect one session's records directory, never staging or a multi-session
   parent. Reload/reset or `!baldurrl_start` creates a fresh namespace, never a
   continuation of an old sequence. Module --help contains detailed options.

Offline CI checks Lua with stubbed game APIs, Python on Windows 3.11/3.13 and
fake-converter packaging; everything else needs the live checks below. The ready
marker/publication contract is in [BG3_INTEGRATION.md](../../docs/BG3_INTEGRATION.md).

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

Install the pinned minimal manifest and use this environment's Python:

```text
python -m pip install -r requirements-harness-test.txt
python -m pytest tests/test_bg3_h0_lua.py tests/test_integration_publish.py
```

The Windows workflow also runs every integration test and the preflight,
session-runner and trial tests. The root README gives the pinned full-suite setup.

`test_bg3_h0_lua.py` uses the **real Lua 5.4 runtime** in `lupa.lua54`, loading the
actual mod scripts with stubbed Ext/Osiris calls and a real temporary filesystem.
It checks the event trace through `parse_record`, round-trips all produced records,
inspects it with the existing reader, checks session resets, unknown facts,
shared turns, JSON escaping, and failed writes. These tests explicitly skip if
Lupa is unavailable; a skip is not Lua validation. Its metadata/layout test and
`test_integration_publish.py` are **Python/static-only** tests; they do not execute
Lua. No tests emulate engine timing or prove an installed SE version works.

## Live-verification acceptance checks (all pending)

- [ ] Record BG3/SE versions; confirm updater, console, package activation, server
  bootstrap, all 12 listener registrations, and session GUID generation.
- [ ] Confirm exact IO root and UTF-8 bytes; test `.pending` readback, `.ready`
  completion, publisher atomic replacement, immutable `.json` records, and
  `python -m integration.inspect_records <directory>`. Interrupt a write: no partial
  completed JSON; an interrupted capture may require a fresh session. Measure
  the delay between a record's `.ready` marker and the publisher picking it up.
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

- [ ] Compare selected host and enemy component dumps: HP/max HP, armor class,
  position, alive/dead/downed state. Check that the Osiris HP/max HP in
  snapshots matches the ECS values. HP=0 alone does not establish death.
- [ ] Identify the Basic Attack, Second Wind and Action Surge ability IDs, the
  Action/Bonus Action/Movement/slot resource IDs, and how Second Wind / Action
  Surge availability is represented; compare SpellBook, HasSpell and visible UI
  abilities.
- [ ] Compare candidate legality to enabled UI actions, including range, LOS and
  status restrictions, and test `CanCastSpellOnEnemyInSameCombat`; an
  absent/unknown result never becomes LEGAL.
- [ ] For future command trials, establish capture-after-terminal-result timing,
  excluding queued old snapshots. Acceptance and callback start are insufficient.

Keep unique live captures under one ignored/external session archive; record its
location, game/SE/code versions and retention condition in the results summary.
Inspect anomalies before summarizing or retiring evidence; preserve unresolved
examples rather than retaining duplicate raw bundles indefinitely.
