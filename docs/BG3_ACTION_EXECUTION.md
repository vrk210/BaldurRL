# BG3 combat action execution: feasibility research

Research date: 2026-10-09. This is documentation only; no executor, native hook,
queue mutation, or BG3 live test was implemented or run. Read alongside
[BG3_INTEGRATION.md](BG3_INTEGRATION.md): observation collection and H3 execution
remain separate, and unknown legality must remain unknown.

## Conclusion and evidence levels

The best first experiment is **the normal client UI/input path**, using Script
Extender's public keyboard injection API where it can perform the observed human
interaction. Use **Windows keyboard/mouse input automation** as the fallback if
target selection or confirmation needs a mouse, or injected SE keys do not work.
Both paths need live validation; neither has been demonstrated in BaldurRL.

There is no dedicated public Lua cast-request submission function identified in
the inspected source. However, the Lua array/conversion implementation and Brawl
provide evidence that Lua tables can become cast-request array entries. It would
be incorrect to conclude either that requests are impossible to submit from Lua,
or that this constitutes a supported, safe action API. Do not adopt this internal
queue route under the present task's restrictions.

Evidence labels used below:

- **DOCUMENTED:** explicit upstream API documentation or release notes.
- **SOURCE:** directly visible behavior of the pinned binding implementation or
  mod source; narrower than a promise about the closed-source game.
- **INFERRED:** proposed interpretation of names, structures, or routing.
- **LIVE-UNVERIFIED:** requires this project's Windows BG3 session. This label
  can accompany any of the other three; source evidence is not a live test.

Norbyte source is pinned to
[`91abbf230d02faa2b093d0a458d9caf6d1494901`][se-revision]. A checkout of `main`
is not proof of what a particular installed SE binary supports. Record the game
and SE versions before testing. Wiki references use the inspected revision where
available; source references below link to exact commits and line ranges.

## Execution-path comparison

| Path | Evidence and rules implications | Suitability for BaldurRL |
| --- | --- | --- |
| `Osi.UseSpell` | DOCUMENTED call, explicitly bypasses preconditions, including spell access and insufficient resources [U1]. Normal affordability/rejection cannot be assumed. | Reject as the ordinary-action executor. A visual attack alone would not validate it. |
| `Ext.System.ServerCastRequest` internal arrays | SOURCE: mapped system, generic array writes and table conversion exist; Brawl uses them [S1–S8, M1]. Queue-specific validation and lifetime contract are not established. | Research/inspection only. No internal queue writes. |
| SE client keyboard injection | DOCUMENTED since v22; SOURCE: client-only key down/up event injection [I1–I3]. INFERRED: a correctly routed UI action should go through normal game validation. | Recommended first public-API experiment. Complete autonomous targeting remains LIVE-UNVERIFIED. |
| Windows keyboard/mouse automation | DOCUMENTED OS input synthesis [W1]; BG3 acceptance and UI behavior LIVE-UNVERIFIED. | Fallback for full targeting/confirmation; requires stable focus, camera, coordinates, and bindings. |
| Noesis UI commands | SOURCE: `CanExecute`/`Execute` wrappers exist [I5–I6]. No verified attack command, target argument, or UI object path identified. | Secondary research lead, not a ready semantic attack API. |
| Native combat AI delegation | UTAC source uses control-loss statuses/archetypes and optionally `MakeNPC` [M3–M4]. Ordinary native behavior is plausible but LIVE-UNVERIFIED here. | Useful precedent; not an executor for an exact policy-chosen ability/target. |

The bracketed source IDs refer to the reference index at the end.

## `Osi.UseSpell`: callable does not mean rule-respecting

**DOCUMENTED:** the official [UseSpell page][use-spell] describes target-based
casting, optional second target, and optional movement suppression. Its caveat
says Osiris casting bypasses preconditions, including not knowing the spell and
not having resources. `WithoutMove` only concerns moving into range; the page
does not offer a flag that restores normal player preconditions [U1].

**LIVE-UNVERIFIED:** that caveat does not precisely specify which counters are
deducted in each successful cast, or every range/LOS/status restriction. Do not
convert it into a claim that every cast is always free, or that every target is
accepted. It establishes that the API cannot be trusted to reject an action for
insufficient resources. Actual Action/Bonus Action/slot deductions would require
measurements. Manually charging resources or recreating legality would not
establish equivalence with BG3's ordinary action pipeline.

**Decision:** do not use `UseSpell` for the valid/invalid ordinary-attack trials.
Also, `CastSpell` linked in the wiki is an **event**, not an alternative command;
`CastSpellFailed` is a failure event [U3, U5].

## Cast requests: exposure, conversion, submission, and safety

### What is exposed

**DOCUMENTED:** v25 introduced `Ext.System` and mapped `ServerCastRequest` [S1].
**SOURCE:** the system-map indexer returns a reference to a registered ECS system;
it is not a cast-submission call [S2]. IDE metadata exposes
`EsvSpellCastCastRequestSystem` and `EsvSpellCastCastStartRequest` [S4].

**SOURCE:** `CastStartRequest` includes `Spell`, `CastOptions`, `Caster`,
`Targets`, `Originator`, `Item`, `CastPosition`, `StoryActionId`, `NetGuid`,
`RequestGuid`, and unnamed fields. Some fields have C++ defaults, including
`CastOptions = 0`, `StoryActionId = 0`, and `field_A8 = 1` [S3]. These defaults
are not a documented minimal valid request.

`Spell` is a composite identity, not just the spell's stats name: its definition
includes originator prototype, source type, source GUID, progression source GUID,
and prototype [S9]. Targets include an entity/optional position and a targeting
type, with additional optional target data [S10]. Do not invent values for these
fields or assume an observed request can be replayed with stale identifiers.

### The requested queues

**SOURCE:** `CastRequestSystem` contains separate `Array<CastStartRequest>`
members named `AnubisCastRequests`, `NetworkStartRequests`, and
`CommandProtocolStartRequests`, as well as Osiris, reaction, item, jump, and
other request arrays [S3].

| Array | What the source establishes | What remains inferred/unverified |
| --- | --- | --- |
| `AnubisCastRequests` | Named start-request array. | Likely associated with Anubis behavior/AI. Exact producer and validation policy not shown. |
| `NetworkStartRequests` | Separate named start-request array. | Likely related to client/network requests. Name does not prove permission to fabricate a player request or that it performs all player checks. |
| `CommandProtocolStartRequests` | Separate named start-request array. | Likely associated with game command routing. Not evidence of a callable public command protocol. |

The extender definitions map layouts; they do not contain the full closed-source
game implementations that consume these queues. Which array a human attack
visits, when it is drained, and how start/preview/confirm requests cooperate are
**LIVE-UNVERIFIED**. Sampling an empty array between frames proves little.

### Can Lua actually create and submit them?

There are three different questions:

1. **Standalone construction:** `Ext.Types.Construct` is exported, but its
   inspected implementation checks type/constructibility and ends in an
   unfinished branch returning no Lua values [S5]. It is not a usable factory
   for a standalone cast-request userdata in this revision. A type-info entry
   or IDE class declaration does not prove construction works.
2. **Conversion during array assignment:** the binding provides more than a
   layout. `Array<T>` uses `DynamicArrayProxyImpl`; for default-constructible
   elements its setter converts a Lua value using `get<T>` and can append at
   the next index [S6]. Non-by-value conversion initializes a C++ value and
   unserializes it [S7]. Generic object unserialization accepts table fields via
   property mappings [S8]. **SOURCE/INFERRED:** this supports table-to-request
   conversion through exposed arrays, subject to each field's mapping and the
   installed build. It does not use the unfinished standalone factory.
3. **Supported submission and game correctness:** no dedicated `SubmitCast`,
   `StartCast`, or equivalent cast-request method was identified in the system
   declarations, IDE entries, and Lua libraries inspected. Brawl does use array
   assignment to submit requests [M1], so a categorical claim that Lua cannot
   enqueue one would contradict available evidence. That mechanism mutates
   engine-owned queues and has no demonstrated public contract for safe timing,
   invariants, ownership, complete legality, or future-version compatibility.

**SOURCE:** option mappings include `IgnoreHasSpell`, `IgnoreCastChecks`,
`IgnoreTargetChecks`, `Forced`, and `FromClient`; failure reasons include
`CantCast`, `InvalidTarget`, and `CantSpendUseCosts` [S11]. These names reveal
possible bypasses and failures. They do **not** prove what a particular flag or
queue enforces. In particular, setting `FromClient` is not proof of player-equivalent
authorization or resource handling.

**Decision:** read these queues only. Do not append, replace, clear, reorder,
unserialize into, or retain engine references for later replay. A dedicated
supported API or a separately authorized, carefully evidenced investigation
would be needed before reconsidering this route.

## Public client input and fallback

**DOCUMENTED:** v22 lists `Ext.Input.InjectKeyPress`, `InjectKeyDown`, and
`InjectKeyUp`; v23 adds input-manager and character input-controller mappings
[I1]. **SOURCE:** these injection functions are registered in the client context.
They take an SDL scan code and optional modifier value. `InjectKeyPress` produces
one down event followed by one up event; the other calls produce a single event
[I2]. The SDL manager buffers injected events and returns them through its poll
wrapper [I3]. This is evidence for actual input injection, not just mapped data.

The input module's exported list has only those three injectors and
`GetInputManager` [I2, I4]. No mouse-position, mouse-click, or controller-event
injector was identified in that module. `GetInputManager` and
`ClientCharacter.InputController` expose state; do not treat mapped internal
controllers as documented action methods. A hotbar shortcut may only enter
targeting mode. Target selection and confirmation are a separate feasibility
gate, not an assumed capability of `InjectKeyPress`.

**SOURCE:** client events include `KeyInput`, `MouseButtonInput`,
`MouseWheelInput`, `ControllerAxisInput`, and `ControllerButtonInput` [I7].
These observe input; they are not corresponding injection methods. They can help
record the human control sequence. The inspected poll wrapper returns buffered
injected events before its normal SE event-processing branch [I3], so do not
require a `KeyInput` callback as proof that a synthetic key was delivered.
Observe the UI and game outcome instead.

**SOURCE:** Noesis command wrappers expose `CanExecute` and `Execute` [I5–I6].
The execution wrapper does not itself call `CanExecute`. An actual attack UI
command, correct argument object, thread/context, and rejection behavior would
need identification. Do not call arbitrary UI commands during this research.

**Recommendation:** reproduce the human's UI interaction with public input,
then measure the resulting action. If keyboard-only targeting cannot be shown,
use the fallback immediately rather than fabricate an engine request.
**DOCUMENTED:** Windows `SendInput` supports keys, pointer movement, and clicks;
it returns a count of inserted input events and is subject to integrity-level
restrictions [W1]. **LIVE-UNVERIFIED:** an inserted event count is not a BG3
action acknowledgment. Screen targeting, focus, key timing, UI overlays, and
resolution/camera changes must be tested. This fallback requires a later Windows
input driver; none is implemented here.

## Existing mod precedents

### Brawl: useful submission evidence, limited rules evidence

Inspected [`tinybike/Brawl` at `1ac3c7c1a6bc01cdc21cb9ec8a0f857210f0ef31`][brawl].
Its action code builds Lua request tables and assigns them into
`ServerCastRequest.OsirisCastRequests` [M1]. Player requests use prepared-spell
source metadata when found and a `FromClient` option; NPC/Hogwild branches use
bypass options. Comments attribute player resource/cooldown handling to the game,
but this is a mod author's assertion, not an upstream guarantee [M1].

The same file also has manual extra-attack damage and resource handling [M2].
Consequently, Brawl is concrete open-source evidence of a deployed request-writing
approach, **not** proof that the approach preserves all ordinary turn-based rules.
Do not copy its queue-writing code into BaldurRL.

### UTAC: native action selection, not exact policy execution

Inspected [`Vercadi/Ultimate-Tactical-AI-Companions` at
`ea486a3a0caa62552f6672a2114c7843bb2860e1`][utac]. Combat statuses specify native
AI archetype overrides and control-loss flags [M3]; Lua applies the selected
status and optionally calls `MakeNPC`, with a host/player exclusion [M4].
This is a public source precedent for handing combat to native AI rather than
forcing individual casts. **INFERRED/LIVE-UNVERIFIED:** native AI should use its
usual action pipeline, but this study did not verify its resources or invalid
action rejection. It also chooses the action itself, so it cannot be assumed to
execute BaldurRL's specific ability/target intent.

No inspected open-source example establishes an arbitrary Lua ability/target
submit API with proven normal checks, costs, and lifetime safety. This is a
bounded research result, not a claim that no such mod exists. Party Tactics was
also checked, but its public repository explicitly says the development source
is private; its release descriptions cannot supply source-level execution proof
[M5].

## Read-only Windows Lua inspection

Run in SE's console on a loaded, disposable single-player test save. The console
selects contexts with the bare commands `server` and `client` [R1]. For each
multiline block, enter `--[[`, paste the block, then finish with `]]--` [R1].
The snippets below read state or install/remove logging callbacks; they do not initiate game
actions or write ECS state. Lua globals hold inspection settings only. Logging
subscriptions are optional instrumentation and should be removed afterward.
These snippets have been source-reviewed, **not executed in BG3**. Read-only
access is not a guarantee against version-specific mapping faults; stop on errors.

### Server: build, entities, abilities, and resources

First type `server`, then run blocks separately. The build getters are exported
in Utils [R2]. `GetAllComponentNames` is documented [R3]. Host lookup uses the
same method as SE's server `_C()` helper [R4]. Dump helpers serialize for
console output [R11]; component names used below are present in the source
metadata, but their presence on a live actor is unverified [R12].

```lua
Ext.Utils.Print("SE", Ext.Utils.Version(), "BG3", Ext.Utils.GameVersion())
BRL_ACTOR = Osi.GetHostCharacter()
Ext.Utils.Print("host", BRL_ACTOR)
local actor = Ext.Entity.Get(BRL_ACTOR)
if actor then Ext.Dump(actor:GetAllComponentNames()) end
```

```lua
local actor = Ext.Entity.Get(BRL_ACTOR)
if actor then
    Ext.Dump(actor.SpellBook)
    Ext.Dump(actor.SpellBookPrepares)
    Ext.Dump(actor.ActionResources)
    Ext.Dump(actor.TurnBased)
end
```

Do not assume a property exists on this entity/build: compare the enumerated
component names first. Verify the controlled combatant rather than equating host
with current selection. Set the enemy to an actual UUID obtained in the session:

```lua
BRL_TARGET = "REPLACE_WITH_OBSERVED_ENEMY_UUID"
local target = Ext.Entity.Get(BRL_TARGET)
if target then Ext.Dump(target:GetAllComponentNames()) end
```

After discovering the real ability and resource names/levels, query them without
changing them. Resource query signature and level semantics are documented [U2];
the Lua bridge returns Osiris query outputs [R5]. Ownership and HP queries are
documented separately [U6–U7]; neither establishes attack legality.

```lua
BRL_ABILITY = "REPLACE_WITH_OBSERVED_ATTACK_ID"
BRL_RESOURCE = "REPLACE_WITH_OBSERVED_RESOURCE_NAME"
BRL_RESOURCE_LEVEL = 0 -- replace if the observed resource uses another level
Ext.Utils.Print("owns ability", Osi.HasSpell(BRL_ACTOR, BRL_ABILITY))
Ext.Utils.Print("resource", Osi.GetActionResourceValuePersonal(
    BRL_ACTOR, BRL_RESOURCE, BRL_RESOURCE_LEVEL))
Ext.Utils.Print("actor HP", Osi.GetHitpoints(BRL_ACTOR))
Ext.Utils.Print("target HP", Osi.GetHitpoints(BRL_TARGET))
```

### Server: system/type inspection and short human-action trace

Type getters return metadata; they are not construction probes [R6]. Queue reads
and type names follow the exposed system metadata [S2, S4].

```lua
local system = Ext.System.ServerCastRequest
if system then
    Ext.Utils.Print("system type", Ext.Types.GetObjectType(system))
    Ext.DumpShallow(system)
end
Ext.Dump(Ext.Types.GetTypeInfo("EsvSpellCastCastStartRequest"))
```

A plain console dump can miss transient requests. Optionally subscribe briefly
to public system pre/post-update notifications [R7–R8], and perform **one human
attack** while the subscriptions are active. The callback reacquires the system
each time and dumps nonempty arrays immediately; it does not keep request proxies.
SE documents scope-limited object lifetimes [R10].
Seeing a request establishes observation only, not replay suitability.

```lua
BRL_READ_QUEUES = function(phase)
    local system = Ext.System.ServerCastRequest
    if not system then return end
    for _, name in ipairs({"AnubisCastRequests", "OsirisCastRequests",
        "NetworkStartRequests", "CommandProtocolStartRequests"}) do
        local requests = system[name]
        if requests and #requests > 0 then
            Ext.Utils.Print(phase, name, #requests)
            Ext.Dump(requests)
        end
    end
end
BRL_PRE = Ext.Entity.OnSystemUpdate("ServerCastRequest", function()
    BRL_READ_QUEUES("pre")
end)
BRL_POST = Ext.Entity.OnSystemPostUpdate("ServerCastRequest", function()
    BRL_READ_QUEUES("post")
end)
```

Type `exit` to return the console to log mode before the human attack [R1].
Remove subscriptions promptly after the trial to limit console overhead:

```lua
if BRL_PRE then Ext.Entity.Unsubscribe(BRL_PRE); BRL_PRE = nil end
if BRL_POST then Ext.Entity.Unsubscribe(BRL_POST); BRL_POST = nil end
BRL_READ_QUEUES = nil
```

If callbacks never see a request, record that sampling limitation; do not infer
the human action avoided these systems. Use a separate passive event logger for
`UsingSpellOnTarget`, completion, failure, and attack-result events. The former
has six parameters including StoryActionID [U4]; `CastSpellFailed` has five [U3].
The integration contract's event correlation remains subject to live verification.

### Client: available input bindings and state

Type `client` first. Inspect availability without calling an injector:

```lua
Ext.Utils.Print("press", type(Ext.Input.InjectKeyPress),
    "down", type(Ext.Input.InjectKeyDown), "up", type(Ext.Input.InjectKeyUp))
Ext.Dump(Ext.Types.GetTypeInfo("InputInputManager"))
Ext.DumpShallow(Ext.Input.GetInputManager())
local controlled = _C()
if controlled then
    Ext.Dump(controlled:GetAllComponentNames())
    if controlled.ClientCharacter then
        Ext.DumpShallow(controlled.ClientCharacter.InputController)
    end
end
```

SE's client `_C()` selects a control entity for reserved user 1 [R9]; confirm it
matches the visible Fighter, especially with split-screen/co-op. These checks do
not establish that an injected key is delivered to the correct action UI. No
injection or queue-mutation commands are included in this research document.

## Live validation plan (future Windows session)

1. **Prepare and identify.** Record BG3 build, SE version, OS, input mode, active
   mods, and keybindings. Use a separate save with an in-range 1v1 turn-based
   encounter, a living selected Fighter, and no Extra Attack/free-attack feature
   for the initial test. Disable combat-changing mods. Discover actual actor,
   target, attack, resource, and spell-source IDs; confirm whose turn it is and
   that the normal UI enables the attack. Do not substitute simulator IDs.
2. **Inspect one human attack.** Take a pre-action resource/HP/turn snapshot;
   enable the short read-only queue trace and passive events. Human-select the
   attack and target through the ordinary UI. Record each input, targeting and
   confirmation step, any observed requests/flags/source metadata, StoryActionID,
   hit/miss/completion, and post-action state before the next turn refresh. A
   miss still counts as an executed attack. Stop instrumentation and retain logs.
3. **Establish a complete input route.** Reload that save. First establish whether
   the observed interaction is fully expressible using SE key injection. If a
   hotbar key merely opens targeting, mark this as partial feasibility. Use
   Windows pointer/click automation for the missing steps, or the entire input
   sequence if SE delivery fails. Keep all routing through the normal UI. This
   is a future execution experiment, not permission to mutate a cast queue.
4. **Attempt exactly one valid autonomous attack.** From the same preconditions,
   issue the chosen input sequence without a human choosing/confirming the
   target during the trial. Emit a local attempt ID and record it alongside the
   resulting game events; do not equate it with StoryActionID or RequestGuid.
   Require the correct actor, ability, and target and one completed attack or
   miss. An accepted input event, UI animation, or nonempty queue is insufficient.
   Use a bounded wait; do not retry blindly and risk duplicate actions.
5. **Check resource consumption.** Compare before/after Action, Bonus Action,
   movement, relevant slots, and ability-specific resources against the human
   control, measured before refresh. For this simple trial, the same ordinary
   attack must incur the same costs even on a miss; record actual measured
   deltas rather than hardcoding simulator assumptions. Check that no extra
   attack or turn end occurred and that no manual deduction/restore was used.
   If costs differ, fail this execution-path trial.
6. **Test rejection of an invalid attack.** Without advancing the turn after
   consuming the attack's available Action resource, confirm the ordinary UI
   rejects another attack. Remove confounders such as Extra Attack, Action Surge,
   haste, or a free attack. Submit the same autonomous UI attempt once. Require
   no second completed attack/damage and no inappropriate resource change;
   record the visible rejection and any failure event. If the UI blocks it
   before a server request exists, record **UI-layer rejection**, not proof of
   server-side validation. Compare a human attempt in the same invalid state.
7. **Broaden only after passing.** Repeat with range/LOS, invalid target, wrong
   turn, insufficient spell slots, and a status-based restriction in separate
   controlled trials. One valid attack plus one resource rejection validates
   only that scenario/build. Preserve uncertainty for every untested condition.

Store trial ID, versions, selected path, initial state, input sequence/timing,
observed request data (if any), events, final state, resource deltas, rejection
layer, and pass/fail reason. Keep policy decisions and legality evidence separate
from action outcomes as required by BG3_INTEGRATION.md. No live result exists yet.

## Questions requiring BG3

- Can public SE keyboard injection complete target selection and confirmation,
  or does it only enter targeting mode? What timing/focus conditions work?
- Which queues and source/option fields appear for a human basic attack, and can
  the available read-only callbacks observe them reliably?
- Does the installed binary expose the same types and array behavior as the
  pinned source, and what faults or lifetime restrictions occur during reads?
- Does an input-driven autonomous attack consume exactly the human-control
  resources, including miss, reactions, cooldowns, and special resource types?
- At which layer are invalid attacks rejected, and which events identify that
  rejection without confusing an attack miss, interruption, or timeout?
- Can Windows input select the intended entity reliably with camera/UI changes,
  and how are duplicates and completion recognized?
- What precisely does `UseSpell` deduct despite bypassing preconditions? This
  is diagnostic research only, not a reason to adopt it for ordinary actions.

## Exact reference index

### Official game documentation

- **U1:** [UseSpell, revision 2261][use-spell] — description, optional parameters,
  and precondition caveat.
- **U2:** [GetActionResourceValuePersonal, revision 3198][resource-query] — query
  arguments, resource names, levels, and amount.
- **U3:** [CastSpellFailed, revision 2273][cast-failed] — failure-event signature
  and relationship to CastSpell.
- **U4:** [UsingSpellOnTarget, revision 2285][spell-event] — target/use event and
  StoryActionID.
- **U5:** [CastSpell, revision 2271][cast-start] — cast-start event.
- **U6:** [HasSpell, revision 2293][has-spell] — ownership query.
- **U7:** [GetHitpoints, revision 1673][hitpoints] — remaining HP query.

### Norbyte source and documentation (pinned revision)

- **S1:** [ReleaseNotes.md, v25, lines 153–157][systems-release].
- **S2:** [LuaSystemMap.inl, lines 5–26][system-map] and
  [ExposedSystemTypes.inl, line 25][system-exposed].
- **S3:** [SpellCast.h, request lines 260–274][request-layout],
  [system lines 500–525][queue-layout], and
  [per-entity requests lines 322–330][entity-requests].
- **S4:** [ExtIdeHelpers.lua, lines 16732–16778][cast-metadata].
- **S5:** [Types.inl, lines 286–303][construct].
- **S6:** [LuaArrayProxy.h, setter lines 65–94][array-setter] and
  [Array specialization lines 533–543][array-specialization];
  [LuaArrayProxy.inl, lines 45–57][array-dispatch].
- **S7:** [LuaGetObject.h, lines 5–11][value-conversion].
- **S8:** [LuaUnserialize.h, lines 522–587][unserialize] and
  [LuaPropertyMap.inl, lines 349–371][property-conversion].
- **S9:** [ExposedTypes.h, lines 18–48][spell-identity].
- **S10:** [SpellCastShared.h, lines 43–69][target-layout].
- **S11:** [ECS.inl, lines 2195–2240][cast-options].
- **I1:** [ReleaseNotes.md, v22 lines 229–230][input-release] and
  [v23 lines 195–202][input-mapping-release].
- **I2:** [ClientInput.inl, lines 6–53][input-api].
- **I3:** [SDL.cpp, lines 70–96][sdl-input].
- **I4:** [ExtIdeHelpers.lua, lines 24095–24100][input-metadata].
- **I5:** [UI.inl, lines 95–106][ui-command-map].
- **I6:** [NsHelpers.inl, lines 660–678][ui-command-impl].
- **I7:** [BuiltinLibraryClient.lua, published events lines 3–9][input-events]
  and [LuaClient.cpp, input dispatch lines 148–201][input-event-dispatch].
- **R1:** [API.md, console/multiline lines 126–166][console].
- **R2:** [Utils.inl, lines 79–96][version-getters] and
  [registration lines 230–231][version-registration].
- **R3:** [API.md, component enumeration lines 800–808][components].
- **R4:** [BuiltinLibraryServer.lua, lines 51–53][host-helper].
- **R5:** [API.md, Osiris calls/queries][osiris-lua].
- **R6:** [Types.inl, lines 81–99 and 127–135][type-inspection].
- **R7:** [Entity.inl, lines 222–240][system-observers].
- **R8:** [SystemEvents.inl, pre/post dispatch lines 103–145][observer-dispatch].
- **R9:** [BuiltinLibraryClient.lua, lines 28–36][client-helper].
- **R10:** [API.md, object scopes lines 194–218][object-lifetime].
- **R11:** [BuiltinLibrary.lua, lines 29–59][dump-helpers].
- **R12:** [ExtIdeHelpers.lua, component names lines 26395–26423][state-components]
  and [TurnBased name, line 26660][turn-component].

### Mod/OS evidence

- **M1:** [Brawl Actions.lua, lines 218–315][brawl-submit] — spell identity,
  queue assignment, options, and request table.
- **M2:** [Brawl Actions.lua, lines 84–111][brawl-custom-rules] — manual damage
  and resource modification; limits the ordinary-rules claim.
- **M3:** [UTAC Status_Combat.txt, lines 1–13][utac-status] — native archetype
  and control-loss configuration.
- **M4:** [UTAC AIArchetypeManager.lua, lines 1299–1344][utac-control] — combat
  status application and NPC-mode/host handling.
- **M5:** [Party Tactics README, Source section][party-tactics] — public
  distribution repository, private development source.
- **W1:** [Microsoft SendInput documentation][windows-input] — input synthesis,
  return value, and integrity restrictions; no BG3-specific guarantee.

[se-revision]: https://github.com/Norbyte/bg3se/commit/91abbf230d02faa2b093d0a458d9caf6d1494901
[use-spell]: https://docs.baldursgate3.game/index.php?title=UseSpell&oldid=2261
[resource-query]: https://docs.baldursgate3.game/index.php?title=GetActionResourceValuePersonal&oldid=3198
[cast-failed]: https://docs.baldursgate3.game/index.php?title=CastSpellFailed&oldid=2273
[spell-event]: https://docs.baldursgate3.game/index.php?title=UsingSpellOnTarget&oldid=2285
[brawl]: https://github.com/tinybike/Brawl/tree/1ac3c7c1a6bc01cdc21cb9ec8a0f857210f0ef31
[utac]: https://github.com/Vercadi/Ultimate-Tactical-AI-Companions/tree/ea486a3a0caa62552f6672a2114c7843bb2860e1
[party-tactics]: https://github.com/seyelive5/bg3-companion-ai/blob/76df219478245e2b34824d0a1ceb1ae1d53ef58e/README.md#소스--source
[windows-input]: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
[systems-release]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/ReleaseNotes.md#L153-L157
[system-map]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/Proxies/LuaSystemMap.inl#L5-L26
[system-exposed]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Components/ExposedSystemTypes.inl#L25-L25
[request-layout]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Components/SpellCast.h#L260-L274
[queue-layout]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Components/SpellCast.h#L500-L525
[entity-requests]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Components/SpellCast.h#L322-L330
[cast-metadata]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/IdeHelpers/ExtIdeHelpers.lua#L16732-L16778
[construct]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/Types.inl#L286-L303
[array-setter]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/Proxies/LuaArrayProxy.h#L65-L94
[array-specialization]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/Proxies/LuaArrayProxy.h#L533-L543
[array-dispatch]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/Proxies/LuaArrayProxy.inl#L45-L57
[value-conversion]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Helpers/LuaGetObject.h#L5-L11
[unserialize]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Helpers/LuaUnserialize.h#L522-L587
[property-conversion]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/Proxies/LuaPropertyMap.inl#L349-L371
[spell-identity]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Base/ExposedTypes.h#L18-L48
[target-layout]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Components/SpellCastShared.h#L43-L69
[cast-options]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/Enumerations/ECS.inl#L2195-L2240
[input-release]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/ReleaseNotes.md#L229-L230
[input-mapping-release]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/ReleaseNotes.md#L195-L202
[input-api]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/ClientInput.inl#L6-L53
[sdl-input]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Extender/Client/SDL.cpp#L70-L96
[input-metadata]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/IdeHelpers/ExtIdeHelpers.lua#L24095-L24100
[ui-command-map]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/GameDefinitions/PropertyMaps/UI.inl#L95-L106
[ui-command-impl]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/ClientUI/NsHelpers.inl#L660-L678
[console]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/API.md#L126-L166
[version-getters]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/Utils.inl#L79-L96
[version-registration]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/Utils.inl#L230-L231
[components]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/API.md#L800-L808
[host-helper]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/LuaScripts/BuiltinLibraryServer.lua#L51-L53
[osiris-lua]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/API.md#L476-L510
[type-inspection]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/Types.inl#L81-L135
[system-observers]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Libs/Entity.inl#L222-L240
[observer-dispatch]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Shared/SystemEvents.inl#L103-L145
[client-helper]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/LuaScripts/BuiltinLibraryClient.lua#L28-L36
[object-lifetime]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/Docs/API.md#L194-L218
[dump-helpers]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/LuaScripts/BuiltinLibrary.lua#L29-L59
[state-components]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/IdeHelpers/ExtIdeHelpers.lua#L26395-L26423
[brawl-submit]: https://github.com/tinybike/Brawl/blob/1ac3c7c1a6bc01cdc21cb9ec8a0f857210f0ef31/Brawl/Mods/Brawl/ScriptExtender/Lua/Server/Actions.lua#L218-L315
[brawl-custom-rules]: https://github.com/tinybike/Brawl/blob/1ac3c7c1a6bc01cdc21cb9ec8a0f857210f0ef31/Brawl/Mods/Brawl/ScriptExtender/Lua/Server/Actions.lua#L84-L111
[utac-status]: https://github.com/Vercadi/Ultimate-Tactical-AI-Companions/blob/ea486a3a0caa62552f6672a2114c7843bb2860e1/UTAC/Public/UTAC/Stats/Generated/Data/Status_Combat.txt#L1-L13
[utac-control]: https://github.com/Vercadi/Ultimate-Tactical-AI-Companions/blob/ea486a3a0caa62552f6672a2114c7843bb2860e1/UTAC/Mods/UTAC/ScriptExtender/Lua/Server/AIArchetypeManager.lua#L1299-L1344
[turn-component]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/IdeHelpers/ExtIdeHelpers.lua#L26660
[cast-start]: https://docs.baldursgate3.game/index.php?title=CastSpell&oldid=2271
[has-spell]: https://docs.baldursgate3.game/index.php?title=HasSpell&oldid=2293
[hitpoints]: https://docs.baldursgate3.game/index.php?title=GetHitpoints&oldid=1673
[input-events]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/LuaScripts/BuiltinLibraryClient.lua#L3-L9
[input-event-dispatch]: https://github.com/Norbyte/bg3se/blob/91abbf230d02faa2b093d0a458d9caf6d1494901/BG3Extender/Lua/Client/LuaClient.cpp#L148-L201
