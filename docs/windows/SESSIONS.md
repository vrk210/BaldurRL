# Windows capture sessions

The session runner publishes and archives passive H0 collector output. It does
not run Baldur's Gate 3, choose actions, inject input, or turn observations into
an autonomous combat executor.

## Capture and archive roots

Pass the exact directory whose immediate children are collector GUID session
directories. Do not pass the general Script Extender directory or a game
installation. A typical verified-on-machine capture root will resemble:

```text
%LOCALAPPDATA%\Larian Studios\Baldur's Gate 3\Script Extender\BaldurRL\H0
```

The usual Script Extender root above is documentation, not a path the runner
guesses. Capture and archive roots must be separate and neither may contain the
other. Session and data-directory symlinks are rejected.

From the repository root in PowerShell, process the sessions currently present:

```powershell
python -m integration.session_runner `
  --capture-root "$env:LOCALAPPDATA\Larian Studios\Baldur's Gate 3\Script Extender\BaldurRL\H0" `
  --archive-root "D:\BaldurRL-captures" `
  --once `
  --game-version "OPERATOR_REPORTED_GAME_VERSION" `
  --se-version "OPERATOR_REPORTED_SE_VERSION"
```

Omit a version when it is unknown; the manifest stores `null`. Values supplied
with `--game-version`, `--se-version`, and `--code-revision` are labeled as
operator-reported. Without `--code-revision`, the runner safely asks the local
Git checkout for `HEAD` and records whether tracked or untracked changes are
present. Lookup failure remains `null`. A revision identifies the checkout base;
the collector source hashes identify the exact archived collector inputs.

For a continuing capture, omit `--once` and optionally change the polling rate:

```powershell
python -m integration.session_runner `
  --capture-root "C:\verified\Script Extender\BaldurRL\H0" `
  --archive-root "D:\BaldurRL-captures" `
  --poll-interval 0.5
```

Stop watch mode with Ctrl-C. Every poll discovers new session directories. Each
session keeps its own sequence namespace and its own cooperative lock. A second
runner fails visibly if `.baldurrl-session-runner.lock` exists. The runner never
deletes a stale lock: first confirm that no runner is using that source session,
then remove only that session's lock file manually.

## Publication and archive layout

For each `<session-id>`, the runner calls the existing `publish_ready` protocol
on `<capture-root>/<session-id>/staging` and publishes into that source session's
`records` directory. Completed malformed payloads, byte-count mismatches, and
differing immutable records are errors that name the session and path. A partial
marker or unmarked payload is incomplete and is not promoted.

The archive is isolated by session:

```text
<archive-root>/<session-id>/
  staging/                 completed .pending/.ready bytes
  records/                 immutable published JSON bytes
  diagnostics/
    health-v1.json
    incomplete/            content-addressed snapshots of incomplete bytes
    probes/                content-addressed probe JSON
  manifest/session-v1.json
```

Completed staging and records are copied byte-for-byte. An existing identical
copy is accepted; a differing copy at an immutable name stops the run. Incomplete
files can grow, so each observed version is archived under its SHA-256 rather
than overwriting an earlier diagnostic. The manifest binds the session ID to its
resolved source path before processing; the same archive session ID cannot be
combined with a different capture root.

The versioned manifest records origin paths, operator-reported versions, code
revision provenance, SHA-256 hashes of the H0 collector source and archived
bytes, and per-session file/publication statistics. It does not capture the
process environment, credentials, unrelated mods, general game directories, or
logs.

`health-v1.json` comes from one `RecordDirectoryReader.read_all()` collection
validated from expected sequence 1. It reports snapshot/event counts, first and
last sequence, gaps, latest record kind, latest snapshot sequence, snapshot-
reported controlled/selected and active entity known/unknown status, and counts
of observed resources, known abilities, and candidates. An empty candidate
array means candidate enumeration remains unknown; it does not mean no legal
actions. Health never infers legality, life state from HP, a complete participant
set, encounter completion, or selected identity from the host.

Offline schema validity is separate from live completeness and decision
readiness. The filesystem protocol does not provide exactly-once processing or
a power-loss durability guarantee. Validate local Windows replacement behavior,
collector completeness, and all live facts before relying on a capture.

## Read-only server probe

[`bg3/probes/server_inspect.lua`](../../bg3/probes/server_inspect.lua) is an
optional manual diagnostic. Supply the UUID of the actor actually selected in
the game and a session-relative diagnostics path. It enumerates loaded component
names, then dumps `SpellBook`, `SpellBookPrepares`, `ActionResources`, and
`TurnBased` only when present. API and query failures remain in its JSON and
console output. See [`bg3/probes/README.md`](../../bg3/probes/README.md) for the
call example.

The probe's pure Lua behavior is exercised offline with a Lua 5.4 stub harness.
It has been source-reviewed but not run in BG3 or on Windows. Live checks remain:

- verify the exact Script Extender IO root, reported game/extender versions, and
  `SaveFile` output;
- compare the explicitly supplied actor and component names with the UI;
- confirm component mapping and `DumpExport` behavior on the installed build;
- interrupt staging writes and verify partial data stays diagnostic-only;
- confirm atomic publication and immutable archive behavior on the target local
  Windows filesystem;
- reload/reset and create new sessions to verify independent sequence namespaces.
