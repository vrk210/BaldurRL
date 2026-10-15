# Read-only BG3 server probe

`server_inspect.lua` is a source-reviewed, live-unverified Script Extender server
console probe. It reads the explicitly supplied actor, reports component names,
and dumps only components that the actor reports as present. It does not select
an actor, infer selection from the host, mutate ECS state, inject input, or touch
cast-request queues. Engine userdata stays within one function call.

On a disposable local save, select `server` in the Script Extender console, load
the file using the console's supported development workflow, and call:

```lua
BaldurRL_ServerInspect(
    "REPLACE_WITH_ACTUALLY_SELECTED_ACTOR_UUID",
    "BaldurRL/H0/REPLACE_WITH_SESSION_GUID/diagnostics/server-inspect.json"
)
```

Both arguments are operator choices. Obtain the selected actor UUID from live
inspection; the probe contains no entity, ability, spell, or resource IDs. The
output path is relative to Script Extender's IO root, usually
`%LOCALAPPDATA%\Larian Studios\Baldur's Gate 3\Script Extender`, which must be
verified on the installation. `Ext.IO.SaveFile` creates missing parent folders.

The probe always prints its JSON diagnostic. When `Ext.IO.SaveFile` is present,
it also saves the JSON at the requested path so the session runner can archive
it. Missing APIs, failed queries, and save failures remain visible in structured
`status`/`error` fields. Reported game and extender versions are null with an
error when their exported functions are unavailable.

Run this only as a manual diagnostic. It does not prove that the passive
collector is complete, that an actor is decision-ready, or that any action is
legal. Verify the Script Extender API mappings and output on Windows before
relying on the result.
