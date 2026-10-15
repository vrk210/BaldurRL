# Windows H0 bootstrap, preflight, and packaging

These tools prepare and inspect the passive H0 collector. They do not install BG3, Script Extender, or LSLib; modify the registry, settings, or load order; launch the game; or prove that the mod loads. Core preflight and integration code uses only the Python standard library.

## Static preflight

From the repository root, inspect an installation with explicit paths:

```powershell
python -m integration.windows_preflight `
  --game-dir "C:\Program Files (x86)\Steam\steamapps\common\Baldurs Gate 3" `
  --script-extender-root "$env:LOCALAPPDATA\Larian Studios\Baldur's Gate 3\Script Extender" `
  --modsettings "$env:LOCALAPPDATA\Larian Studios\Baldur's Gate 3\PlayerProfiles\Public\modsettings.lsx"
```

`--game-dir` may name the game root containing `bin` or the `bin` directory itself. If the other two paths are omitted, the doctor derives the documented paths above from `LOCALAPPDATA`; it never searches unrelated directories. Add `--json` for a machine-readable report. Schema version 1 contains `live_verified=false`, `static_ready`, and checks with a status, evidence/path, and next action. `missing`, `error`, and `unsupported` checks make the exit status nonzero. `present_unverified` means only that a static prerequisite was found.

The doctor reads the repository mod layout and required Lua sources, `meta.lsx`, mod `Config.json`, game executables, `DWrite.dll`, `ScriptExtenderSettings.json`, the Script Extender staging root, and `modsettings.lsx`. It requires the actual H0 module UUID in `ModOrder` and JSON booleans `CreateConsole: true` and `LogRuntime: true`. A UUID appearing only in the installed-mod descriptions does not count as active. Malformed JSON/XML is reported without a traceback.

The convenience bootstrap creates an isolated venv and runs the same doctor:

```powershell
.\scripts\windows\bootstrap-harness.ps1 `
  -Python py `
  -VenvPath ".venv-harness" `
  -GameDir "D:\SteamLibrary\steamapps\common\Baldurs Gate 3" `
  -InstallTestDependencies `
  -Json
```

`-InstallTestDependencies` is optional and installs only the pinned packages in `requirements-harness-test.txt`: pytest 9.1.1 and, on Python 3.11+, Lupa 2.8. Lupa 2.8 platform-wheel compatibility remains untested on the target Windows host. Omitting the switch performs no package installation. The script forwards `-ScriptExtenderRoot` and `-Modsettings` when supplied and propagates Python, pip, and doctor exit codes.

## Package the source mod

Install LSLib/Divine separately, then run:

```powershell
.\scripts\windows\package-h0.ps1 `
  -DivinePath "C:\Tools\LSLib\Divine.exe" `
  -OutputPath "C:\BaldurRL\packages\BaldurRL_H0.pak" `
  -DryRun
```

Dry run prints JSON containing the exact argv and SHA-256 source-file manifest. It does not invoke Divine or create a placeholder package. Remove `-DryRun` to execute:

```text
divine.exe --game bg3 --action create-package --source <repo>\bg3\h0 --destination <temporary-output>
```

The selected source has `Mods` at its root. The script rejects output beneath that source and existing output unless `-Force` is supplied. Divine writes a temporary output in the destination directory; only a successful, nonempty result replaces the requested package. A versioned `<output>.manifest.json` records source-file and output SHA-256 hashes. The Divine command line follows the upstream CLI, but this script and real converter invocation have not been run on Windows.

## Remaining live checks

After static readiness, all Windows/game behavior remains unverified. Launch BG3 manually and confirm the installed BG3 and Script Extender versions, console and runtime logging, package activation, the server bootstrap message, all listener registrations, session GUID generation, and the exact writable staging path. Exercise a manual encounter and verify UTF-8 IO, completion publication, callbacks, event ordering, observed values, save reload/reset behavior, and inspection as listed in `bg3/h0/README.md`. File presence and a listed UUID do not establish mod loading, API availability, ability legality, or collector correctness.
