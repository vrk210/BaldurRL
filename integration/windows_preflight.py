"""Read-only static checks for H0; READY never proves that the mod loaded.

Pass --game-dir as the BG3 installation root or bin directory. Optional
--script-extender-root and --modsettings default from LOCALAPPDATA. Missing,
malformed and unsupported prerequisites exit 1; --json emits the v1 report.
Settings and load orders are inspected only, never changed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = 1
MODULE_UUID = "d860477b-c599-4f12-9e95-5b8ec7235860"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    evidence: str
    path: str | None
    next_action: str


def _check(name: str, status: str, evidence: str, path: Path | None, next_action: str) -> Check:
    return Check(name, status, evidence, str(path) if path is not None else None, next_action)


def resolve_bin_directory(game_dir: Path) -> Path:
    """Accept either the BG3 installation root or its bin directory."""
    game_dir = game_dir.expanduser().resolve()
    nested = game_dir / "bin"
    return nested if nested.is_dir() else game_dir


def default_windows_paths(local_app_data: str | None) -> tuple[Path | None, Path | None]:
    if not local_app_data:
        return None, None
    larian = Path(local_app_data) / "Larian Studios" / "Baldur's Gate 3"
    return (
        larian / "Script Extender",
        larian / "PlayerProfiles" / "Public" / "modsettings.lsx",
    )


def _json_object(path: Path, name: str) -> tuple[dict[str, object] | None, Check | None]:
    if not path.is_file():
        return None, _check(name, "missing", "file does not exist", path, f"Create or install {path.name}, then rerun preflight.")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _check(name, "error", f"invalid JSON: {exc}", path, f"Repair {path.name} as valid JSON, then rerun preflight.")
    if not isinstance(value, dict):
        return None, _check(name, "error", "JSON root is not an object", path, f"Change {path.name} to contain a JSON object.")
    return value, None


def _repo_checks(repo_root: Path) -> list[Check]:
    mod = repo_root / "bg3" / "h0" / "Mods" / "BaldurRL_H0"
    meta = mod / "meta.lsx"
    config = mod / "ScriptExtender" / "Config.json"
    lua_dir = mod / "ScriptExtender" / "Lua"
    required_lua = (
        "BootstrapServer.lua",
        "BootstrapClient.lua",
        "Observation.lua",
        "Wire.lua",
        "Emitter.lua",
        "Inspect.lua",
    )
    checks: list[Check] = []
    if not mod.is_dir():
        return [_check("repo_mod_layout", "missing", "expected mod source directory is absent", mod, "Use a complete BaldurRL checkout containing bg3/h0/Mods/BaldurRL_H0.")]
    checks.append(_check("repo_mod_layout", "present_unverified", "Mods/BaldurRL_H0 source layout is present; game loading is not verified", mod, "Package with Mods at the package root, then verify loading in BG3 on Windows."))
    missing_lua = [name for name in required_lua if not (lua_dir / name).is_file()]
    checks.append(_check(
        "repo_lua_sources",
        "missing" if missing_lua else "present_unverified",
        f"missing required Lua files: {', '.join(missing_lua)}" if missing_lua else "all required H0 bootstrap/helper Lua files are present; execution is not verified",
        lua_dir,
        "Restore the complete H0 Lua source set." if missing_lua else "Run the Lua tests and verify execution through Script Extender on Windows.",
    ))
    if not meta.is_file():
        checks.append(_check("meta_uuid", "missing", "meta.lsx does not exist", meta, "Restore the source mod metadata."))
    else:
        try:
            root = ET.parse(meta).getroot()
            uuids = [
                child.get("value")
                for node in root.iter("node")
                if node.get("id") == "ModuleInfo"
                for child in node.iter("attribute")
                if child.get("id") == "UUID"
            ]
            status = "present_unverified" if MODULE_UUID in uuids else "error"
            evidence = f"module UUID {MODULE_UUID} found" if status == "present_unverified" else f"expected module UUID {MODULE_UUID} not found"
            action = "Verify this UUID is active in the live load order." if status == "present_unverified" else "Restore meta.lsx with the documented H0 module UUID."
            checks.append(_check("meta_uuid", status, evidence, meta, action))
        except (OSError, UnicodeError, ET.ParseError) as exc:
            checks.append(_check("meta_uuid", "error", f"invalid XML: {exc}", meta, "Repair meta.lsx as valid XML, then rerun preflight."))
    config_value, config_error = _json_object(config, "mod_config")
    if config_error:
        checks.append(config_error)
    else:
        checks.append(_check("mod_config", "present_unverified", "Config.json is a JSON object", config, "Verify Script Extender accepts this configuration in the live game."))
    return checks


def run_preflight(
    game_dir: Path,
    *,
    script_extender_root: Path | None = None,
    modsettings: Path | None = None,
    repo_root: Path | None = None,
    local_app_data: str | None = None,
    system: str | None = None,
) -> dict[str, object]:
    """Return a stable machine-readable static readiness report without writes."""
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    system = system or platform.system()
    checks = [_check(
        "platform",
        "present_unverified" if system == "Windows" else "unsupported",
        f"detected {system}",
        None,
        "Run this preflight on the target Windows host." if system != "Windows" else "Continue with static checks; live game verification is still required.",
    )]
    checks.extend(_repo_checks(repo_root))

    bin_dir = resolve_bin_directory(game_dir)
    executables = [bin_dir / "bg3.exe", bin_dir / "bg3_dx11.exe"]
    found_executables = [path.name for path in executables if path.is_file()]
    if found_executables:
        checks.append(_check("game_bin", "present_unverified", f"found {', '.join(found_executables)}", bin_dir, "Launch BG3 manually and verify the installed game version."))
    else:
        checks.append(_check("game_bin", "missing", "neither bg3.exe nor bg3_dx11.exe was found", bin_dir, "Pass the BG3 installation root (containing bin) or its bin directory."))

    dwrite = bin_dir / "DWrite.dll"
    checks.append(_check(
        "script_extender_loader",
        "present_unverified" if dwrite.is_file() else "missing",
        "DWrite.dll is present; loading is not verified" if dwrite.is_file() else "DWrite.dll does not exist",
        dwrite,
        "Launch BG3 and verify Script Extender startup/version output." if dwrite.is_file() else "Install the BG3 Script Extender updater manually into the game bin directory.",
    ))

    settings = bin_dir / "ScriptExtenderSettings.json"
    settings_value, settings_error = _json_object(settings, "script_extender_settings")
    if settings_error:
        checks.append(settings_error)
    else:
        enabled = settings_value.get("CreateConsole") is True and settings_value.get("LogRuntime") is True
        checks.append(_check(
            "script_extender_settings",
            "present_unverified" if enabled else "error",
            "CreateConsole and LogRuntime are true" if enabled else "CreateConsole and LogRuntime must both be JSON true",
            settings,
            "Verify console and runtime logging during a live launch." if enabled else "Merge CreateConsole=true and LogRuntime=true while preserving existing settings.",
        ))

    default_extender, default_modsettings = default_windows_paths(local_app_data if local_app_data is not None else os.environ.get("LOCALAPPDATA"))
    extender = (script_extender_root or default_extender)
    if extender is None:
        checks.append(_check("staging_root", "missing", "no Script Extender root was supplied and LOCALAPPDATA is unavailable", None, "Pass --script-extender-root explicitly."))
    else:
        staging_root = extender.expanduser().resolve()
        checks.append(_check(
            "staging_root",
            "present_unverified" if staging_root.is_dir() else "missing",
            "Script Extender staging root exists; collector IO is not verified" if staging_root.is_dir() else "Script Extender staging root does not exist",
            staging_root,
            "Verify the exact BaldurRL/H0 session staging path after a live collector start." if staging_root.is_dir() else "Create or select the documented Script Extender writable root deliberately.",
        ))

    modsettings_path = modsettings or default_modsettings
    if modsettings_path is None:
        checks.append(_check("mod_activation", "missing", "no modsettings.lsx was supplied and LOCALAPPDATA is unavailable", None, "Pass --modsettings explicitly."))
    else:
        modsettings_path = modsettings_path.expanduser().resolve()
        if not modsettings_path.is_file():
            checks.append(_check("mod_activation", "missing", "modsettings.lsx does not exist", modsettings_path, "Export an active load order or pass the correct modsettings.lsx path."))
        else:
            try:
                root = ET.parse(modsettings_path).getroot()
                # Patch 7 removed ModOrder; Mods now carries the active list.
                # Retain the old interpretation when a legacy ModOrder exists:
                # its Mods descriptions alone did not establish activation.
                orders = [node for node in root.iter('node') if node.get('id') == 'ModOrder']
                containers = orders or [node for node in root.iter('node') if node.get('id') == 'Mods']
                layout = 'legacy ModOrder' if orders else 'Mods/ModuleShortDesc (Patch 7+)'
                module_type = 'Module' if orders else 'ModuleShortDesc'
                active = any(
                    module.get("id") == module_type
                    and any(attribute.get("id") == "UUID" and attribute.get("value", "").lower() == MODULE_UUID for attribute in module.findall("attribute"))
                    for order in containers
                    for module in order.iter("node")
                )
                checks.append(_check(
                    "mod_activation",
                    "present_unverified" if active else "missing",
                    f"module UUID {MODULE_UUID} {'found' if active else 'not found'} in {layout}; live activation is not verified",
                    modsettings_path,
                    "Verify the active load order and H0 bootstrap in BG3." if active else "Activate BaldurRL H0 and export the load order, then rerun preflight.",
                ))
            except (OSError, UnicodeError, ET.ParseError) as exc:
                checks.append(_check("mod_activation", "error", f"invalid XML: {exc}", modsettings_path, "Repair or re-export modsettings.lsx, then rerun preflight."))

    blocking = {"missing", "error", "unsupported"}
    return {
        "schema_version": SCHEMA_VERSION,
        "live_verified": False,
        "static_ready": not any(check.status in blocking for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def _print_human(report: dict[str, object]) -> None:
    print(f"Static readiness: {'READY' if report['static_ready'] else 'NOT READY'}")
    print("Live verified: no")
    for check in report["checks"]:
        path = f" ({check['path']})" if check["path"] else ""
        print(f"[{check['status']}] {check['name']}: {check['evidence']}{path}")
        print(f"  Next: {check['next_action']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", required=True, type=Path, help="BG3 installation root or bin directory")
    parser.add_argument("--script-extender-root", type=Path, help="Script Extender writable root (the parent of BaldurRL/H0)")
    parser.add_argument("--modsettings", type=Path, help="modsettings.lsx to inspect for the H0 UUID")
    parser.add_argument("--json", action="store_true", help="emit the versioned JSON report")
    args = parser.parse_args(argv)
    report = run_preflight(args.game_dir, script_extender_root=args.script_extender_root, modsettings=args.modsettings)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["static_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
