"""Offline filesystem tests for the read-only Windows preflight."""

import json
from pathlib import Path

import pytest

from integration.windows_preflight import MODULE_UUID, main, run_preflight


def write_repo(root: Path, *, config: str = "{}", uuid: str = MODULE_UUID) -> None:
    mod = root / "bg3" / "h0" / "Mods" / "BaldurRL_H0"
    (mod / "ScriptExtender").mkdir(parents=True)
    (mod / "meta.lsx").write_text(
        f'<save><node id="ModuleInfo"><attribute id="UUID" value="{uuid}" /></node></save>',
        encoding="utf-8",
    )
    (mod / "ScriptExtender" / "Config.json").write_text(config, encoding="utf-8")
    lua = mod / "ScriptExtender" / "Lua"
    lua.mkdir()
    for name in ("BootstrapServer.lua", "BootstrapClient.lua", "Observation.lua", "Wire.lua", "Emitter.lua", "Inspect.lua"):
        (lua / name).write_text("-- fixture", encoding="utf-8")


def write_install(root: Path, *, settings: str = '{"CreateConsole": true, "LogRuntime": true}') -> Path:
    game = root / "BG3 install with spaces"
    bin_dir = game / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "bg3_dx11.exe").write_bytes(b"fake")
    (bin_dir / "DWrite.dll").write_bytes(b"fake")
    (bin_dir / "ScriptExtenderSettings.json").write_text(settings, encoding="utf-8")
    return game


def write_runtime(root: Path, *, active_uuid: str = MODULE_UUID) -> tuple[Path, Path]:
    extender = root / "Local App Data" / "Larian Studios" / "Baldur's Gate 3" / "Script Extender"
    (extender / "BaldurRL" / "H0").mkdir(parents=True, exist_ok=True)
    modsettings = root / "profile with spaces" / "modsettings.lsx"
    modsettings.parent.mkdir(exist_ok=True)
    modsettings.write_text(
        # Patch 7 export structure from BG3ModManager cd6a840 (not a live capture).
        '<?xml version="1.0" encoding="UTF-8"?><save>'
        '<version major="4" minor="7" revision="1" build="3"/>'
        '<region id="ModuleSettings"><node id="root"><children>'
        '<node id="Mods"><children><node id="ModuleShortDesc">'
        '<attribute id="Folder" type="LSString" value="BaldurRL_H0"/>'
        '<attribute id="PublishHandle" type="uint64" value="0"/>'
        f'<attribute id="UUID" type="guid" value="{active_uuid}"/>'
        '</node></children></node></children></node></region></save>',
        encoding="utf-8",
    )
    return extender, modsettings


def by_name(report):
    return {check["name"]: check for check in report["checks"]}


def test_ready_static_report_uses_explicit_spaced_paths(tmp_path):
    repo = tmp_path / "repo"
    write_repo(repo)
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)

    report = run_preflight(
        game,
        script_extender_root=extender,
        modsettings=modsettings,
        repo_root=repo,
        system="Windows",
    )

    assert report["schema_version"] == 1
    assert report["live_verified"] is False
    assert report["static_ready"] is True
    checks = by_name(report)
    assert Path(checks["game_bin"]["path"]) == (game / "bin").resolve()
    assert checks["mod_activation"]["status"] == "present_unverified"
    assert 'Patch 7+' in checks['mod_activation']['evidence']
    assert all({"status", "evidence", "path", "next_action"} <= check.keys() for check in report["checks"])


def test_accepts_explicit_bin_directory(tmp_path):
    repo = tmp_path / "repo"
    write_repo(repo)
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)
    report = run_preflight(game / "bin", script_extender_root=extender, modsettings=modsettings, repo_root=repo, system="Windows")
    assert report["static_ready"] is True


@pytest.mark.parametrize('layout,active', [('legacy', True), ('legacy_descriptions_only', False), ('unrelated', False)])
def test_legacy_and_unrelated_uuid_locations(tmp_path, layout, active):
    repo = tmp_path / "repo"
    write_repo(repo)
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)
    description = f'<node id="Mods"><children><node id="ModuleShortDesc"><attribute id="UUID" value="{MODULE_UUID}" /></node></children></node>'
    if layout == 'legacy':
        body = f'<node id="ModOrder"><children><node id="Module"><attribute id="UUID" value="{MODULE_UUID}" /></node></children></node>' + description
    elif layout == 'legacy_descriptions_only':
        body = '<node id="ModOrder"><children/></node>' + description
    else:
        body = f'<node id="Unrelated"><attribute id="UUID" value="{MODULE_UUID}" /></node>'
    modsettings.write_text(f'<save>{body}</save>')
    report = run_preflight(game, script_extender_root=extender, modsettings=modsettings, repo_root=repo, system="Windows")
    assert by_name(report)["mod_activation"]["status"] == ('present_unverified' if active else 'missing')
    assert report["static_ready"] is active


def test_missing_required_lua_source_fails_closed(tmp_path):
    repo = tmp_path / "repo"
    write_repo(repo)
    (repo / "bg3" / "h0" / "Mods" / "BaldurRL_H0" / "ScriptExtender" / "Lua" / "Emitter.lua").unlink()
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)
    report = run_preflight(game, script_extender_root=extender, modsettings=modsettings, repo_root=repo, system="Windows")
    assert by_name(report)["repo_lua_sources"]["status"] == "missing"
    assert "Emitter.lua" in by_name(report)["repo_lua_sources"]["evidence"]
    assert report["static_ready"] is False


@pytest.mark.parametrize(
    ("change", "check_name", "status"),
    [
        ("missing_exe", "game_bin", "missing"),
        ("missing_loader", "script_extender_loader", "missing"),
        ("disabled_logging", "script_extender_settings", "error"),
        ("wrong_uuid", "mod_activation", "missing"),
        ("missing_staging", "staging_root", "missing"),
    ],
)
def test_missing_prerequisites_fail_closed(tmp_path, change, check_name, status):
    repo = tmp_path / "repo"
    write_repo(repo)
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)
    if change == "missing_exe":
        (game / "bin" / "bg3_dx11.exe").unlink()
    elif change == "missing_loader":
        (game / "bin" / "DWrite.dll").unlink()
    elif change == "disabled_logging":
        (game / "bin" / "ScriptExtenderSettings.json").write_text('{"CreateConsole": true, "LogRuntime": false}')
    elif change == "wrong_uuid":
        write_runtime(tmp_path, active_uuid="00000000-0000-0000-0000-000000000000")
    else:
        extender = tmp_path / "does not exist"

    report = run_preflight(game, script_extender_root=extender, modsettings=modsettings, repo_root=repo, system="Windows")
    assert report["static_ready"] is False
    assert by_name(report)[check_name]["status"] == status


@pytest.mark.parametrize(
    ("target", "contents", "check_name"),
    [
        ("settings", "{oops", "script_extender_settings"),
        ("modsettings", "<save>", "mod_activation"),
        ("config", "[]", "mod_config"),
        ("meta", "<save>", "meta_uuid"),
    ],
)
def test_malformed_or_unknown_configuration_is_reported_without_crash(tmp_path, target, contents, check_name):
    repo = tmp_path / "repo"
    write_repo(repo, config=contents if target == "config" else "{}")
    game = write_install(tmp_path, settings=contents if target == "settings" else '{"CreateConsole": true, "LogRuntime": true}')
    extender, modsettings = write_runtime(tmp_path)
    if target == "modsettings":
        modsettings.write_text(contents)
    if target == "meta":
        (repo / "bg3" / "h0" / "Mods" / "BaldurRL_H0" / "meta.lsx").write_text(contents)

    report = run_preflight(game, script_extender_root=extender, modsettings=modsettings, repo_root=repo, system="Windows")
    check = by_name(report)[check_name]
    assert report["static_ready"] is False
    assert check["status"] == "error"
    assert "JSON" in check["evidence"] or "XML" in check["evidence"] or "object" in check["evidence"]


def test_localappdata_defaults_and_non_windows_status(tmp_path):
    repo = tmp_path / "repo"
    write_repo(repo)
    game = write_install(tmp_path)
    local = tmp_path / "Local App Data"
    extender = local / "Larian Studios" / "Baldur's Gate 3" / "Script Extender"
    (extender / "BaldurRL" / "H0").mkdir(parents=True)
    modsettings = local / "Larian Studios" / "Baldur's Gate 3" / "PlayerProfiles" / "Public" / "modsettings.lsx"
    modsettings.parent.mkdir(parents=True)
    modsettings.write_text(f'<save><node id="ModOrder"><node id="Module"><attribute id="UUID" value="{MODULE_UUID}" /></node></node></save>')
    report = run_preflight(game, repo_root=repo, local_app_data=str(local), system="Darwin")
    assert by_name(report)["platform"]["status"] == "unsupported"
    assert report["static_ready"] is False


def test_json_cli_output_and_exit_semantics(tmp_path, monkeypatch, capsys):
    game = write_install(tmp_path)
    extender, modsettings = write_runtime(tmp_path)
    monkeypatch.setattr("integration.windows_preflight.platform.system", lambda: "Windows")
    exit_code = main(["--game-dir", str(game), "--script-extender-root", str(extender), "--modsettings", str(modsettings), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["static_ready"] is True

    (game / "bin" / "DWrite.dll").unlink()
    exit_code = main(["--game-dir", str(game), "--script-extender-root", str(extender), "--modsettings", str(modsettings), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["static_ready"] is False
