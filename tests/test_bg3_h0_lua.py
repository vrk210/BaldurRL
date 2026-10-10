"""Execute actual collector Lua under Lua 5.4; BG3/Ext/Osi are test doubles.

Requires optional `pip install lupa`. Skips explicitly when unavailable. These
tests verify Lua execution and Python compatibility, never live BG3 behavior.
"""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from integration.inspect import inspect_directory
from integration.publish import publish_ready
from integration.reader import RecordDirectoryReader
from integration.schema import EventKind, GameEvent, GameSnapshot
from integration.serialization import parse_record


MOD = Path(__file__).resolve().parents[1] / "bg3/h0/Mods/BaldurRL_H0"
LUA = MOD / "ScriptExtender/Lua"
SIGNATURES = {
    "CombatStarted": 1, "CombatEnded": 1, "EnteredCombat": 2,
    "CombatRoundStarted": 2, "TurnStarted": 1, "TurnEnded": 1,
    "UsingSpell": 5, "UsingSpellOnTarget": 6, "AttackedBy": 7,
    "MissedBy": 4, "CriticalHitBy": 4, "Died": 1,
}


@pytest.fixture
def harness(tmp_path):
    lua54 = pytest.importorskip("lupa.lua54", reason="Real Lua tests require optional lupa (Lua 5.4)")
    runtime = lua54.LuaRuntime(unpack_returned_tuples=True)
    listeners, lifecycle, commands, messages, modules = {}, {}, {}, [], {}
    state = {"controlled": "synthetic-player", "combat": None, "hp": 20}
    counter = 0

    def save(path, contents):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents.encode("utf-8"))
        return True

    def load(path):
        target = tmp_path / path
        return target.read_text(encoding="utf-8") if target.exists() else None

    def require(name):
        if name not in modules:
            modules[name] = runtime.execute((LUA / name).read_text())
        return modules[name]

    def register(name, arity, when, callback):
        assert arity == SIGNATURES[name] and when == "after"
        assert name not in listeners
        listeners[name] = callback

    def guid():
        nonlocal counter
        counter += 1
        return f"{counter:08x}-0000-4000-8000-000000000001"

    def subscribe(name):
        return runtime.table_from({"Subscribe": lambda _, callback: lifecycle.setdefault(name, []).append(callback)})

    runtime.globals().Ext = runtime.table_from({
        "Require": require, "IO": runtime.table_from({"SaveFile": save, "LoadFile": load}),
        "Utils": runtime.table_from({"Print": messages.append, "GenerateGuid": guid}),
        "Osiris": runtime.table_from({"RegisterListener": register}),
        "Events": runtime.table_from({name: subscribe(name) for name in ("SessionLoaded", "SessionLoading", "ResetCompleted")}),
        "RegisterConsoleCommand": lambda name, callback: commands.setdefault(name, callback),
    })
    runtime.globals().Osi = runtime.table_from({
        "GetHostCharacter": lambda: "synthetic-avatar",
        "GetReservedUserID": lambda _: 17,
        "GetCurrentCharacter": lambda user: state["controlled"] if user == 17 else None,
        "CombatGetGuidFor": lambda _: state["combat"],
        "GetHitpoints": lambda _: state["hp"],
        "GetMaxHitpoints": lambda _: 20,
    })
    require("BootstrapServer.lua")

    def fire(name, *args):
        assert len(args) == SIGNATURES[name]
        listeners[name](*args)

    def life(name):
        for callback in lifecycle[name]:
            callback()

    return SimpleNamespace(lua=runtime, root=tmp_path, state=state, fire=fire,
                           life=life, commands=commands, messages=messages, require=require)


def publish_session(harness, number=1):
    session = harness.root / "BaldurRL/H0" / f"{number:08x}-0000-4000-8000-000000000001"
    publish_ready(session / "staging", session / "records")
    reader = RecordDirectoryReader(session / "records")
    records = reader.read_all()
    assert not reader.validate_all().gaps
    for path in (session / "records").glob("*.json"):
        payload = json.loads(path.read_text())
        assert parse_record(payload).to_dict() == payload
    return records, session


def test_real_lua_h0_trace_parses_and_inspects(harness):
    harness.life("SessionLoaded")
    harness.state["combat"] = "synthetic-combat"
    harness.fire("CombatStarted", "synthetic-combat")
    harness.fire("EnteredCombat", "synthetic-player", "synthetic-combat")
    harness.fire("EnteredCombat", "synthetic-enemy", "synthetic-combat")
    harness.fire("CombatRoundStarted", "synthetic-combat", 1)
    harness.fire("TurnStarted", "synthetic-player")
    spell = 'synthetic-ability-"\\\n\t\x00-é'
    harness.fire("UsingSpell", "synthetic-player", spell, "target", "", 71)
    harness.fire("UsingSpellOnTarget", "synthetic-player", "synthetic-enemy", spell, "target", "", 71)
    harness.state["hp"] = 15
    harness.fire("AttackedBy", "synthetic-enemy", "synthetic-owner", "synthetic-player", "Slashing", 5, "Attack", 71)
    harness.fire("CriticalHitBy", "synthetic-enemy", "synthetic-owner", "synthetic-player", 71)
    harness.fire("TurnEnded", "synthetic-player")
    harness.fire("TurnStarted", "synthetic-enemy")
    harness.fire("MissedBy", "synthetic-player", "synthetic-enemy", "synthetic-enemy", 72)
    harness.fire("Died", "synthetic-enemy")
    harness.state["combat"] = None
    harness.fire("CombatEnded", "synthetic-combat")
    records, session = publish_session(harness)
    assert [r.sequence for r in records] == list(range(1, len(records) + 1))
    events = [r for r in records if isinstance(r, GameEvent)]
    assert set(r.kind for r in events) == set(EventKind) - {EventKind.ABILITY_USED_AT_POSITION}
    use = next(r for r in events if r.kind == EventKind.ABILITY_USED_ON_TARGET)
    assert (use.ability_id, use.story_action_id) == (spell, 71)
    damage = next(r for r in events if r.kind == EventKind.DAMAGE)
    assert (damage.actor_id, damage.target_id, damage.damage, damage.story_action_id) == ("synthetic-player", "synthetic-enemy", 5, 71)
    assert next(r for r in events if r.kind == EventKind.CRITICAL_HIT).critical is True
    assert next(r for r in events if r.kind == EventKind.DIED).story_action_id is None
    assert isinstance(records[0], GameSnapshot) and records[0].controlled_entity_id == "synthetic-player"
    for record in records:
        if isinstance(record, GameSnapshot):
            assert not record.candidate_decisions
            assert record.combat is None or not record.combat.participant_ids
            for entity in record.entities:
                assert entity.armor_class is entity.alive is entity.position is None
                assert not entity.resources and not entity.abilities
    assert "gaps: none" in inspect_directory(session / "records")
    inspected = subprocess.run([sys.executable, "-m", "integration.inspect", str(session / "records")],
                               capture_output=True, text=True, check=True)
    assert "gaps: none" in inspected.stdout
    assert not any("HALTED" in message or "UNAVAILABLE" in message for message in harness.messages)


def test_real_lua_reload_reset_and_manual_start_use_new_namespaces(harness):
    harness.life("SessionLoaded")
    harness.life("SessionLoading")
    harness.fire("Died", "synthetic-enemy")  # Not collected during loading.
    harness.life("SessionLoaded")
    harness.life("ResetCompleted")
    harness.commands["baldurrl_stop"]()
    harness.fire("Died", "synthetic-enemy")
    harness.commands["baldurrl_start"]()
    for number in range(1, 5):
        records, _ = publish_session(harness, number)
        assert len(records) == 1 and records[0].sequence == 1


def test_real_lua_unknown_queries_and_overlapping_turns(harness):
    harness.life("SessionLoaded")
    harness.state["combat"] = "synthetic-combat"
    harness.fire("TurnStarted", "synthetic-player")
    harness.fire("TurnStarted", "synthetic-enemy")
    harness.lua.execute("Osi.GetHitpoints = function() error('query unavailable') end; Osi.GetMaxHitpoints = function() return nil end")
    harness.commands["baldurrl_snapshot"]()
    records, _ = publish_session(harness)
    assert records[-1].combat.active_entity_id is None
    assert records[-1].combat.round_number is None
    assert all(entity.hp is None and entity.max_hp is None for entity in records[-1].entities)
    assert sum("Query GetHitpoints failed" in message for message in harness.messages) == 1


def test_real_lua_selection_turn_and_combat_end_facts(harness):
    harness.life("SessionLoaded")
    harness.state["combat"] = "synthetic-combat"
    harness.fire("CombatRoundStarted", "synthetic-combat", 3)
    harness.fire("TurnStarted", "synthetic-player")
    records, _ = publish_session(harness)
    assert records[-1].combat.active_entity_id == "synthetic-player"
    assert records[-1].combat.round_number == 3
    harness.fire("TurnEnded", "synthetic-player")
    harness.state["controlled"] = "synthetic-companion"
    harness.commands["baldurrl_snapshot"]()
    records, _ = publish_session(harness)
    assert records[-1].controlled_entity_id == "synthetic-companion"
    assert records[-1].combat.active_entity_id is None
    # Even a stale post-end query must not resurrect observed ended combat.
    harness.fire("CombatEnded", "synthetic-combat")
    records, _ = publish_session(harness)
    assert records[-1].combat is None
    harness.lua.execute("Osi.GetReservedUserID = function() return -65536 end")
    harness.commands["baldurrl_snapshot"]()
    records, _ = publish_session(harness)
    assert records[-1].controlled_entity_id is None


@pytest.mark.parametrize("failure", ["partial", "save_false", "ready_false", "serialization"])
def test_real_lua_emitter_fails_closed_and_never_completes_partial_json(harness, failure):
    lua = harness.lua
    lua.globals().Wire = harness.require("Wire.lua")
    lua.globals().Emitter = harness.require("Emitter.lua")
    lua.execute("e = Emitter.new(Ext.IO, Wire, 'failure-session')")
    if failure == "partial":
        lua.execute("Ext.IO.LoadFile = function(path) if path:match('pending$') then return '{' end end")
    elif failure == "save_false":
        lua.execute("Ext.IO.SaveFile = function() return false end")
    elif failure == "ready_false":
        lua.execute("saved = Ext.IO.SaveFile; Ext.IO.SaveFile = function(path, body) if path:match('ready$') then return false end return saved(path, body) end")
    facts = "{kind = 'damage', damage = math.huge}" if failure == "serialization" else "{kind = 'died'}"
    assert not lua.execute(f"return pcall(function() e:emit('event', {facts}) end)")[0]
    assert lua.globals().e.halted
    assert not list((harness.root / "failure-session").glob("*.ready"))
    assert not lua.execute("return pcall(function() e:emit('event', {kind = 'died'}) end)")[0]


def test_real_lua_emitter_refuses_reusing_session(harness):
    lua = harness.lua
    lua.globals().Wire = harness.require("Wire.lua")
    lua.globals().Emitter = harness.require("Emitter.lua")
    lua.execute("e = Emitter.new(Ext.IO, Wire, 'same-session')")
    assert not lua.execute("return pcall(function() Emitter.new(Ext.IO, Wire, 'same-session') end)")[0]


def test_real_lua_zero_story_action_and_damage_are_preserved(harness):
    harness.life("SessionLoaded")
    harness.fire("UsingSpell", "synthetic-player", "synthetic-ability", "target", "", 0)
    harness.fire("AttackedBy", "synthetic-enemy", "synthetic-player", "synthetic-player", "", 0, "Status_Enter", 0)
    records, _ = publish_session(harness)
    events = [record for record in records if isinstance(record, GameEvent)]
    assert all(record.story_action_id == 0 for record in events)
    assert events[-1].damage == 0 and events[-1].damage_type == ""
    assert events[-1].ability_id is events[-1].critical is None


def test_mod_layout_static_only():
    """Metadata/layout check only; this test does NOT execute Lua or BG3."""
    config = json.loads((MOD / "ScriptExtender/Config.json").read_text())
    assert config == {"RequiredVersion": 30, "ModTable": "BaldurRLH0", "FeatureFlags": ["Lua"]}
    attributes = {node.get("id"): node.get("value") for node in ET.parse(MOD / "meta.lsx").iter("attribute")}
    assert attributes["Folder"] == MOD.name and attributes["UUID"]
    assert (LUA / "BootstrapServer.lua").is_file() and (LUA / "BootstrapClient.lua").is_file()
