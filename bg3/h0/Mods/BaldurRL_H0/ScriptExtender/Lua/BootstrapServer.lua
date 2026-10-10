local Wire = Ext.Require('Wire.lua')
local Emitter = Ext.Require('Emitter.lua')
local Observation = Ext.Require('Observation.lua')
local emitter, observation
local function report(message) Ext.Utils.Print('[BaldurRL H0] ' .. message) end
local function snapshot()
    if emitter and not emitter.halted then emitter:emit('snapshot', observation:snapshot()) end
end
local function start()
    emitter = nil
    -- GenerateGuid is exported by BG3SE Utils.inl; runtime/version must be checked live.
    local session = tostring(Ext.Utils.GenerateGuid())
    assert(session:match('^[%x%-]+$'), 'Invalid collector session GUID')
    local directory = 'BaldurRL/H0/' .. session .. '/staging'
    emitter = Emitter.new(Ext.IO, Wire, directory)
    observation = Observation.new(Osi, report)
    report('New session: ' .. directory)
    snapshot()
end

local function emit(kind, facts, capture)
    if not emitter or emitter.halted then return end
    observation:remember(facts.actor_id)
    observation:remember(facts.target_id)
    if facts.combat_id == nil then
        facts.combat_id, facts.round_number = observation:context(facts.actor_id, facts.target_id)
    elseif facts.round_number == nil then
        facts.round_number = observation.rounds[facts.combat_id]
    end
    facts.kind = kind
    emitter:emit('event', facts)
    if capture then snapshot() end
end

local function listen(name, arity, handler)
    -- Missing symbols/signature mismatches are visible, never silently discarded.
    local ok, err = pcall(Ext.Osiris.RegisterListener, name, arity, 'after', function(...)
        if not emitter or emitter.halted then return end
        local success, failure = pcall(handler, ...)
        if not success then emitter.halted = true; report('HALTED at ' .. name .. ': ' .. tostring(failure)) end
    end)
    if not ok then report('UNAVAILABLE listener ' .. name .. '/' .. arity .. ': ' .. tostring(err)) end
end

listen('CombatStarted', 1, function(combat) emit('combat_started', {combat_id = combat}, true) end)
listen('CombatEnded', 1, function(combat)
    emit('combat_ended', {combat_id = combat})
    observation:ended(combat)
    snapshot()
end)
listen('EnteredCombat', 2, function(entity, combat)
    emit('entered_combat', {actor_id = entity, combat_id = combat}, true)
end)
listen('CombatRoundStarted', 2, function(combat, round)
    observation:round(combat, round)
    emit('round_started', {combat_id = combat, round_number = round}, true)
end)
listen('TurnStarted', 1, function(entity)
    observation:turn(entity, true)
    emit('turn_started', {actor_id = entity}, true)
end)
listen('TurnEnded', 1, function(entity)
    observation:turn(entity, false)
    emit('turn_ended', {actor_id = entity}, true)
end)
listen('UsingSpell', 5, function(caster, spell, spellType, element, action)
    emit('ability_used', {actor_id = caster, ability_id = spell, story_action_id = action})
end)
listen('UsingSpellOnTarget', 6, function(caster, target, spell, spellType, element, action)
    emit('ability_used_on_target', {actor_id = caster, target_id = target,
        ability_id = spell, story_action_id = action})
end)
listen('AttackedBy', 7, function(defender, owner, attacker, damageType, amount, cause, action)
    -- actor_id is the direct attacker; ownership/cause have no v1 fields.
    observation:remember(owner)
    emit('damage', {actor_id = attacker, target_id = defender, damage = amount,
        damage_type = damageType, story_action_id = action}, true)
end)
listen('MissedBy', 4, function(defender, owner, attacker, action)
    observation:remember(owner)
    emit('miss', {actor_id = attacker, target_id = defender, story_action_id = action}, true)
end)
listen('CriticalHitBy', 4, function(defender, owner, attacker, action)
    observation:remember(owner)
    emit('critical_hit', {actor_id = attacker, target_id = defender, story_action_id = action, critical = true}, true)
end)
listen('Died', 1, function(entity)
    emit('died', {actor_id = entity}, true)
end)

Ext.Events.SessionLoading:Subscribe(function() emitter, observation = nil, nil end)
Ext.Events.SessionLoaded:Subscribe(start)
-- reset creates a fresh namespace too; never restore sequence from a savegame.
Ext.Events.ResetCompleted:Subscribe(start)
Ext.RegisterConsoleCommand('baldurrl_snapshot', snapshot)
Ext.RegisterConsoleCommand('baldurrl_stop', function() emitter = nil; report('Stopped') end)
Ext.RegisterConsoleCommand('baldurrl_start', start)
