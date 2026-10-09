-- All BG3-specific read-only queries and event state live here.
-- Signatures and limitations: bg3/h0/README.md. No ECS guesses.
local O = {}
function O.new(osi, report)
    local self = {rounds = {}, turns = {}, seen = {}, endedCombats = {}}
    local warned = {}
    local function query(name, ...)
        local ok, value = pcall(osi[name], ...)
        if not ok then
            if not warned[name] then report('Query ' .. name .. ' failed: ' .. tostring(value)); warned[name] = true end
            return nil
        end
        return value
    end
    local function id(value)
        if type(value) == 'string' and value ~= '' and
            value ~= '00000000-0000-0000-0000-000000000000' then return value end
        return nil
    end
    local function integer(value)
        if type(value) == 'number' then return math.tointeger(value) end
        return nil
    end
    function self:remember(value)
        value = id(value)
        if value then self.seen[value] = true end
        return value
    end
    function self:combatFor(entity)
        if not id(entity) then return nil end
        local combat = id(query('CombatGetGuidFor', entity))
        if combat and not self.endedCombats[combat] then return combat end
        return nil
    end
    function self:context(actor, target)
        local combat = self:combatFor(actor) or self:combatFor(target)
        return combat, combat and self.rounds[combat] or nil
    end
    function self:round(combat, number)
        self.rounds[combat], self.turns[combat] = integer(number), {}
    end
    function self:turn(entity, started)
        local combat = self:combatFor(entity)
        if combat then
            self.turns[combat] = self.turns[combat] or {}
            self.turns[combat][entity] = started and true or nil
        else
            -- TurnEnded may arrive after an entity has left combat.
            for _, turns in pairs(self.turns) do turns[entity] = nil end
        end
    end
    function self:ended(combat)
        self.rounds[combat], self.turns[combat] = nil, nil
        self.endedCombats[combat] = true
    end
    function self:snapshot()
        -- Host avatar is NOT necessarily the selected character. Resolve its user.
        local host = id(query('GetHostCharacter'))
        local user = host and integer(query('GetReservedUserID', host)) or nil
        local controlled = user and user ~= -65536 and id(query('GetCurrentCharacter', user)) or nil
        self:remember(controlled)
        local combat = self:combatFor(controlled)
        local facts = {controlled_entity_id = controlled, entities = {}}
        if combat then
            local active, count = nil, 0
            for entity in pairs(self.turns[combat] or {}) do active = entity; count = count + 1 end
            -- Shared turns cannot be represented by this singular v1 field.
            facts.combat = {combat_id = combat, round_number = self.rounds[combat],
                active_entity_id = count == 1 and active or nil}
        end
        local ids = {}
        for entity in pairs(self.seen) do ids[#ids + 1] = entity end
        table.sort(ids)
        for _, entity in ipairs(ids) do
            facts.entities[#facts.entities + 1] = {entity_id = entity,
                hp = integer(query('GetHitpoints', entity)),
                max_hp = integer(query('GetMaxHitpoints', entity))}
        end
        return facts
    end
    return self
end
return O
