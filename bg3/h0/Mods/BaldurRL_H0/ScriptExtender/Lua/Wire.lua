-- Pure Lua v1 wire encoding. No engine objects or API calls belong here.
local W = {}
W.null = {}
local arrayTag = {}
function W.array(values) return setmetatable(values or {}, arrayTag) end

local function quote(value)
    return '"' .. value:gsub('[%z\1-\31\\"]', function(char)
        if char == '"' then return '\\"' end
        if char == '\\' then return '\\\\' end
        return string.format('\\u%04x', string.byte(char))
    end) .. '"'
end

function W.encode(value)
    if value == nil or value == W.null then return 'null' end
    local kind = type(value)
    if kind == 'string' then return quote(value) end
    if kind == 'boolean' then return value and 'true' or 'false' end
    if kind == 'number' then
        assert(value == value and value ~= math.huge and value ~= -math.huge, 'Non-finite number')
        return tostring(value)
    end
    assert(kind == 'table', 'Only JSON primitives may be serialized')
    local parts = {}
    if getmetatable(value) == arrayTag then
        for _, item in ipairs(value) do parts[#parts + 1] = W.encode(item) end
        return '[' .. table.concat(parts, ',') .. ']'
    end
    local keys = {}
    for key in pairs(value) do
        assert(type(key) == 'string', 'Object keys must be strings')
        keys[#keys + 1] = key
    end
    table.sort(keys)
    for _, key in ipairs(keys) do parts[#parts + 1] = quote(key) .. ':' .. W.encode(value[key]) end
    return '{' .. table.concat(parts, ',') .. '}'
end

local function fields(source, names)
    local result = {}
    for _, name in ipairs(names) do
        local value = source[name]
        if value == nil then value = W.null end
        result[name] = value
    end
    return result
end

function W.event(facts)
    return fields(facts, {'kind', 'combat_id', 'round_number', 'actor_id', 'target_id',
        'ability_id', 'target_position', 'story_action_id', 'damage', 'damage_type', 'critical'})
end

function W.snapshot(facts)
    local result = fields(facts, {'controlled_entity_id', 'combat'})
    if facts.combat then
        result.combat = fields(facts.combat, {'combat_id', 'round_number', 'active_entity_id'})
        result.combat.participant_ids = W.array() -- Enumeration is unavailable in H0.
    end
    result.entities = W.array()
    for _, entity in ipairs(facts.entities or {}) do
        local item = fields(entity, {'entity_id', 'name', 'hp', 'max_hp', 'armor_class', 'position', 'alive'})
        item.resources, item.abilities = W.array(), W.array()
        result.entities[#result.entities + 1] = item
    end
    result.candidate_decisions = W.array()
    return result
end

return W
