-- BaldurRL source-reviewed, live-unverified read-only server probe.
-- Loaded at bootstrap; !baldurrl_inspect <selected-actor-uuid> saves a diagnostic
-- inside the current capture session. No host-selection inference or game action.
-- dump_json is an opaque JSON string from Ext.DumpExport, not a parsed component.

local JSON_NULL = {}

local function json_escape(value)
    return value:gsub('[%z\1-\31\\"]', function(character)
        local escapes = {['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b',
            ['\f'] = '\\f', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t'}
        return escapes[character] or string.format('\\u%04x', string.byte(character))
    end)
end

local function is_array(value)
    local largest = 0
    local count = 0
    for key, _ in pairs(value) do
        if type(key) ~= "number" or key < 1 or key % 1 ~= 0 then return false, 0 end
        largest = math.max(largest, key)
        count = count + 1
    end
    return largest == count, largest
end

local function json(value)
    if value == JSON_NULL then return "null" end
    if value == nil then return "null" end
    if type(value) == "boolean" or type(value) == "number" then return tostring(value) end
    if type(value) == "string" then return '"' .. json_escape(value) .. '"' end
    if type(value) ~= "table" then error("unsupported diagnostic value: " .. type(value)) end
    local array, length = is_array(value)
    local parts = {}
    if array then
        for index = 1, length do parts[#parts + 1] = json(value[index]) end
        return "[" .. table.concat(parts, ",") .. "]"
    end
    local keys = {}
    for key, _ in pairs(value) do
        if type(key) ~= "string" then error("diagnostic object key must be a string") end
        keys[#keys + 1] = key
    end
    table.sort(keys)
    for _, key in ipairs(keys) do
        parts[#parts + 1] = json(key) .. ":" .. json(value[key])
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

local function query(label, fn)
    local ok, value = pcall(fn)
    if ok then return {status = "ok", value = value} end
    return {status = "error", error = label .. ": " .. tostring(value)}
end

local function version_query(name)
    if not Ext or not Ext.Utils or type(Ext.Utils[name]) ~= "function" then
        return {status = "unavailable", value = JSON_NULL, error = "Ext.Utils." .. name .. " is unavailable"}
    end
    local result = query(name, function() return Ext.Utils[name]() end)
    if result.status == "ok" then
        if result.value == nil then
            return {status = "error", value = JSON_NULL, error = "Ext.Utils." .. name .. " returned nil"}
        end
        result.value = tostring(result.value)
    end
    return result
end

local function output(message)
    if Ext and Ext.Utils and type(Ext.Utils.Print) == "function" then
        Ext.Utils.Print(message)
    else
        print(message)
    end
end

local function inspect(selected_actor, output_path)
    local report = {
        report_version = 1,
        source_reviewed = true,
        live_verified = false,
        selected_actor_id = selected_actor == nil and JSON_NULL or selected_actor,
        script_extender_version = version_query("Version"),
        game_version = version_query("GameVersion"),
        component_names = {status = "unavailable", value = JSON_NULL},
        components = {},
        errors = {},
    }
    if type(selected_actor) ~= "string" or selected_actor == "" then
        report.errors[#report.errors + 1] = "selected_actor must be an explicit non-empty entity UUID"
    elseif not Ext or not Ext.Entity or type(Ext.Entity.Get) ~= "function" then
        report.errors[#report.errors + 1] = "Ext.Entity.Get is unavailable in this context"
    else
        -- Engine userdata is acquired and consumed entirely within this call.
        local actor_result = query("Ext.Entity.Get", function() return Ext.Entity.Get(selected_actor) end)
        if actor_result.status ~= "ok" then
            report.errors[#report.errors + 1] = actor_result.error
        elseif actor_result.value == nil then
            report.errors[#report.errors + 1] = "selected actor was not found"
        else
            local actor = actor_result.value
            local names_result = query("GetAllComponentNames", function() return actor:GetAllComponentNames() end)
            if names_result.status ~= "ok" then
                report.component_names = names_result
                report.errors[#report.errors + 1] = names_result.error
            else
                local present = {}
                local names = {}
                for _, name in pairs(names_result.value) do
                    local text = tostring(name)
                    present[text] = true
                    names[#names + 1] = text
                end
                table.sort(names)
                report.component_names = {status = "ok", value = names}
                for _, name in ipairs({"SpellBook", "SpellBookPrepares", "ActionResources", "TurnBased"}) do
                    if not present[name] then
                        report.components[name] = {status = "absent", value = JSON_NULL}
                    elseif type(Ext.DumpExport) ~= "function" then
                        report.components[name] = {
                            status = "unavailable", value = JSON_NULL, error = "Ext.DumpExport is unavailable"
                        }
                    else
                        local component_result = query(name, function()
                            -- Dump immediately; no component userdata escapes this scope.
                            local dump = Ext.DumpExport(actor[name])
                            assert(type(dump) == 'string', 'Ext.DumpExport did not return a JSON string')
                            return dump
                        end)
                        if component_result.status == "ok" then
                            -- Name the raw representation explicitly. Parsing and
                            -- re-encoding engine dumps could lose null/array shape.
                            component_result.dump_json = component_result.value
                            component_result.value = nil
                        end
                        report.components[name] = component_result
                        if component_result.status == "error" then
                            report.errors[#report.errors + 1] = component_result.error
                        end
                    end
                end
            end
        end
    end

    local payload = json(report) .. "\n"
    local saved = false
    if type(output_path) == "string" and output_path ~= "" and Ext and Ext.IO
        and type(Ext.IO.SaveFile) == "function" then
        local save_ok, save_result = pcall(Ext.IO.SaveFile, output_path, payload)
        saved = save_ok and save_result ~= false
        if not saved then
            report.errors[#report.errors + 1] = "Ext.IO.SaveFile failed: " .. tostring(save_result)
            payload = json(report) .. "\n"
        end
    elseif output_path ~= nil then
        report.errors[#report.errors + 1] = "Ext.IO.SaveFile is unavailable; diagnostic was not saved"
        payload = json(report) .. "\n"
    end
    output("[BaldurRL probe] " .. payload)
    return saved, payload
end

return inspect

