-- Stage immutable JSON; a Python publisher provides atomic .json publication.
local E = {}
function E.new(io, wire, directory)
    assert(io.LoadFile(directory .. '/session.txt') == nil, 'Session directory already reserved')
    assert(io.SaveFile(directory .. '/session.txt', 'BaldurRL H0 schema_version=1\n'), 'Cannot reserve session')
    assert(io.LoadFile(directory .. '/session.txt') == 'BaldurRL H0 schema_version=1\n', 'Session reservation readback failed')
    local self = {sequence = 0, halted = false}
    function self:emit(recordType, facts)
        assert(not self.halted, 'Emitter halted; start a new session')
        local ok, result = pcall(function()
            assert(recordType == 'event' or recordType == 'snapshot', 'Unknown record type')
            local record = recordType == 'event' and wire.event(facts) or wire.snapshot(facts)
            local sequence = self.sequence + 1
            record.record_type, record.schema_version = recordType, 1
            record.sequence, record.timestamp_ms = sequence, wire.null
            local payload = wire.encode(record) .. '\n'
            local stem = directory .. '/record-' .. string.format('%08d', sequence)
            assert(io.LoadFile(stem .. '.pending') == nil, 'Record already exists')
            self.sequence = sequence -- Never reuse a number after an attempted write.
            assert(io.SaveFile(stem .. '.pending', payload), 'Record write failed')
            assert(io.LoadFile(stem .. '.pending') == payload, 'Record readback mismatch')
            assert(io.SaveFile(stem .. '.ready', tostring(#payload) .. '\n'), 'Completion write failed')
            return sequence
        end)
        if not ok then self.halted = true; error(result) end
        return result
    end
    return self
end
return E
