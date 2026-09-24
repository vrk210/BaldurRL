# Game knowledge: living implementation reference

This is an implementation reference, not a general BG3 wiki. Labels mean:

- **VERIFIED_BG3:** A mechanic or API fact explicitly checked against BG3 documentation or observed game behavior. Record its source where practical. No BG3 fact is marked verified here yet.
- **M0_SIMPLIFICATION:** A deliberate prototype choice that may differ from full BG3.
- **TODO_VERIFY:** Possible BG3 behavior or API detail requiring verification before implementation.

**TODO_VERIFY information must never be treated as implementation truth.** `docs/M0_SPEC.md` governs current simulator behavior.

## Current M0 knowledge

### M0_SIMPLIFICATION

- Exactly one Fighter versus one Goblin; Fighter always acts first; both begin within melee range.
- Movement, terrain, elevation, line of sight, and opportunity attacks are ignored.
- Goblin always attacks the Fighter when alive and has no alternative decision-making.
- Second Wind and Action Surge are once-per-encounter abilities.
- Fighter is Level 2.
- Initial rewards are terminal win/loss only.
- M0 represents actions, bonus actions, Second Wind, and Action Surge as integer resources. Action Surge grants one additional Action, allowing up to two Actions in a Fighter turn.
- The fixed M0 preset values are Fighter 20 HP / AC 16 / +5 attack / 1d8+3 damage and Goblin 15 HP / AC 15 / +4 attack / 1d6+2 damage. These are simulator presets, not `VERIFIED_BG3` stat blocks.
- One environment step is one Fighter decision. The fixed Goblin attack happens only after `END_TURN`; round 50 truncates if neither combatant has died.

### CURRENT_COMBAT_RULE (M0 only)

- Attack: `1d20 + attack_bonus` versus Armor Class.
- Natural 1: automatic miss. Natural 20: automatic hit and critical hit.
- Critical damage doubles damage dice but not the static damage modifier.

These are M0 implementation rules; this section does not claim they have been independently verified against BG3.

### CURRENT_FIGHTER_RULE (M0 only)

- Fighter resources: Action, Bonus Action, Second Wind, Action Surge.
- `ATTACK` consumes Action. `SECOND_WIND` consumes Bonus Action and heals `1d10 + 2` in M0.
- `ACTION_SURGE` grants an additional Action and becomes unavailable afterward.
- The Fighter knows `ATTACK`, `SECOND_WIND`, and `ACTION_SURGE`; the Goblin knows `ATTACK`. Shared immutable ability definitions describe costs and targets. Resource affordability alone does not establish living-target or missing-HP legality.

## Future BG3 instrumentation: TODO_VERIFY

Investigate Osiris events, queries, and calls; Script Extender / Lua; entity/component state; combat turn events; HP state; spell/ability use; action resources; and legal-action extraction. The exact API mapping is **TODO_VERIFY** unless explicitly documented as verified elsewhere in this repository.

## Future game mechanics: TODO_VERIFY / outside M0

Initiative; movement and movement distance; jumping; range; line of sight; terrain; elevation; advantage/disadvantage; saving throws; status effects; spell slots; concentration; items; potions; multiple attacks; multiple enemies; multiple party members; death saving throws; short rests; long rests; class-specific resources; enemy AI abilities.

Listing a topic here does **not** authorize its implementation. These are future research and design questions.

## Rule for additions

1. Label new knowledge `VERIFIED_BG3`, `M0_SIMPLIFICATION`, or `TODO_VERIFY`.
2. Cite or describe the source of verified mechanics where practical.
3. Never silently promote `TODO_VERIFY` material into implementation behavior.
4. Update `M0_SPEC.md` if a mechanic becomes part of the current environment.
