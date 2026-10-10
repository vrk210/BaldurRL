# Game knowledge: living implementation reference

This is an implementation reference, not a general BG3 wiki. Labels mean:

- **VERIFIED_BG3 / VERIFIED_API:** A mechanic or API fact explicitly checked against BG3 documentation or observed game behavior. Record its source where practical. An API's documented existence does not prove its live behavior in our encounters.
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

## BG3 observation APIs: VERIFIED_API

The [BG3 Osiris API reference](https://docs.baldursgate3.game) and [Script Extender API documentation](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md) support these API facts. Live encounter behavior and exact identifiers remain separate verification tasks in [BG3_INTEGRATION.md](BG3_INTEGRATION.md).

- [GetHitpoints](https://docs.baldursgate3.game/index.php?title=GetHitpoints) and [GetMaxHitpoints](https://docs.baldursgate3.game/index.php?title=GetMaxHitpoints) query current and maximum HP.
- [CombatStarted](https://docs.baldursgate3.game/index.php?title=CombatStarted), [CombatEnded](https://docs.baldursgate3.game/index.php?title=CombatEnded), [EnteredCombat](https://docs.baldursgate3.game/index.php?title=EnteredCombat), [CombatRoundStarted](https://docs.baldursgate3.game/index.php?title=CombatRoundStarted), [TurnStarted](https://docs.baldursgate3.game/index.php?title=TurnStarted), and [TurnEnded](https://docs.baldursgate3.game/index.php?title=TurnEnded) provide combat and turn lifecycle events. [CombatGetActiveEntity](https://docs.baldursgate3.game/index.php?title=CombatGetActiveEntity) is an active-entity query; it does not by itself enumerate all participants.
- [GetPosition](https://docs.baldursgate3.game/index.php?title=GetPosition) returns world coordinates, with Y vertical; [GetDistanceTo](https://docs.baldursgate3.game/index.php?title=GetDistanceTo) queries distance between objects.
- [GetActionResourceValuePersonal](https://docs.baldursgate3.game/index.php?title=GetActionResourceValuePersonal) queries a named resource using a resource level and returns a numeric amount. Exact names for our abilities still need live verification.
- [HasSpell](https://docs.baldursgate3.game/index.php?title=HasSpell) checks ownership; [CanShowSpellForCharacter](https://docs.baldursgate3.game/index.php?title=CanShowSpellForCharacter) checks visibility. Script Extender documents `SpellBook.Spells` entity access in its [API guide](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md).
- [UsingSpell](https://docs.baldursgate3.game/index.php?title=UsingSpell), [UsingSpellOnTarget](https://docs.baldursgate3.game/index.php?title=UsingSpellOnTarget), and [UsingSpellAtPosition](https://docs.baldursgate3.game/index.php?title=UsingSpellAtPosition) expose spell/ability IDs. The targeted event exposes caster and target IDs.
- [AttackedBy](https://docs.baldursgate3.game/index.php?title=AttackedBy) exposes defender, attacker, damage amount/type, and StoryActionID. The same documentation states regular attacks are handled internally as spells. [MissedBy](https://docs.baldursgate3.game/index.php?title=MissedBy) and [CriticalHitBy](https://docs.baldursgate3.game/index.php?title=CriticalHitBy) events exist.
- Script Extender documents [ECS/entity-component access and `Ext.IO.LoadFile` / `Ext.IO.SaveFile`](https://github.com/Norbyte/bg3se/blob/main/Docs/API.md).

### TODO_VERIFY in a live game

Exact Action and Bonus Action resource IDs; Second Wind and Action Surge ability IDs and resource representation; basic attack spell ID; armor-class ECS field; alive/dead/downed representation; event ordering and StoryActionID correlation reliability; complete UI action legality. None of these identifiers or structures is promoted to `VERIFIED_API`.

## M1A, M1B, and M2 simulator choices

### SIMULATOR_SIMPLIFICATION

- M1A samples enemy HP, AC, attack bonus, die size, and damage bonus independently at reset. These ranges are for policy-learning experiments and are not verified BG3 creature distributions.
- M1B uses two enemy slots in fixed melee range. Living enemies attack in slot order after Fighter `END_TURN`.
- M2 Cleave is a once-per-encounter ability represented by a single resource. This models a short-rest-recharge weapon ability for the simulator; the encounter does not implement rests.
- M2 Cleave makes a separate normal attack roll against each living enemy and deals half of resolved weapon damage on a hit, floored with minimum one. This calculation is an explicit simulator simplification, not a verified complete BG3 Cleave implementation.
- M1A, M1B, and baseline M2 use terminal rewards. The M2 reward experiment adds bounded progress from Fighter damage during training only.
- M3 is a 2v2 simulator stage: two preset Fighters with per-ally Cleave, Second Wind, and Action Surge pools act in fixed `ALLY 0, ALLY 1` order against two M1B-sampled enemies. Combat continues while any ally lives; enemies attack the lowest living ally slot.
- M3 uses terminal rewards only. Its observation adds `active_actor_index` before fixed ally and enemy slots; evaluation counts a loss only when every ally is dead and reports total remaining ally HP.
- M4 is the 2v3 counterpart with scaled enemy sampling (HP 12–24, attack +3–+7, damage bonus 2–6; same AC range and damage dice as M1B). These ranges are a first fairness guess placing enemies near Fighter potency, not verified BG3 distributions. M4 uses terminal rewards only.

No new behavior in this section is labeled `VERIFIED_BG3`.

## M5 and M6 simulator choices

### SIMULATOR_SIMPLIFICATION (owner-authorized designs, not BG3 rules)

The project owner authorized designing these mechanics for decision headroom. They are specified in [STAGES_SPEC.md](STAGES_SPEC.md) and must not be cited as BG3 behavior:

- Enemy roles Brute, Archer, and Healer, one of each per M5/M6 encounter, with role-specific stat ranges; the Healer heals the most-injured enemy at or below half HP (2d6+2, limited charges) instead of attacking.
- Enemy targeting policies `WEAKEST`, `RETALIATE` (last ally to attack or trip it), and `RANDOM`, sampled per enemy and visible to the policy.
- Advantage/disadvantage: two d20, keep higher/lower; any advantage plus any disadvantage cancel; sources never stack.
- Trip: a Bonus Action plus one of two per-encounter charges; an attack roll against AC knocks the target Prone. Prone grants advantage to ally melee attacks and disadvantage to the Prone creature's attacks until the end of its next turn.
- Dodge: the Action; enemy attacks against the ally have disadvantage until its next turn.
- M6 ranks: Brute in front, Archer and Healer behind; allies are at the line or deep; reach, the engaged-Archer disadvantage, Movement, Reaction, Advance, opportunity attacks, and Disengage follow the stage specification. There are no distances or grid.

Advantage, Prone, Dodge, Disengage, opportunity attacks, and Reactions resemble BG3/5e concepts, but every number, duration, and trigger above is a simulator choice.

## Future game mechanics: TODO_VERIFY / outside the implemented stages

Initiative; movement and movement distance; jumping; range; line of sight; terrain; elevation; actual BG3 advantage/disadvantage sources; saving throws; status effects beyond the M5/M6 simulator conditions; spell slots; concentration; items; potions; multiple attacks beyond the specified enemy slots; multiple party members; death saving throws; actual short rests; long rests; other class-specific resources; actual BG3 enemy AI. (M5/M6 implement simplified simulator versions of a few of these; see above.)

Listing a topic here does **not** authorize its implementation. These are future research and design questions.

## Rule for additions

1. Label new knowledge `VERIFIED_BG3`, `M0_SIMPLIFICATION`, `SIMULATOR_SIMPLIFICATION`, or `TODO_VERIFY`.
2. Cite or describe the source of verified mechanics where practical.
3. Never silently promote `TODO_VERIFY` material into implementation behavior.
4. Update `M0_SPEC.md` if a mechanic becomes part of the current environment.
