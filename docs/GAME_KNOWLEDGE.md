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

## Simulator tactical evidence (hypotheses for BG3)

Use these as simulator-backed starting rules, then verify the named mechanics
in BG3. **A** = paired single-rule intervention with simultaneous CI excluding
zero **and** Bonferroni-adjusted exact McNemar p<.05; **B** = replayed local
counterfactual; **C** = observational hypothesis. Brackets are 95% CIs; new A
intervals adjust across all 13 predeclared contrasts. Effects are win-rate
percentage points (pp), conditional on the tested continuation, not guarantees.
New target/resource tests use M5, positioning uses M6, and Dodge uses both.
Resource effects assume fresh encounter-start pools.

1. **TODO_VERIFY — Focus an affordable kill; do not automatically prioritize the healer.**
   **A · general:** both allies using lowest current target HP beat changing
   ally 2 to highest-HP targeting by +22.38 pp [+19.78, +24.97]. This tests
   that exact priority change, not a universal value of concentrating attacks.
   **A · mechanic-dependent:** lowest-HP targeting beats healer-first by
   +4.05 pp [+1.93, +6.17]; check actual healing triggers, charges and
   healer durability. **C · general:** expected overkill is
   13.76% [13.60, 13.92] of modeled uncapped M5 damage.
   Consider low-damage finishers, but this is not a proven benefit of switching
   targets. Logs cap damage; waste is reconstructed from exact dice given each
   observed hit/critical. **C · mechanic-dependent:** modal full kill orders
   are Archer→Healer→Brute in 54.92% [53.54, 56.29] of all
   M5 fights and Brute→Archer→Healer in 33.08% [31.79, 34.40]
   on M6; actual role stats/healing need checking.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

2. **TODO_VERIFY — Spend burst damage while it can remove enemy turns; heal before critical
   danger.** **A · general:** early Action Surge beats withholding it until
   round 2 by +11.70 pp [+8.91, +14.49]. **A · mechanic-dependent:** healing
   at 12/20 HP instead of 10/20 gains +1.65 pp [+0.51, +2.79]; verify BG3 healing
   amount, action cost and incoming damage before adopting the threshold.
   **B · mechanic-dependent:** replacing PPO's Attack with the heuristic's
   focus Trip in sampled disagreement states improves local win probability by +1.53 pp [+0.82, +2.23] on M5 and
   +0.94 pp [+0.37, +1.51] on M6. Targets can differ; this does not isolate same-target timing.
   Check Prone duration, accuracy,
   advantage and its competition with healing. **Unresolved intervention · mechanic-dependent:**
   requiring 3 targets rather than 2 for half-damage Cleave gives
   +0.97 pp [-0.09, +2.04]; no validated universal Cleave threshold or
   threat-target Trip advantage. Verify Cleave's real damage and reach.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

3. **TODO_VERIFY — Commit allies to one positioning plan.** **A · mechanic-dependent:**
   adaptive positioning beats stopping after one diver by
   +6.78 pp [+4.88, +8.67], holding everywhere by +2.00 pp [+0.03, +3.97], and
   diving everywhere by +1.80 pp [+0.06, +3.54]. The tested trigger is front
   enemy HP / normal hit chance ≥42; dive when durable, otherwise break the
   front together. Check real reach, path costs and ranged disadvantage.
   **A · mechanic-dependent:** taking the opportunity attack beats
   Disengaging before a planned dive while Surge remains by
   +9.18 pp [+7.11, +11.24]. This does not show Disengage never helps.
   Earlier Advance timing remains unresolved in local replays.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

4. **TODO_VERIFY — Keep attacking by default; do not add the tested emergency Dodge rule.**
   **A · mechanic-dependent:** disabling Dodge when estimated incoming damage
   reaches 75% of current HP gains +1.70 pp [+0.48, +2.92] on M5 and
   +3.57 pp [+2.07, +5.08] on M6. This rejects that trigger, not every defensive
   use. Check real enemy targeting, Dodge duration and action economy; no
   beneficial Dodge state was established by this work.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

5. **TODO_VERIFY — Audit legal actions before ending a turn.** **A · general:** spending
   Surge versus leaving it unused gains +31.18 pp [+28.31, +34.04].
   **C · general:** tuned/adaptive never end with a legal attack in these logs;
   final PPO s0 does so on 1.86% [1.54, 2.18] /
   5.19% [4.59, 5.78] of M5/M6 END_TURN decisions.
   These frequencies do not measure the cost of each skipped attack. A legal
   heal or Trip can reasonably be saved; availability alone is not waste.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

6. **TODO_VERIFY — Do not label one lost fight unwinnable.** **C · unknown:** high total enemy
   HP (M5) and durable front enemies (M6) predict more baseline losses.
   M6 durable-minus-soft front loss rate: +7.45 pp [+5.25, +9.65].
   Every sampled old loss won under some of the 128 replay dice seeds;
   fixed-policy replay rates span 58.6–89.8% on M5 and 43.0–90.6%
   on M6. These selected 32 losses per stage do not identify the irreducible
   dice share of the remaining losses. That needs an optimal-policy bound;
   verify real encounter balance and randomness before transferring.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

7. **TODO_VERIFY — Use PPO's exceptions as review examples, not replacement rules.**
   **C · unknown:** PPO s0/s1/s2 win 540/696/632 M5 and 604/615/610
   M6 encounters that the best heuristic loses, out of 5,000 paired encounters
   each. None of the predeclared opening-feature strata establishes PPO
   superiority, and no tested local decision type reliably favors PPO.
   Pooled PPO-minus-heuristic gaps are -4.14 pp [-5.31, -2.97]
   / -5.39 pp [-6.59, -4.20] (M5/M6). Changed dice paths can
   produce these unique wins; inspect concrete BG3 states before inferring a
   transferable tactic.

   Reproducer: `evaluation.sim_mining_render --from-summary`.

Sources, per-policy audits, full kill orders, inventory and provenance:
[`evaluation/sim_mining.py`](../evaluation/sim_mining.py) and its committed
[`sim_mining_summary.json`](../evaluation/sim_mining_summary.json).
