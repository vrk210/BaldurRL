# Combat turn model (internal refactor)

Internal organization for M0/M1A/M1B/M2 turns. Public observations, action
indices, masks, rewards, and seeded transitions are unchanged.

## Pieces

- `ActorRef(side, slot)`: immutable address such as `ALLY/0` or `ENEMY/1`.
  No `Character` pointer; resolve through the roster.
- `CombatRoster(allies, enemies)`: stable tuples plus `get`, `is_alive`,
  `living_allies`, `living_enemies`, and `side_defeated`. No combat rules.
- `TurnManager(order, round_number)`: deterministic order, current actor,
  round number, `peek_next`, and `advance` with dead-actor skipping and
  wrap detection. No dice, HP mutation, rewards, or attacks.
  An optional keyword `current=ActorRef(...)` restores an actor in the order
  without advancing the round or refreshing resources. Observation-based
  [simulation reconstruction](ROLLOUT_PLANNING.md) uses this for partial turns.
- Controlled vs automatic: the policy decides for refs in
  `controlled_refs` (currently `{ALLY 0}`); `END_TURN` advances the manager
  and runs `_run_automatic_turn` for each following non-controlled ref.
  Automatic turns use the shared mask path (fixed policy: first legal
  `ATTACK` in stage order, today always `ALLY 0`) and stay hidden.
- Execution dispatches on the active actor: `step()` resolves the
  attacker as `turns.current` and targets via `resolve_target`, and
  staged rewards derive from side defeat rather than Fighter HP.
- Decisions and legality: `DecisionSpec(action, target_index)` lives in
  `combat/legality.py` (re-exported from `combat/stages.py`).
  `resolve_target` maps a target slot to an `ActorRef` on the acting
  side's opposing tuple (`None` means slot 0); out-of-range raises.
  `legal_mask_for` computes one actor's selectable set from its own
  `known_abilities`, resource costs, and effect rules, using the same
  checks as execution. Decisions naming a nonexistent slot for the
  acting side are illegal, not an error.
- Turn gating (decided: gate at the caller): `legal_mask_for` is
  turn-agnostic — "what could this actor do on its turn." Environments
  only ever query `turns.current`, and expose all-false unless the
  current actor is policy-controlled; `TurnManager` keeps answering
  whose-turn-it-is and legality keeps answering what-can-they-do.
- Per-turn refresh: each `Character` carries an immutable `turn_refresh`
  mapping (`Fighter: ACTION, BONUS_ACTION`; `Goblin: ACTION`).
  Per-turn resources refresh whenever an actor's actual turn begins,
  whether that actor is policy-controlled or automatic.
  On reset, the environment refreshes the initial actor. On `END_TURN`,
  `_advance_to_next_turn(turns, roster)` advances to the next living actor
  and calls `refresh_turn_resources(actor)` before deciding whether to
  expose a controlled decision or execute an automatic turn.
  `_run_automatic_turn()` executes the fixed policy and does not refresh.
  Second Wind, Action Surge, and Cleave never refresh. Dead skipped actors
  are not refreshed.
- Rounds: round 1 starts with the first allied turn. Advancing past the
  final living ref wraps to the first ref and increments the round.
  `MAX_ROUNDS = 50` truncates instead of wrapping or refreshing, and
  termination stops the loop before any wrap or refresh.

## 2vX is implemented as M3 (2v2) and M4 (2v3)

```text
allies: ALLY 0, ALLY 1
enemies: ENEMY 0, ENEMY 1 (, ENEMY 2 on M4)
TurnManager order: ALLY 0, ALLY 1, ENEMY 0, ENEMY 1 (, ENEMY 2)
policy: same policy controls whichever allied actor is active
observation: active_actor_index, fixed ally slots, fixed enemy slots
```

See [STAGES_SPEC.md](STAGES_SPEC.md) for the normative M3/M4 contracts. M4
scales enemy sampling toward Fighter parity as a first fairness guess.
Larger encounters, real BG3 initiative, interleaved/grouped turns, and
movement remain future mechanics and are not guessed here.

## M5/M6 turn hooks

`TacticalCombatEnv` uses the same `TurnManager` order and round/truncation
logic, with two lifecycle hooks from `combat/tactics.py`: `begin_tactical_turn`
(refresh per-turn resources, end an ally's Dodging and Disengaged) and
`end_tactical_turn` (an enemy stands up from Prone; an ally's Disengaged ends).
Enemy turns are automatic (`run_enemy_turn`: Healer heal or targeted attack).
In M6 an opportunity attack can kill the active ally during its own turn; the
turn then passes exactly as if it had chosen `END_TURN`.
