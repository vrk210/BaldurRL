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
- Controlled vs automatic: the policy decides for refs in
  `controlled_refs` (currently `{ALLY 0}`); `END_TURN` advances the manager
  and runs `_run_automatic_turn` for each following non-controlled ref.
  Enemy turns stay hidden from the policy and attack `ALLY 0`.
- Per-turn refresh: each `Character` carries an immutable `turn_refresh`
  mapping (`Fighter: ACTION, BONUS_ACTION`; `Goblin: ACTION`).
  `refresh_turn_resources(actor)` runs at the start of that actor's actual
  turn. Second Wind, Action Surge, and Cleave never refresh. Dead skipped
  actors are not refreshed.
- Rounds: round 1 starts with the first allied turn. Advancing past the
  final living ref wraps to the first ref and increments the round.
  `MAX_ROUNDS = 50` truncates instead of wrapping or refreshing, and
  termination stops the loop before any wrap or refresh.

## Future 2vX sketch (not implemented)

```text
allies: ALLY 0, ALLY 1
enemies: ENEMY 0, ENEMY 1, ENEMY 2 (or fixed slots)
TurnManager order: ALLY 0, ALLY 1, ENEMY 0, ENEMY 1 (, ENEMY 2)
policy: same policy controls whichever allied actor is active
future observation: active_actor_index, fixed ally slots, fixed enemy slots
```

Real BG3 initiative, interleaved/grouped turns, and movement remain future
mechanics and are not guessed here.
