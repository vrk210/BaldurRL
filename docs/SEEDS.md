# Encounter seed ledger

Every evaluation, data-collection and selection run draws encounters from
`env.reset(seed=s)`. Reusing a range that has already informed a decision turns a
test into a selection set without anyone noticing. This file records every range
used so far.

**Rules**

- Before a new experiment, reserve its ranges here in the same commit as its
  code or plan. Pick ranges from the unallocated space at the bottom.
- Treat a range as used once any result from it has been looked at, even if the
  run is abandoned.
- Keep training, validation, selection and final-test ranges disjoint.
- 130000–132999 is the frozen M5/M6 final test range. It has never been used.
  Don't use it for development, debugging or selection.
- Learner seeds (PPO `--seed`, agent RNG seeds) are separate from encounter
  seeds. `train.py` resets its training environments with `seed + rank`, which
  touches encounters 0–15 only once each.

| Range | Use | Where recorded |
|---|---|---|
| 0–999 | Unit tests and smoke runs | `tests/` |
| 1000–1199 | PPO checkpoint selection during training, all stages | `train.py`, stage reports |
| 2000–2199 | M2 opening-action diagnostic | `reports/m2_ppo_1m_followup.md` |
| 3000–3999 | Original held-out evaluation, M0–M4 | `reports/m*_heldout_3000_3999.md` |
| 3000–7999 | M5/M6 held-out evaluation, PPO disagreement diagnostics, rollout pilot | `reports/m5_m6_decision_stages.md` |
| 4000–8999 | Fresh 5,000-seed comparisons, M1A/M1B/M2 | `reports/m1a_*`, `m1b_*`, `m2_heldout_*` |
| 9000–13999 | M2 fresh PPO comparison and exact-optimum evaluation | `reports/m2_ppo_1m_followup.md`, `runs/m2_exact_optimum_9000_13999/` |
| 9000–9999 | M2 regret against the exact optimum | `reports/m4_expectimax.md` |
| 10000–14999 | M4 expectimax final test | `reports/m4_expectimax.md` |
| 18000–18099 | M4 rollout-planner pilot | `docs/ROLLOUT_PLANNING.md` |
| 20000–21999 | M5/M6 stage design (development) | `reports/m5_m6_decision_stages.md` |
| 20000–39999 | M4 value data (round 0); M2 value data 20000–41999 | `reports/m4_expectimax.md` |
| 40000–41999 | M4 value validation | `reports/m4_expectimax.md` |
| 42000–43999 | M4 planner selection (behavior check 42000–42499) | `reports/m4_expectimax.md` |
| 40000–44999 | External code review spot checks | M5/M6 planning `PROGRESS.md` |
| 51000–51399 | Transition-model tests and validation | `tests/test_tactical_transitions.py` |
| 60000–67999 | M4 self-play value data | `reports/m4_expectimax.md` |
| 100000–119999 | M5/M6 value-network data | `reports/m5_m6_planning.md` |
| 120000–121999 | M5/M6 value validation (follow-up used 120000–120119) | `reports/m5_m6_planning.md` |
| 122000–123999 | M5/M6 planner configuration selection | `reports/m5_m6_planning.md` |
| **130000–132999** | **Frozen M5/M6 final test. Unused** | |
| 140000–199999 | M5/M6 value data extension; guided-planning data 180000–180659 and review checks 180600–180699 | `reports/m5_m6_planning.md` |
| 210000–210009 | Table transition benchmark | `docs/EXPECTIMAX_PLANNING.md` |
| 210010–212399 | Review of the M5/M6 follow-up: adversarial transition audit (211000–211029), environment fingerprints | `reports/m5_m6_planning.md` |
| 220000–229999 | **Reserved:** M5/M6 headroom probe (rollout-improved heuristic) | Reserved 2026-10-15; the experiment's report replaces this entry |
| 230000–239999 | Simulator tactics mining: 230000–230007 smoke/tests; 232000–235999 once-only paired confirmations (M5/M6); 236000–236127 combat RNG replay of old encounters only. 230008–231999 and 236128–239999 unused | `docs/GAME_KNOWLEDGE.md` (Simulator tactical evidence), `evaluation/sim_mining.py` |
| 240000 and above | Unallocated | |
