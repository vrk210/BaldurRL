# M5/M6 decision-rich stages: design, tuning, baselines, and PPO

Verdict: both new stages show far more measurable decision headroom than M4. On held-out seeds 3000–7999 (5,000 encounters per policy), the best hand-written heuristic beats the naive M4-style heuristic by **+17.1 points on M5** (78.5% vs 61.4%) and **+39.7 points on M6** (80.6% vs 40.9%). Smaller but significant gaps also separate the smarter heuristics from each other. MaskablePPO trained for 1M steps (seeds 0/1/2, default hyperparameters) learns most of the gap but **does not beat the best hand heuristic on either stage**: it is 4.1 points behind on M5 and 5.4 points behind on M6 (pooled, paired SE ≈ 0.6). Corrected stratified diagnostics support a local benefit from Trip before Attack on both stages and Trip before Cleave on M5, under heuristic continuation; they do not establish a consistent benefit from replacing Attack with Cleave or advancing earlier. Whole-fight Trip/Cleave repairs recover about 1.05 points of M5's 4.14-point pooled gap, so the rest remains unattributed. On M6, PPO s0 remains 4.42 points behind adaptive after forcing its opening dive/hold plan. A one-step rollout planner with the default 32 rollouts per action does much worse than its own continuation heuristic (finite-sample selection error), so it is not yet a useful headroom probe. Part of the naive-to-best spread is simply "use the new Trip button"; the finer-grained headroom among strong policies is 1–5 points.

All M5/M6 rules are **simulator choices** authorized by the project owner and specified in [STAGES_SPEC.md](../docs/STAGES_SPEC.md#m5--roles-and-conditions). They are not verified BG3 rules.

## Design summary

**M5 (2 Fighters vs Brute, Archer, Healer; no geometry).** It keeps the M4 Fighter preset, d20 rules, Second Wind, Action Surge, and Cleave. It adds:

- Roles are a random permutation over the three enemy slots, with role-specific stat ranges. The Healer heals the most-injured enemy at or below half HP (2d6+2, 3 charges) instead of attacking.
- Each enemy samples a targeting policy that the policy can observe: `WEAKEST`, `RETALIATE` (the last ally that attacked or tripped it), or `RANDOM`.
- Advantage and disadvantage as a mechanics primitive.
- Trip: Bonus Action plus one of 2 per-encounter charges; an attack roll against AC knocks the target Prone. While Prone, a target grants advantage to ally attacks and its own attacks have disadvantage, until the end of its next turn.
- Dodge: costs the Action; enemy attacks against the ally have disadvantage until its next turn.
- 11 flat masked actions; 63 float32 observation fields; terminal ±1 reward only.

**M6 (M5 plus two ranks).** The Brute stands in the front rank; the Archer and Healer stand in the back. While the Brute lives, allies at the line can only melee the Brute. `ADVANCE` (per-turn Movement) moves an ally deep, where it reaches everyone, but first provokes an opportunity attack from the Brute if the Brute still has its Reaction. `DISENGAGE` (Action) prevents that opportunity attack. A deep ally or a broken front gives the Archer disadvantage. An opportunity attack can kill the active ally on its own turn, which ends its turn. The M6 Brute has HP 20–26 and the Healer 2 charges. 13 actions; 75 observation fields.

Every transition-relevant fact is observed, so `simulation_from_observation` reconstructs M5/M6 exactly. Restored simulators match reseeded live environments step for step in tests, and the existing `RolloutAgent` runs on both stages unchanged.

## Tuning history (one retune)

Targets were set before measuring: a naive heuristic between about 35% and 65%, and a spread of at least 8–10 points to the best hand heuristic, without the best heuristic at the ceiling.

**v1** (first design, held-out 3000–7999, n=5000). Brute +4–+6 / 1d10+2–4; Archer +5–+7 / 1d8+2–4; Healer 1d8+2 with 3 charges; the same ranges in M6; Trip limited only by the Bonus Action.

| v1 policy | M5 win rate (SE) | v1 policy | M6 win rate (SE) |
|---|---|---|---|
| random | 19.48% (0.56) | random | 11.86% (0.46) |
| naive | 75.92% (0.60) | naive | 44.84% (0.70) |
| trip_focus | **93.96%** (0.34) | hold (kill Brute first, Trip, threat order) | **84.60%** (0.51) |
| healer_first | 92.30% (0.38) | dive (one ally advances) | 79.54% (0.57) |
| threat (+Dodge) | 92.18% (0.38) | dive_all (both, Disengage if Surge) | 84.32% (0.51) |

Diagnosis: M5 was too easy (naive 75.9%). Almost the whole spread came from one free, nearly always-correct button: Trip every turn. That left the best policies near the ceiling instead of creating real choices.

**v2** (final). Four changes:

1. Trip gets 2 charges per ally per encounter, which turns it from a free buff into an allocation decision.
2. The Healer heal becomes 2d6+2.
3. Brute and Archer get +1 attack bonus and +1 damage bonus.
4. M6 only: Brute HP 20–26 and Healer 2 charges, to offset the rank wall.

I chose these from a small set of candidates, each evaluated with the hand heuristics on **development seeds 20000–21999 (n=2000, never the held-out range)**:

| Candidate (dev seeds) | M5 naive / trip_focus / healer_first / threat |
|---|---|
| c1: Trip ×2 | 75.3 / 87.7 / 84.5 / 86.0 |
| c2: c1 + heal 2d6+2 | 72.3 / 85.4 / 84.1 / 83.5 |
| c3: c2 + Brute/Archer +1 attack | 67.4 / 82.4 / 79.8 / 79.7 |
| **c4: c3 + Brute/Archer +1 damage (chosen for M5)** | **60.0 / 76.5 / 73.5 / 72.9** |
| c5: c1 + unlimited heals | 74.8 / 87.5 / 84.3 / 85.7 |

On M6, c4 pushed naive to 23.1%. With a better dive heuristic (both allies advance, no Disengage), diving beat holding (70.4% vs 61.9%), while a single diver did badly (46.0%). Brute HP sweeps with c4 (naive / hold / dive-all) gave 22–28: 29.5 / 71.4 / 73.4; 20–26: 36.0 / 76.1 / 75.5; 18–24: 43.5 / 81.1 / 77.4; 16–22: 50.5 / 85.4 / 79.7. I chose Brute HP 20–26 plus Healer 2 charges (39.8 / 78.1 / 77.3) over Brute 20–26 plus weaker Archer damage (39.6 / 77.9 / 76.2), because it keeps the M5 Archer unchanged and balances "hold" and "dive" on average. The final heuristic variants were developed afterwards on the same dev seeds; their tuning thresholds (Second Wind ≤12, Cleave at 3 targets, dive threshold 42) also come from dev seeds only.

v1 was measured before the `trip_count` observation field existed; its transitions are identical to `V1_RULES` in code.

## Final (v2) results on held-out seeds 3000–7999

The heuristics are deterministic. Random uses agent seed 0. PPO is MaskablePPO with 1,003,520 steps per run, 4 envs, train.py defaults, final models, and deterministic masked prediction; each run took about 7 minutes. Model selection during training used seeds 1000–1199. SE is the binomial standard error. "First kill" is the share of fights whose first enemy death was the Archer, Healer, or Brute.

### M5

| Policy | Wins | Win rate (SE) | Rounds | Final ally HP (/40) | SW fights | Cleave fights | First kill A/H/B % |
|---|---|---|---|---|---|---|---|
| random | 335/5000 | 6.70% (0.35) | 6.95 | 1.2 | 79% | 87% | 29/19/1 |
| naive (M4 rules) | 3072/5000 | 61.44% (0.69) | 6.37 | 13.9 | 86% | 100% | 63/31/0 |
| trip_focus | 3831/5000 | 76.62% (0.60) | 6.04 | 19.0 | 81% | 100% | 69/29/0 |
| healer_first | 3693/5000 | 73.86% (0.62) | 5.49 | 18.3 | 79% | 100% | 3/93/0 |
| **tuned** (best) | **3926/5000** | **78.52% (0.58)** | 5.92 | 20.4 | 86% | 100% | 69/29/0 |
| PPO s0 | 3745/5000 | 74.90% (0.61) | 6.35 | 19.1 | 92% | 100% | 77/20/0 |
| PPO s1 | 3734/5000 | 74.68% (0.61) | 6.54 | 19.1 | 92% | 80% | 71/27/0 |
| PPO s2 | 3678/5000 | 73.56% (0.62) | 6.45 | 19.0 | 92% | 100% | 84/14/0 |
| PPO pooled | 11157/15000 | 74.38% | | | | | |

Heuristics:

- **naive** applies the M4 priorities and never trips, dodges, or advances: Second Wind at ≤10 HP, Cleave while ≥2 enemies live, attack the lowest-HP target, Surge for a second attack.
- **trip_focus** is naive targeting, but trips the focus target before attacking it.
- **healer_first** focuses the Healer while it has charges, then the Archer, then the Brute, with the same Trip rule.
- **tuned** is trip_focus with Cleave only against 3 living enemies and Second Wind at ≤12 HP.

### M6

| Policy | Wins | Win rate (SE) | Rounds | Final ally HP (/40) | SW fights | Cleave fights | First kill A/H/B % |
|---|---|---|---|---|---|---|---|
| random | 310/5000 | 6.20% (0.34) | 6.39 | 1.2 | 76% | 86% | 17/12/5 |
| naive (M4 rules) | 2046/5000 | 40.92% (0.70) | 5.97 | 9.5 | 85% | 100% | 0/0/54 |
| hold | 3944/5000 | 78.88% (0.58) | 5.69 | 24.0 | 69% | 0% | 0/0/86 |
| dive_all | 3968/5000 | 79.36% (0.57) | 5.30 | 20.1 | 88% | 100% | 68/30/0 |
| **adaptive** (best) | **4030/5000** | **80.60% (0.56)** | 5.54 | 22.8 | 77% | 48% | 32/14/47 |
| PPO s0 | 3787/5000 | 75.74% (0.61) | 6.10 | 20.7 | 90% | 20% | 21/24/48 |
| PPO s1 | 3768/5000 | 75.36% (0.61) | 6.03 | 20.9 | 91% | 47% | 37/8/47 |
| PPO s2 | 3726/5000 | 74.52% (0.62) | 5.88 | 19.6 | 94% | 58% | 32/32/28 |
| PPO pooled | 11281/15000 | 75.21% | | | | | |

Heuristics:

- **hold** is tuned M5 play that never advances; it Cleaves only with 3 reachable targets, which never happens while holding.
- **dive_all** has both allies `ADVANCE` at their first opportunity, taking the opportunity attack.
- **adaptive** dives with both allies only when the Brute's HP divided by our hit chance is ≥42 (committing once a partner is deep); otherwise it holds.

None of the final heuristics uses Dodge or Disengage, because both hurt on dev seeds. No policy truncated.

### Paired comparisons (same 5,000 seeds)

The difference is the mean per-seed win difference; the paired SE comes from per-seed differences. Seeds fix each encounter's sampled enemies, but dice diverge once policies diverge, so pairing removes encounter variance, not all noise. Produced by `evaluation.paired` (runs/v2/analysis/paired_table.py).

| Stage | Comparison | Difference (pts) | Paired SE | z |
|---|---|---:|---:|---:|
| M5 | tuned − naive | +17.08 | 0.86 | +19.8 |
| M5 | trip_focus − naive (Trip alone) | +15.18 | 0.88 | +17.4 |
| M5 | trip_focus − healer_first (target rule) | +2.76 | 0.69 | +4.0 |
| M5 | tuned − trip_focus (resource timing) | +1.90 | 0.43 | +4.4 |
| M5 | PPO s0 / s1 / s2 − tuned | −3.62 / −3.84 / −4.96 | 0.71 / 0.79 / 0.77 | −5.1 / −4.8 / −6.4 |
| M5 | PPO pooled − tuned | −4.14 | 0.60 | −6.9 |
| M5 | PPO pooled − naive | +12.94 | 0.74 | +17.4 |
| M6 | adaptive − naive | +39.68 | 0.87 | +45.4 |
| M6 | hold − naive | +37.96 | 0.86 | +43.9 |
| M6 | dive_all − hold | +0.48 | 0.81 | +0.6 |
| M6 | adaptive − hold | +1.72 | 0.59 | +2.9 |
| M6 | adaptive − dive_all | +1.24 | 0.55 | +2.3 |
| M6 | PPO s0 / s1 / s2 − adaptive | −4.86 / −5.24 / −6.08 | 0.76 / 0.77 / 0.78 | −6.4 / −6.8 / −7.8 |
| M6 | PPO pooled − adaptive | −5.39 | 0.61 | −8.8 |
| M6 | PPO pooled − naive | +34.29 | 0.77 | +44.3 |

### PPO diagnostics

- **Checkpoint selection does not close the gap.** Model-selection-best checkpoints (by seeds 1000–1199) score M5 74.78 / 76.64 / 73.28% and M6 75.16 / 76.06 / 75.26% on the held-out seeds. All are below tuned (78.52%) and adaptive (80.60%).
- **Learning plateaus.** Seed-0 checkpoints on held-out seeds 3000–4999 (n=2000, SE ≈ 1.0) at 250k / 500k / 750k / 1M steps: M5 72.8 / 76.2 / 74.8 / 74.6%; M6 73.3 / 76.2 / 73.8 / 73.9%. More of the same training is unlikely to close the gap without changes to architecture, hyperparameters, or exploration. These checkpoints were inspected after training and were not used for selection.

## Behavior notes

Computed with `evaluation.tactical_behavior` over the held-out traces. "Trip→attack" is the share of Trips followed, in the same ally turn, by an Attack on that target or a Cleave.

| M5 policy | Trips/fight | Trip→attack | Trip targets B/A/H | First kill Archer | Dodges/fight |
|---|---:|---:|---|---:|---:|
| trip_focus | 3.91 | 0.88 | 20/37/43% | 69% | 0 |
| tuned | 3.90 | 0.90 | 23/36/41% | 69% | 0 |
| PPO s0 / s1 / s2 | 3.90 / 3.89 / 3.92 | 0.76 / 0.60 / 0.62 | ≈30–37/31–37/28–33% | 77 / 71 / 84% | 0.07 / 0.37 / 0.17 |

| M6 policy | Fights with Advance | Mean advancing allies | Advance rate, tanky / soft Brute | Win rate, tanky / soft Brute | Trip→attack |
|---|---:|---:|---|---|---:|
| hold | 0% | 0 | 0 / 0% | 73.1 / 84.2% | 0.89 |
| dive_all | 100% | 2.00 | 100 / 100% | 76.7 / 81.8% | 0.91 |
| adaptive | 48% | 0.95 | 100 / 0% | 76.7 / 84.2% | 0.89 |
| PPO s0 / s1 / s2 | 87 / 83 / 97% | 1.51 / 1.48 / 1.80 | 93/81, 88/77, 97/97% | 70.5/80.5, 71.3/79.1, 70.5/78.2% | 0.67 / 0.63 / 0.66 |

"Tanky" means opening Brute HP divided by hit chance ≥42 (47.7% of held-out encounters).

- **Kill priority (M5).** Killing the Healer first is *worse* than lowest-HP focus (−2.8 points). The Healer heals itself once at or below half HP, which absorbs focus fire, while the low-HP, high-damage Archer dies quickly. PPO learned Archer-first (71–84% of first kills) rather than Healer-first.
- **Trip / Prone (M5).** The competent M5 policies use about 3.9 Trip charges per fight; roughly 89–92% of their fights use all four. PPO sequences them worse: only 60–76% of its Trips are followed by an attack on the Prone target in the same turn, versus about 90% for the heuristics. PPO trips the Brute more often, which may be defensive tripping (disadvantage on the Brute's attack) or simply poor sequencing; I did not separate the two. PPO sometimes Dodges, which the heuristics never do.
- **Hold vs dive (M6).** On average, holding and diving are equally good (+0.5 ± 0.8). The better choice depends on the encounter: diving wins against durable Brutes and holding wins against soft ones, and the adaptive rule that reads this beats both. Splitting focus by sending one diver at the opening was weak on dev seeds (61.0%, vs 77–79% for both or neither). PPO learned to advance in round 1 in most fights, and its dive rate barely depends on Brute durability (s2 dives in 97% of fights either way). PPO's advance count per fight is not evidence of a lone-diver habit: advancing becomes illegal once the Brute dies, so counts depend on how the fight went. PPO fights with 0 / 1 / 2 advances won 97–98% / 89–94% / 65–71%.
- **Disengage** is never used by any final policy, including PPO. The Brute's opportunity attack costs about 5.5 HP in expectation (held-out mean). Disengage avoids it but costs the Action (about 5–6 expected damage dealt). On dev seeds under candidate c4, diving with both allies won 51.4% when each Disengaged whenever it still had Action Surge, versus 70.4% for the same plan without Disengage. Under this tuning Disengage looks rarely, if ever, worth taking; I did not prove that it is dominated.

## PPO disagreements: corrected stratified diagnostic

The initial diagnostic kept the first 12 spaced occurrences of each disagreement type. Its frequency counts covered all 15,000 fights, but its rescored states came predominantly from early encounters of one model. The values below replace those estimates with the corrected diagnostic from 2026-10-10 (`evaluation.ppo_diagnostics`; the old `runs/v2/analysis/ppo_gap.py` entry point now delegates to it).

For each of the six most frequent types, I uniformly sampled 12 decision occurrences without replacement **within each PPO model**, using reservoir sampling while scanning every trace in seeds 3000–7999. Sampling seed: 20261010. This gives 36 states per type, except M5 Cleave→Surge, which has no s0 occurrences and therefore has 24 states from s1/s2. In total, 204 M5 and 216 M6 states were scored. The disagreement frequencies and non-forced decision counts are unchanged: 29.1% of M5 and 44.1% of M6 decisions differ from the best heuristic.

Each sampled state was scored with 1,024 common-seed rollouts per action, continuing with the best heuristic. Pooled means weight each model by its share of that disagreement type's occurrences, rather than treating the equal sample quotas as equal population weights. Approximate, pointwise 95% normal intervals include stratified state-sampling and paired Monte Carlo uncertainty; they are not adjusted for multiple comparisons and do not estimate variation from retraining PPO. Values are **local advantages under heuristic continuation**, not effects of correcting PPO while PPO continues playing, and they cannot be summed into a whole-fight gap. Target-swap rows retain the original slot-specific classification and are labelled explicitly.

| Stage | Heuristic choice → PPO choice | Frequency/fight | Sampled states s0/s1/s2 | Weighted local advantage, points (approx. 95% CI) |
|---|---|---:|---|---|
| M5 | Trip focus target → Attack | 0.75 | 12/12/12 | +1.53 (+0.82 to +2.23) |
| M5 | Cleave → Attack | 0.67 | 12/12/12 | +0.12 (-1.49 to +1.74) |
| M5 | Surge → Trip | 0.56 | 12/12/12 | +0.16 (-0.14 to +0.46) |
| M5 | Trip → Cleave | 0.44 | 12/12/12 | +2.70 (+1.82 to +3.58) |
| M5 | Cleave → Surge | 0.43 | 0/12/12 | +0.00 (+0.00 to +0.00) |
| M5 | Trip slot 0 → Trip another slot | 0.36 | 12/12/12 | -0.51 (-1.54 to +0.52) |
| M6 | Cleave → Attack | 2.29 | 12/12/12 | -1.02 (-3.16 to +1.13) |
| M6 | Trip focus target → Attack | 0.82 | 12/12/12 | +0.94 (+0.37 to +1.51) |
| M6 | Advance → Attack | 0.82 | 12/12/12 | -0.73 (-3.00 to +1.53) |
| M6 | Trip slot 1 → Trip another slot | 0.52 | 12/12/12 | -0.28 (-1.84 to +1.27) |
| M6 | Attack slot 1 → Attack another slot | 0.52 | 12/12/12 | -0.51 (-1.69 to +0.67) |
| M6 | Advance → Trip | 0.45 | 12/12/12 | +0.15 (-0.44 to +0.74) |

The zero-variance Cleave→Surge interval records equality in the sampled rollouts; it does not prove that every unsampled state has zero difference.

- **M5: Trip timing is supported; a general Cleave-before-Attack benefit is not.** Trip→Attack and Trip→Cleave have positive local advantages of +1.53 and +2.70 points. The corrected Cleave→Attack mean is +0.12 points, with an interval spanning −1.49 to +1.74, versus +2.73 in the original early-occurrence sample. That old sample came solely from s0, which accounts for only 2.6% of Cleave→Attack disagreements; s1/s2 account for the other 97.4%. Their corrected local means are +3.40 / −0.09 / +0.32 points for s0/s1/s2. The zero Cleave→Surge estimate means no difference was observed in those 24 scored states. The other ordering and target-swap means are not clearly separated from zero.
- **Whole-fight check (M5).** Replacing PPO Attack/Cleave with the heuristic's Trip when the heuristic chooses Trip, and replacing PPO Attack with Cleave when the heuristic chooses Cleave, raises the three-model pooled win rate from 74.38% to 75.43% on the same 5,000 encounter seeds per model. It recovers about 1.05 of the 4.14-point gap, leaving 3.09 points behind tuned. Seed-specific combined gains are +0.12 / +2.16 / +0.86 points. Cleave-only repair does not produce a consistent gain. These interventions support a partial Trip-timing contribution and interactions, rather than the previous suggestion that multiplying local values by frequencies explains a 4–5-point gap.
- **M6: attribution remains incomplete.** Trip→Attack has a positive local advantage of +0.94 points (interval +0.37 to +1.51). All other scored types have intervals spanning zero, including Advance→Attack and Advance→Trip. Frequent missed or delayed advances should therefore not automatically be labelled costly mistakes. A separate whole-fight intervention forces PPO s0 to follow adaptive's opening tanky-Brute dive/soft-Brute hold plan and leaves other actions to PPO; it scores 76.18% versus 75.74% originally (+0.44 points, paired SE 0.68), still 4.42 points behind adaptive. This supports an execution gap for that model under that intervention, but the three models' full gaps remain unallocated.

Samples, model-specific population counts and weights, per-state paired outcomes, and uncertainty estimates are preserved in `runs/v2/analysis/stratified/{m5,m6}/`. The original script and early-occurrence estimates are archived in `runs/review/stratified_diagnostics/`; `comparison.json` there records old versus corrected means. Whole-fight intervention results are in `runs/review/m5/interventions*.json` and `runs/review/m6/macro_intervention.json`.

## Rollout planner pilot

This is extra to the plan, included as a quick check of whether one-step lookahead improves on the best heuristic. Setup: the existing `RolloutAgent`, 32 rollouts per legal action (its default), planner seed 0, best heuristic as the continuation policy. The sample sizes were fixed in advance and cut for wall time: 600 M5 fights (seeds 3000–3599) and 300 M6 fights (seeds 3000–3299).

| Stage | Planner (continuation) | Planner win rate (SE) | Same heuristic on the same seeds | Paired difference (SE) |
|---|---|---:|---:|---:|
| M5 | rollout(tuned) | 348/600 = 58.00% (2.01) | 473/600 = 78.83% | −20.83 (2.47) |
| M6 | rollout(adaptive) | 196/300 = 65.33% (2.75) | 244/300 = 81.33% | −16.00 (3.31) |

The planner is much *worse* than its own continuation policy, as in the earlier M4 pilot (47% vs 64%). It disagrees with the heuristic on about 44% of M5 decisions. The most common deviations are ending the turn instead of using Action Surge, Dodging instead of attacking, attacking before tripping, and single-target attacks instead of Cleave.

Re-scoring 12 sampled M5 states of each type with 1,024 rollouts per action favored the heuristic's choice in 8, 10, 11, and 7 of 12 states, with means of +1.07, +2.95, +1.99, and +0.69 points. So the loss is mostly finite-sample selection error at 32 rollouts across 8–11 legal actions: small per-decision mistakes compounding over about 23 decisions per fight. It is not evidence that the heuristic is near-optimal or that the stages are broken. A few states favored the planner's alternative by up to 11 points, which marks real places where the heuristic is locally wrong.

Reconstruction itself is exact: restored simulators match reseeded live environments in tests. The other agent's planner work would need larger budgets, confidence-aware selection, or a better continuation before it can show headroom here.

## Assessment: more decision headroom than M4?

**Yes, by every measure I have, with caveats.**

1. **Spread.** On M4, PPO (60.5% pooled) and the simple heuristics (61–64%) sit within about 3 points of each other. On M5 and M6, the naive M4-style heuristic is 17.1 and 39.7 points below the best hand heuristic (z ≈ 20 and 45). Both stages hit the plan's targets: naive at 61.4% and 40.9%, inside the 35–65% band, with spreads ≥8–10 points and the best heuristics off the ceiling at 78.5% and 80.6%.
2. **Beyond the new button.** Much of the spread is "use Trip" (trip_focus − naive = +15.2 on M5), which any learner finds. There are also significant second-order decisions among strong policies:
   - target priority: +2.8 for lowest-HP over Healer-first;
   - resource timing: +1.9 for tuned Second Wind and Cleave;
   - an encounter-dependent macro plan on M6: +1.2–1.7 for adaptive over either fixed plan;
   - traps that cost 15–20 points on dev seeds (an opening single diver; Disengage whenever Surge is available).
   These are resolved at n=5,000 but small in absolute terms.
3. **Learner gap.** On M2, PPO came within about 2 points of the exact optimum, and on M4 it matched the heuristic. Here, 1M-step PPO is 4.1 (M5) and 5.4 (M6) points below a hand heuristic. Its learning curve plateaus by about 500k steps, and checkpoint selection does not help. That leaves at least this much room for better learners and planners, plus whatever lies between the best heuristic and the unknown optimum.
4. **Caveats.**
   - No exact optimum: the state space is far larger than M2's, so no DP solve.
   - The heuristics' thresholds were chosen on dev seeds.
   - PPO used train.py defaults only: no hyperparameter, network-size, or longer-training search.
   - The stages remain dice-heavy: the best policies still lose about 20% of fights.
   - Every mechanic is a simulator choice; nothing here shows the same decisions matter in BG3.
   - The rollout pilot does not show headroom above the heuristic, because 32-sample one-step lookahead is too noisy to be a fair test.

## Reproduce

```bash
PY=/path/to/.venv/bin/python
$PY train.py --stage m5 --seed 0 --timesteps 1000000 --save-dir runs/m5_ppo_1m_s0
$PY -m evaluation.evaluate --stage m5 --agent heuristic --heuristic-variant tuned --seed-start 3000 --episodes 5000 --format json --output runs/v2/m5/tuned/summary.json --trace-dir runs/v2/m5/tuned
$PY -m evaluation.evaluate --stage m6 --agent ppo --model runs/m6_ppo_1m_s0/final_model.zip --seed-start 3000 --episodes 5000 --trace-dir runs/v2/m6/ppo_s0
$PY -m evaluation.paired runs/v2/m6/ppo_s0/run_card.json runs/v2/m6/adaptive/run_card.json
$PY -m evaluation.tactical_behavior --stage m6 runs/v2/m6/adaptive/episodes.jsonl
$PY -m evaluation.ppo_diagnostics m5 tuned 6 --samples-per-model 12 --budget 1024 --workers 2
$PY -m evaluation.ppo_diagnostics m6 adaptive 6 --samples-per-model 12 --budget 1024 --workers 2
```

Artifacts (ignored `runs/`):

- `runs/v1/` and `runs/v2/`: summaries, traces, and run cards per stage and policy.
- `runs/{m5,m6}_ppo_1m_s{0,1,2}/`: models, checkpoints, training metadata.
- `runs/v2/analysis/`: tables, paired comparisons, behavior JSON; `stratified/{m5,m6}/` holds corrected diagnostic manifests, per-state scores, and summaries.
- `runs/review/stratified_diagnostics/`: archived original diagnosis, sampling audit, and old-versus-corrected comparison. The old `ppo_gap_m5.txt`/`ppo_gap_m6.txt` outputs are historical; use the `stratified/` summaries for corrected values.
- `runs/v2/curve/`: seed-0 learning curve.
- `runs/v2/rollout/`: planner pilot; `runs/v2/analysis/{pilot,pilot_divergence,pilot_recheck,ppo_gap,m6_dive_plans}.py` and their `.txt` outputs.
- `runs/explore/`: dev-seed tuning scripts.
