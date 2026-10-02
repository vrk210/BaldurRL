# M2 PPO training-budget follow-up

This follow-up tests whether the initial 200,000-step M2 MaskablePPO runs stopped before learning useful decisions. Three new terminal-reward models were trained **from scratch** for 1,000,000 requested timesteps each, using training seeds 0, 1, and 2. Architecture, masks, hyperparameters, environment, and model-selection seeds 1000–1199 were unchanged. Gymnasium/SB3 completed 1,003,520 actual timesteps per run. Wall times were 183.50, 183.70, and 184.35 seconds. The previous 200k models and all run artifacts remain under ignored `runs/`.

## Held-out performance

All policies in each table use identical environment seeds. PPO predictions are deterministic and masked. The first table reuses the original final held-out range; the second uses a fresh range not inspected in the preceding M2 reports.

| Policy | Wins / 1000, seeds 3000–3999 | Win rate | Cleave fights | Mean living enemies at Cleave |
| --- | ---: | ---: | ---: | ---: |
| Heuristic | 650 | 65.0% | 100.0% | 2.000 |
| PPO 200k seed 0 | 666 | 66.6% | 19.9% | 1.050 |
| PPO 200k seed 1 | 648 | 64.8% | 66.9% | 1.200 |
| PPO 200k seed 2 | 663 | 66.3% | 24.0% | 1.000 |
| PPO 1M seed 0 | 693 | 69.3% | 11.0% | 1.000 |
| PPO 1M seed 1 | 687 | 68.7% | 56.5% | 1.216 |
| PPO 1M seed 2 | 687 | 68.7% | 5.2% | 1.000 |

The 1M models' mean was **68.90%**, sample standard deviation **0.35 percentage points**, versus the original 200k mean **65.90%**, SD **0.96 points**, on these 1,000 seeds.

| Policy | Wins / 5000, fresh seeds 9000–13999 | Win rate |
| --- | ---: | ---: |
| M2 heuristic | 3256 | 65.12% |
| Same heuristic, never uses Cleave | 3399 | 67.98% |
| M1B PPO seed 2 mapped into M2, never uses Cleave | 3503 | 70.06% |
| PPO 200k seed 0 | 3389 | 67.78% |
| PPO 1M seed 0 | 3434 | 68.68% |
| PPO 200k seed 1 | 3336 | 66.72% |
| PPO 1M seed 1 | 3474 | 69.48% |
| PPO 200k seed 2 | 3410 | 68.20% |
| PPO 1M seed 2 | 3491 | 69.82% |

Across the three training seeds, the fresh-range mean improved from **67.57%** at 200k (sample SD **0.76 points**) to **69.33%** at 1M (SD **0.59 points**). Paired by environment seed, the 1M-minus-200k gains were **+0.90, +2.76, and +1.62 percentage points**. Episode bootstrap 95% intervals were **−0.30 to +2.10**, **+1.52 to +4.04**, and **+0.46 to +2.80 points**, respectively. Seed 0's gain is inconclusive; seeds 1 and 2 show a gain on this range. The mapped M1B model's 70.06% demonstrates that M2 can do at least that well without Cleave, but its 0.24-point lead over the best native 1M model is inconclusive.

## Checkpoint behavior

Checkpoints from each 1M run were evaluated on the model-selection seeds 1000–1199 and, for diagnosis, seeds 3000–3999. Each cell lists win rates for training seeds 0 / 1 / 2. The held-out checkpoints were inspected **after** training; they were not used to select or continue models.

| Exact checkpoint step | Model-selection 200 episodes | Held-out 1000 episodes |
| ---: | --- | --- |
| 200k | 68.5% / 64.5% / 66.0% | 67.3% / 66.2% / 66.8% |
| 500k | 73.5% / 71.5% / 71.0% | 69.0% / 66.8% / 69.5% |
| 750k | 68.0% / 70.0% / 72.5% | 68.6% / 68.6% / 67.8% |
| 1M | 71.5% / 72.0% / 71.5% | 67.7% / 69.0% / 68.5% |

The final models are a few thousand timesteps beyond the exact 1M checkpoints. Learning is not monotonic at every checkpoint, but the 1M final models improved over the previous 200k final models on both the original and fresh held-out ranges. The saved model-selection-best checkpoints scored 69.48%, 69.46%, and 69.74% on the fresh 5,000 seeds, close to the 1M final models; checkpoint selection was not the main cause of the earlier gap.

## Matched-start action diagnostic

For each of 200 independently sampled M2 encounters (seeds 2000–2199), three legal opening actions were tried under each of 50 shared combat-roll seeds. After that first decision, the same lowest-HP heuristic continued with Cleave disabled. Enemy stats and rollout seed were therefore matched across opening actions. This estimates the value of one opening decision **under the specified continuation policy**, not the optimal action value of every M2 state.

| Opening action | Wins / 10,000 rollouts | Win rate |
| --- | ---: | ---: |
| Attack enemy 0 | 6629 | 66.29% |
| Attack enemy 1 | 6688 | 66.88% |
| Cleave | 6507 | 65.07% |

Resampling the 200 encounter seeds gives a 95% interval of **+0.59 to +3.01 percentage points** for Attack enemy 1 minus Cleave. This supports the trace finding that mandatory opening Cleave is weak under the current half-damage rule, while leaving open states where Cleave could help.

## Interpretation

The longer run addresses the central concern: **200k M2 PPO training did leave win rate on the table**. All three 1M models beat the simple M2 heuristic on the fresh 5,000 seeds by 3.56–4.70 points. This is an empirical improvement, not evidence that 1M is optimal.

Cleave behavior still looks poorly learned. On the original 1,000 seeds, the 1M models used Cleave against only one living enemy in 110/110, 443/565, and 52/52 Cleave uses, respectively. A normal targeted Attack has the same Action cost, at least as much immediate damage on the same roll, and does not spend Cleave; one-enemy Cleave is strategically dominated. The learned gain therefore seems to come from other timing or targeting changes and from avoiding the heuristic's automatic opening Cleave, rather than consistently exploiting Cleave's multi-target potential.

No combat rules, reward, observation, or action-mask code was changed for this follow-up. Detailed models, traces, per-seed outcomes, checkpoint evaluations, and matched-start results are under ignored `runs/`.
