# M2 bounded damage-shaping experiment

This experiment began after all three terminal-only M2 PPO runs and their held-out results were complete. The two training schemes used the same MaskablePPO hyperparameters, training seeds 0–2, and 200,000 requested timesteps per seed (200,704 actual). Terminal training rewarded only +1 win / -1 loss. Damage-shaped training kept those terminal rewards and added, at each Fighter decision, `0.2 × (total enemy HP before − after) / initial total enemy max HP`. With no enemy healing, the added positive reward over an episode is bounded by 0.2. Enemy damage adds nothing. Evaluation used the same unshaped combat outcomes, masks, and environment seeds for both schemes; raw training rewards were not compared.

## Sample efficiency on model-selection seeds 1000–1199

Each saved checkpoint was evaluated externally on the same 200 episodes. Entries are win rates for training seeds 0 / 1 / 2, followed by the three-seed mean. With only 200 evaluation episodes per checkpoint, small differences are noisy.

| Timesteps | Terminal PPO seed rates | Terminal mean | Damage-shaped PPO seed rates | Shaped mean |
| ---: | --- | ---: | --- | ---: |
| 25k | 62.0% / 67.0% / 58.0% | 62.33% | 58.0% / 67.0% / 58.0% | 61.00% |
| 50k | 63.0% / 66.0% / 59.5% | 62.83% | 61.0% / 64.5% / 56.5% | 60.67% |
| 100k | 63.5% / 68.0% / 64.5% | 65.33% | 65.0% / 66.0% / 58.0% | 63.00% |
| 200k | 68.5% / 64.5% / 66.0% | 66.33% | 62.5% / 68.0% / 63.5% | 64.67% |

The shaped mean did not exceed the terminal mean at any saved budget. Training wall times for terminal seeds 0–2 were **36.69, 38.48, 37.14 seconds**; shaped times were **36.00, 36.03, 35.95 seconds**.

## Final external comparison

All final models were evaluated on exactly seeds 3000–3999, with deterministic masked predictions.

| Training reward | Seed | Wins | Losses | Trunc. | Win rate | Binomial SE | Mean rounds | Mean final Fighter HP | Cleave fights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Terminal | 0 | 666 | 334 | 0 | 66.6% | 1.49 pp | 6.107 | 8.329 | 19.9% |
| Terminal | 1 | 648 | 352 | 0 | 64.8% | 1.51 pp | 6.175 | 7.897 | 66.9% |
| Terminal | 2 | 663 | 337 | 0 | 66.3% | 1.49 pp | 6.159 | 8.147 | 24.0% |
| Damage-shaped | 0 | 657 | 343 | 0 | 65.7% | 1.50 pp | 6.096 | 8.062 | 34.5% |
| Damage-shaped | 1 | 670 | 330 | 0 | 67.0% | 1.49 pp | 6.124 | 8.239 | 31.5% |
| Damage-shaped | 2 | 654 | 346 | 0 | 65.4% | 1.50 pp | 6.019 | 7.888 | 15.4% |

Terminal PPO mean was **65.90%** with sample standard deviation **0.96 percentage points** across training seeds. Shaped PPO mean was **66.03%** with sample standard deviation **0.85 points**. A fresh non-overlapping 5,000-seed range (4000–8999) gave terminal seeds **67.76%, 67.18%, 68.28%** and shaped seeds **67.62%, 68.66%, 66.30%**. Shaping did not reliably improve final win rate either.

Trace comparison shows policy changes without an outcome gain: on the 1,000 original held-out seeds, terminal PPO seed 0 and shaped PPO seed 1 first differed in target choice 88 times among fights where either lost, and in a targeted attack versus Cleave 53 times. Their outcomes split 40 terminal-only wins and 44 shaped-only wins. As later action sequences consume the seeded dice stream differently, the first divergence is descriptive rather than a causal estimate.

**Conclusion:** this initial bounded damage term did not improve M2 sample efficiency at 25k, 50k, 100k, or 200k, and its final external performance was indistinguishable from terminal-only training at this scale. Checkpoint evaluations and detailed run artifacts are under ignored `runs/`.
