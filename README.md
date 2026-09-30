# BaldurRL

BaldurRL is a project for training an agent to play Baldur's Gate 3. In progress

## Inspecting decisions

Evaluate a saved policy on a fresh seed range and write a run card plus one JSONL trace per episode:

```bash
.venv/bin/python -m evaluation.evaluate --agent heuristic --seed-start 3000 --episodes 1000 --trace-dir runs/heldout/heuristic
.venv/bin/python -m evaluation.evaluate --agent ppo --model runs/m0_ppo_s0/final_model.zip --seed-start 3000 --episodes 1000 --trace-dir runs/heldout/ppo
```

`run_card.json` lists loss seeds and summary metrics. `episodes.jsonl` records observations, masks, actions, rolls, and resulting states for each fight. Use the same seed range for policies being compared, and keep that range separate from seeds used to select a PPO checkpoint.

Compare two run cards to list seeds where either policy lost and the first decision where their traces diverged:

```bash
.venv/bin/python -m evaluation.compare runs/heldout/heuristic/run_card.json runs/heldout/ppo/run_card.json --output runs/heldout/comparison.json
```

After that first divergence, the policies may consume random rolls in a different order, even when they share an episode seed.
