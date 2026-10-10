# BaldurRL

BaldurRL is a project for training an agent to play Baldur's Gate 3. In progress

The [passive BG3 H0 collector](bg3/h0/README.md) includes Script Extender Lua
source, installation/run instructions, offline compatibility tests, and a Windows
live-verification checklist. It has not been tested in the game.

## Simulator stages

M0 is the fixed 1v1 regression baseline. M1A varies one opponent's statistics. M1B has two opponents and target choice. M2 adds a one-use simplified Cleave. See [M0 specification](docs/M0_SPEC.md) and [stage specifications](docs/STAGES_SPEC.md).

Train or evaluate by stage:

```bash
.venv/bin/python train.py --stage m1b --seed 0 --timesteps 200000 --save-dir runs/m1b_ppo_s0
.venv/bin/python -m evaluation.evaluate --stage m1b --agent heuristic --seed-start 3000 --episodes 1000 --trace-dir runs/m1b_heldout/heuristic
```

The M1A, M1B, and M2 baseline training reward is terminal win/loss. M2 also supports an isolated `--reward damage` training experiment after the terminal baseline is established. Evaluation reports combat outcomes on explicit environment seeds.

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
