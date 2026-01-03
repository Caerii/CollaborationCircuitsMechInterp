# Experiments: Self/Other/User Representation Separation

## Project Overview

This experiment suite investigates how language models internally represent different entity types (User, Self, Other) in multi-agent dialogue contexts. We use mechanistic interpretability techniques to understand the "biology" of multi-agent collaboration.

**Model**: Qwen3-4B-Instruct-2507  
**Hardware**: RTX 3080 (10GB VRAM)  
**Dataset**: 200 synthetic multi-party dialogues (1,186 turns)

---

## Experiment Index

| # | Experiment | Status | Key Finding |
|---|------------|--------|-------------|
| 01 | [Baseline Probing](./01_baseline_probing/) | ✅ Complete | 100% accuracy classifying User/Self/Other from activations |
| 02 | [Representation Geometry](./02_representation_geometry/) | ✅ Complete | U-shaped separation curve, peak at layer 20 |
| 03 | [Causal Steering](./03_causal_steering/) | ✅ Complete | Entity vectors extracted, ~60% flip rate achievable |
| 04 | [3D Visualizations](./04_3d_visualizations/) | ✅ Complete | 7 animated GIFs showing geometric relationships |
| 05 | [Naturalistic Transfer](./05_naturalistic_transfer/) | 🔄 Pending | Test if encoding persists without explicit labels |

---

## Key Scientific Finding

> **Self and Other representations start nearly identical (4° apart at layer 0), diverge maximally at layer 20 (21° apart), then partially converge again at output. The model actively LEARNS to distinguish between AI agents during middle-layer processing.**

This finding has implications for:
- Multi-agent coordination and communication
- AI safety (detecting deceptive agent behavior)
- Understanding how models build "theory of mind" for other agents

---

## Quick Start

```bash
# Run all experiments
python scripts/run_experiment.py --phase all --n-dialogues 200

# Run advanced analysis
python scripts/advanced_experiments.py

# Generate 3D visualizations
python scripts/create_3d_visualizations.py
```

---

## Directory Structure

```
experiments/
├── README.md                      # This file
├── 01_baseline_probing/           # Linear probe experiments
├── 02_representation_geometry/    # Similarity & angle analysis
├── 03_causal_steering/            # Causal intervention tests
├── 04_3d_visualizations/          # Animated GIF visualizations
└── 05_naturalistic_transfer/      # Transfer learning experiments
```

Each experiment folder contains:
- `README.md` - Experiment description, methodology, results
- `results/` - Output files specific to that experiment
- `figures/` - Visualizations for that experiment

