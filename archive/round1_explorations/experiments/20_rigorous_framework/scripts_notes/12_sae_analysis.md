# 12_sae_analysis.md

## Step 13: SAE Feature Analysis

**Goal:** Decompose MLP activations into interpretable sparse features using Sparse Autoencoders (SAEs).

## What Are SAEs?

SAEs learn to represent high-dimensional activations (d=2560) as a **sparse combination** of interpretable features (d=10,240).

```
Activation (2560 dims) → Encoder → Sparse Features (10,240 dims, ~13 active) → Decoder → Reconstruction
```

**Why useful?**
- Only ~13 features active per input (0.1%)
- Each feature = interpretable direction
- Can find "belief update" features!

## Results

### Sparsity & Reconstruction

| Metric | Value |
|--------|-------|
| Dictionary size | 10,240 (4× expansion) |
| Mean active features | 13 per input |
| Sparsity | 0.1% |
| Reconstruction MSE | 0.0003 |
| Cosine similarity | 99.77% |

### Top Discriminative Features

| Feature | FB Activation | TB Activation | Difference | Pattern |
|---------|---------------|---------------|------------|---------|
| **#1979** | 3.63 | 1.51 | +2.12 | **FB>TB** |
| **#4724** | 3.83 | 2.19 | +1.63 | FB>TB |
| **#4772** | 1.19 | 1.96 | -0.77 | TB>FB |
| **#7052** | 1.41 | 2.14 | -0.73 | TB>FB |
| **#3911** | 1.21 | 1.91 | -0.70 | TB>FB |

### Interpretation

**False Belief Features (FB>TB):**
- Feature #1979: Likely encodes "agent left / has outdated information"
- Feature #4724: Likely encodes "object was moved"

**True Belief Features (TB>FB):**
- Feature #4772: Likely encodes "agent stayed / watched"
- Feature #7052: Likely encodes "agent knows current state"

### Visualization

The heatmap shows clear separation between:
- Scenarios 0-7 (False Belief) - left of white line
- Scenarios 8-15 (True Belief) - right of white line

## Implications

1. **Belief is encoded sparsely**: Only ~13 features needed per ToM scenario
2. **Features are interpretable**: Can identify "outdated belief" vs "current knowledge" features
3. **Could enable steering**: Manipulating feature #1979 might flip FB→TB predictions

## Future Directions

1. **Train Transcoder**: Map MLP inputs→outputs to understand computation
2. **Feature ablation**: Ablate specific features (like #1979) to test causal role
3. **Feature steering**: Inject "belief update" feature to change predictions
4. **Cross-layer tracking**: Follow features through L12→L34

## Technical Notes

- Used simple SAE with L1 regularization (coefficient=0.001)
- 500 training epochs on 16 scenarios
- ReLU activation for sparsity
- Decoder weights normalized after each step for stability

