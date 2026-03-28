# 10_mlp_probing.md

## Step 11: MLP Probing for Belief States

**Goal:** Test if MLPs (not just attention) encode belief state information.

**Hypothesis:** While attention heads track AGENTS, MLPs encode BELIEF STATES.

## Result: MLPs Encode Belief State with 95% Accuracy!

### Probe Accuracy by Layer

```
Layer 4:  ██████████████████████████ 50% (chance)
Layer 8:  █████████████████████████████████████████████████████████████████████████████████ 82%
Layer 12: ███████████████████████████████████████████████████████████████████████████████████████████████ 95% ★
Layer 16: ███████████████████████████████████████████████████████████████████████████████████████████████ 95% ★
Layer 20: ███████████████████████████████████████████████████████████████████████████████████████████████ 95% ★
Layer 28: ███████████████████████████████████████████████████████████████████████████████████████████████ 95% ★
Layer 32: ███████████████████████████████████████████████████████████████████████████████████████████████ 95% ★
Layer 33: █████████████████████████████████████████████████████████████████████████████████████████ 88%
Layer 34: █████████████████████████████████████████████████████████████████████████████████████████ 88%
```

### Key Findings

1. **Belief encoding starts at Layer 12**
   - Before Layer 12: chance performance (50%)
   - Layer 12+: 95% probe accuracy

2. **Encoding persists to final layers**
   - The belief state representation is maintained through the network
   - Slight degradation in final output layers (88% at L33-34)

3. **Most discriminative neuron: Layer 12, Neuron #0**
   - Activation difference: 0.473 between true/false belief
   - "T>F" pattern: higher activation for true belief scenarios

### Interpretation: Attention vs MLP Division of Labor

```
ATTENTION HEADS:              MLPs:
├── Track WHO (agents)        ├── Encode WHAT (belief state)
├── "This is about Alice"     ├── "Alice has outdated info"
└── 70.6% attention to names  └── 95% probe accuracy for belief

              ↓ COMPOSE ↓

         ToM PREDICTION
   "Alice thinks ball is in drawer"
```

This supports a clean division of labor:
- **Attention** = Agent tracking / binding
- **MLPs** = Belief state computation

### Neuron-Level Analysis (Layer 12)

| Neuron | Diff | Pattern |
|--------|------|---------|
| 0 | 0.473 | T>F (higher for true belief) |
| 4 | 0.227 | T>F |
| 96 | 0.148 | F>T (higher for false belief) |
| 99 | 0.128 | T>F |
| 70 | 0.118 | F>T |

The mix of T>F and F>T neurons suggests the MLP encodes a rich representation that distinguishes both directions.

