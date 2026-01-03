# Step 7 Part 2: MLP Neuron Analysis

**Script**: `scripts/step7_fine_grained_analysis.py`  
**Results**: `results/step7_fine_grained.json`  
**Method**: `analysis/mlp_analysis.py` → `MLPAnalyzer`  
**Figure**: `figures/step7_mlp_differences.png`

---

## Key Finding

**MLP neuron L34#0 has the largest difference (3.53) between ToM conditions.** This specific neuron may encode belief state information.

---

## Results by Layer

| Layer | Gate Total Diff | Down Total Diff | Top Gate Neuron | Top Down Neuron |
|-------|-----------------|-----------------|-----------------|-----------------|
| L32 | 1579 | 709 | #849 (0.95) | #2355 (1.19) |
| L33 | 1574 | **1007** | #3233 (1.25) | **#293 (2.15)** |
| L34 | **1715** | 958 | #168 (1.22) | **#0 (3.53)** |

---

## Key Neurons

### 1. L34 Neuron #0 (diff = 3.53)
- **Largest single-neuron difference** across all layers
- Located in the "down_proj" (output) of the MLP
- This neuron activates **very differently** between:
  - Condition A: "Alice knows..." (belief updated)
  - Condition B: "Alice returns..." (no update)

### 2. L33 Neuron #293 (diff = 2.15)
- Second largest difference
- Also in down_proj
- May work together with L34#0

### 3. L32 Neuron #2355 (diff = 1.19)
- Earlier in the pipeline
- Lower difference but still notable

---

## Qwen3 MLP Architecture

```
Input → gate_proj → activation → × up_proj → down_proj → Output
        (gating)                            (output to residual)
```

- **gate_proj**: Controls what information passes
- **up_proj**: Expands to intermediate dimension  
- **down_proj**: Projects back, adds to residual stream

Our findings show **down_proj neurons** have larger differences - these directly influence the residual stream.

---

## Methodology

### Condition A (ToM success expected)
```
"Alice told Bob...Alice knows it's in basket. Alice looks in the"
"Sally informed...Sally knows it's in forest. Sally goes to the"
"Mom was told...Mom knows they're on hook. Mom reaches for the"
```

### Condition B (ToM failure expected)
```
"Alice put ball...Bob moved it. Alice returns. Alice looks in the"
"Sally hid gem...Tom moved it. Sally returns. Sally goes to the"
"Mom left keys...Dad moved them. Mom comes back. Mom reaches for the"
```

---

## Interpretation

### Why L34#0 matters
- Final layer before output
- Has the largest diff (3.53) of any single neuron
- Likely encodes "does the agent know about the move?"

### The "Knowledge Update" Circuit
```
L32: Initial processing (attention heads + MLP neurons)
     ↓
L33: Key attention heads (H4, H16, H28) + MLP neuron #293
     ↓
L34: Final decision
     - Head H0 (enabler from ablation)
     - MLP neuron #0 (largest diff from neuron analysis)
     ↓
Output: Predict location based on belief state
```

---

## Next Steps

1. **Ablate L34 neuron #0 specifically** - Does it cause behavior change?
2. **Probe this neuron** - What does it encode?
3. **Test on multi-agent** - Does it generalize?

