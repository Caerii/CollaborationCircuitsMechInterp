# Step 4: Logit Lens Analysis - Where Decisions Happen

**Script**: `scripts/step4_logit_lens.py`  
**Results**: `results/step4_logit_lens.json`  
**Figures**: `figures/step4_logit_lens_evolution.png`, `figures/step4_decision_layer_histogram.png`

---

## Key Finding

**Decisions crystallize in LATE layers (29-34), NOT middle layers (15-25) as hypothesized.**

---

## Results

### Decision Layer Statistics

| Condition | Mean Layer | Std Dev | Correct |
|-----------|------------|---------|---------|
| False Belief | **29.4** | ±4.6 | 35% (7/20) |
| True Belief | **33.9** | ±1.8 | 60% (6/10) |

### Hypothesis Test
- **H2**: Decision in middle layers (15-25)
- **Result**: NOT SUPPORTED
- **Actual**: Decisions happen at layers 28-36

---

## Layer-by-Layer Dynamics

Observing the logit evolution plots:

```
Layer 0-10:   Initial lexical representations
              Logit diff often positive (first-mention bias?)

Layer 10-25:  "Competition" phase
              Predictions oscillate between correct/wrong
              Model seems uncertain

Layer 25-35:  Decision crystallizes
              Final prediction locks in
              This is where ToM "happens"
```

---

## Key Observations

1. **True Belief decisions are LATER than False Belief**
   - TB: 33.9 vs FB: 29.4
   - Suggests TB might be simpler (just track final location)

2. **Prediction can flip multiple times before settling**
   - Some scenarios show green→red→green pattern
   - Model "considers" both answers

3. **Maximum diff often at very late layers (32-35)**
   - The model becomes most confident at the end

---

## Implications for Circuit Analysis

Focus ablation and patching on **layers 25-35**, not 15-25.

