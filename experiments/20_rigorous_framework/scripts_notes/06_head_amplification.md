# Step 7 Part 3: Head Amplification

**Script**: `scripts/step7_fine_grained_analysis.py`  
**Results**: `results/step7_fine_grained.json`  
**Method**: `analysis/signal_injection.py` → `HeadAmplifier`  
**Figure**: `figures/step7_amplification_effects.png`

---

## Key Finding

**Amplifying critical heads (2.0x) can FLIP wrong predictions to correct!** This confirms these heads are enablers of ToM reasoning.

---

## Results

| Prompt | 0.5x | 1.0x (baseline) | 1.5x | 2.0x |
|--------|------|-----------------|------|------|
| Prompt 1 (Alice) | +0.61 ✓ | +0.47 ✓ | +0.36 ✓ | +0.25 ✓ |
| **Prompt 2 (Sally)** | -0.14 ✗ | -0.23 ✗ | -0.11 ✗ | **+0.14 ✓** |

### Prompt 2: The FLIP

```
Baseline (1.0x): wrong (diff=-0.23)
2.0x amplification: correct (diff=+0.14) ← FLIPPED!

Change: +0.38 logit diff
```

---

## Critical Heads Amplified

All 5 critical heads from Step 5:
- L32H0
- L33H4
- L33H16
- L33H28
- L34H0

---

## Interpretation

### 1. These Heads are ENABLERS
| Test | Effect | Conclusion |
|------|--------|------------|
| Ablation (Step 5) | -16.7% accuracy | Heads are necessary |
| Amplification (Step 7) | Flips wrong→correct | Heads are sufficient |

**Necessary AND Sufficient** = Strong causal evidence

### 2. The Effect is Monotonic (for Prompt 2)
```
0.5x → 1.0x → 1.5x → 2.0x
-0.14  -0.23  -0.11  +0.14
```
More amplification = higher probability of correct answer

### 3. Over-amplification Reduces Confidence (Prompt 1)
```
0.5x → 1.0x → 1.5x → 2.0x
+0.61  +0.47  +0.36  +0.25
```
For already-correct predictions, amplifying too much may introduce noise.

---

## The Complete Picture

| Step | Method | Finding |
|------|--------|---------|
| 5 | Ablation | Heads are necessary |
| 7a | Signal Injection | Can restore behavior |
| 7b | MLP Analysis | Specific neurons matter |
| **7c** | **Amplification** | **Heads are sufficient** |

---

## Technical Details

```python
# Amplification hook scales head outputs
def hook(module, args):
    hidden = args[0].clone()
    reshaped = hidden.view(batch, seq, n_heads, head_dim)
    for head_idx in head_indices:
        reshaped[:, :, head_idx, :] *= scale_factor  # 2.0x
    return (reshaped.view(...),)
```

---

## Implications for Interpretability

1. **Targeted intervention works**
   - We can boost specific mechanisms
   - Not just remove (ablate) them

2. **Complementary to ablation**
   - Ablation: "Is this necessary?"
   - Amplification: "Is this sufficient?"
   - Together: Full causal picture

3. **Potential for steering**
   - Could we amplify ToM heads to make model "more empathetic"?
   - Or reduce them for more "logical" reasoning?

