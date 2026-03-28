# Step 7 Part 1: Signal Injection (Causal Evidence)

**Script**: `scripts/step7_fine_grained_analysis.py`  
**Results**: `results/step7_fine_grained.json`  
**Method**: `analysis/signal_injection.py` → `SignalExtractor`

---

## Key Finding

**Signal injection CAN restore correct ToM behavior** - providing causal evidence that the extracted "belief update signal" contains the actual mechanism.

---

## Method: Signal Extraction & Injection

1. Run **CLEAN** prompt (model gets it right) → cache activations
2. Run **CORRUPTED** prompt (model gets it wrong) → cache activations  
3. Compute **SIGNAL** = clean - corrupted
4. **INJECT** signal into corrupted prompt
5. Measure if behavior is restored

---

## Results

| Prompt Type | Signal Magnitude (L34) | Original | After Injection | **RESTORED?** |
|-------------|------------------------|----------|-----------------|---------------|
| told_explicit | 20.55 | ❌ Wrong | ❌ Wrong | No |
| **saw_explicit** | **29.31** | ❌ Wrong | ✅ Correct | **YES** |
| informed | 32.94 | ✅ Correct | ✅ Correct | (N/A) |

---

## Signal Magnitudes by Layer

The signal gets **stronger** in later layers:

```
told_explicit:
  L32: 15.13
  L33: 19.77
  L34: 20.55  ← strongest

saw_explicit:
  L32: 17.86
  L33: 22.41
  L34: 29.31  ← strongest

informed:
  L32: 29.61
  L33: 39.50  ← peak
  L34: 32.94
```

---

## What This Means

### Successful Restoration (saw_explicit)
- The "clean" prompt explicitly says "Sally watched the whole time. Sally knows..."
- The "corrupted" prompt lacks this knowledge update
- **Injecting the difference** teaches the model that Sally knows
- This is **causal evidence** that the signal encodes belief state

### Failed Restoration (told_explicit)
- Signal was smaller (20.55 vs 29.31)
- May need stronger signal or different injection position
- Or the mechanism differs between "told" vs "watched"

---

## Technical Details

```python
# The signal extraction process
signal = SignalExtractor.extract_signal(
    clean_prompt="...Sally watched the whole time. Sally knows...",
    corrupted_prompt="...Sally comes back. Sally searches...",
    layers=[32, 33, 34]
)

# signal[layer] = clean_activation - corrupted_activation
# This difference IS the "belief update" signal
```

---

## Implications

1. **The "belief update" is a specific activation pattern**
   - Can be extracted and transferred
   - Not just prompt engineering

2. **Later layers have stronger signals**
   - Consistent with Logit Lens findings
   - L33-34 are key

3. **Causal manipulation possible**
   - We can inject signals to change behavior
   - Foundation for interpretability interventions

