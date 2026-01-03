# Experiment 05: Naturalistic Transfer Learning

## Research Question

**Is entity encoding semantic or lexical?**

Do models truly understand WHO is speaking, or are they just pattern-matching explicit labels like "User:", "You:", "Helper:"?

---

## Results: CRITICAL FINDING

### Transfer Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Within-domain accuracy | **99.9%** | Probes work perfectly on labeled data |
| Transfer accuracy | **32.0%** | **CHANCE LEVEL** (33% for 3 classes) |
| Transfer gap | **+68%** | Almost all signal is lexical |

### Per-Layer Results

| Layer | Within-Domain | Transfer | Gap |
|-------|---------------|----------|-----|
| 0 | 99.6% | 32.3% | +67.2% |
| 4 | 100% | 33.7% | +66.3% |
| 8 | 100% | 29.7% | +70.3% |
| 12 | 100% | 27.7% | +72.3% |
| 16 | 100% | 33.0% | +67.0% |
| 20 | 100% | 30.0% | +70.0% |
| 24 | 100% | 36.0% | +64.0% |
| 28 | 99.9% | 33.3% | +66.6% |
| 32 | 99.9% | 20.1% | +79.8% |
| 35 | 99.9% | 43.9% | +56.0% |

---

## Interpretation

### The Encoding is LEXICAL, Not Semantic

```
┌─────────────────────────────────────────────────────────────┐
│                     KEY FINDING                              │
│                                                             │
│  When explicit labels are removed, probes fail completely.  │
│  Transfer accuracy = 32% ≈ random chance (33%)              │
│                                                             │
│  The model is NOT tracking "who is speaking" semantically.  │
│  It's simply encoding the presence of label TOKENS.         │
└─────────────────────────────────────────────────────────────┘
```

### What This Means for Previous Experiments

| Experiment | Original Interpretation | Revised Interpretation |
|------------|------------------------|------------------------|
| 01: Probing | "Model encodes entity identity" | Model encodes **label tokens** |
| 02: Geometry | "User/Self/Other form distinct clusters" | **Token** representations form clusters |
| 03: Steering | "Can manipulate entity perception" | Can manipulate **token** representations |

### The U-Shaped Curve Revisited

The U-shaped Self↔Other similarity curve is still **real**, but it describes:
- How the model processes the **token "You:"** vs **token "Helper:"**
- NOT how it tracks abstract entity concepts through conversation

---

## Scientific Implications

### For AI Safety 🔒

**Good News:**
- Models don't have deep "theory of mind"
- Multi-agent behavior is more predictable
- Coordination relies on explicit signaling, not hidden understanding

**Concerns:**
- Easy to fool with label manipulation
- Multi-agent systems may be brittle
- No robust context-based entity tracking

### For Multi-Agent Systems 🤖

- Explicit labeling is CRITICAL
- Without labels, models lose entity tracking
- Can't rely on "natural conversation flow"

### For Mechanistic Interpretability 🔬

- Be careful distinguishing **token encoding** from **concept encoding**
- Transfer tests are essential controls
- Probe accuracy ≠ semantic understanding

---

## Methodology

### Training Data (Labeled)
```
User: I need help with Python.
You: I'd be happy to help!
Helper: I've seen this before.
```

### Test Data (Naturalistic)
```
[Turn 1] I need help with Python.
[Turn 2] I'd be happy to help!
[Turn 3] I've seen this before.
```

### Procedure
1. Train probes on labeled data (1,186 samples)
2. Extract activations from naturalistic dialogues (303 samples)
3. Test probe transfer
4. Compare within-domain vs transfer accuracy

---

## Visualization

![Transfer Learning Results](./figures/transfer_learning.png)

Left panel: Within-domain (blue) vs Transfer (red) accuracy by layer
Right panel: Transfer gap (smaller = more semantic)

---

## Conclusion

> **Entity encoding in Qwen3-4B is LEXICAL, not SEMANTIC.**
> 
> The 100% probe accuracy from Experiment 01 was detecting the presence of label tokens, not true entity understanding. When labels are removed, the model has no ability to distinguish speakers beyond chance.

This is a **critical control result** that recontextualizes all previous findings.

---

## Files

- `transfer_results.json` - Full numerical results
- `summary.json` - Summary statistics
- `figures/transfer_learning.png` - Visualization
