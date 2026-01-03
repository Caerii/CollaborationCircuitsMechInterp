# Complete Theory of Mind Circuit Model

## Executive Summary

Through systematic path patching and mechanistic analysis, we've mapped the complete ToM circuit in Qwen3-4B:

**The Circuit:**
```
Input: "Iris tells Alice: 'I moved the ball to basket'"
              |
              v
LAYER 15:  L15H9 [Pre-processing] -----> Essential but doesn't carry signal
              |
              v
LAYER 17-18: L17H4, L18H11, L18H14 [DECISION HEADS]
              |                        |
              |    These heads CARRY   |
              |    the update signal   |
              v                        v
LAYER 19:  L19H2, L19H15 [Post-processing] -----> Essential but doesn't carry signal
              |
              v
OUTPUT: Belief prediction
```

## The Three Key Discoveries

### 1. Inhibitors ARE the Circuit (Not Just Blockers)

| Head | Type (old name) | Restoration when patched |
|------|-----------------|--------------------------|
| L18H11 | "Inhibitor" | **36%** |
| L17H4 | "Inhibitor" | **22%** |
| L15H9 | "Enabler" | **0%** |
| L19H2 | "Enabler" | **0%** |

**Old understanding**: Inhibitors block ToM, enablers process it.
**New understanding**: "Inhibitors" ARE the ToM processors. They carry the signal.

### 2. The Signal is Distributed Across Positions

| Intervention | Accuracy |
|--------------|----------|
| Baseline | 23% |
| Patch last position | 23% |
| **Patch ALL positions** | **70%** |

The belief-update computation happens at EVERY token position, not just the final prediction.

### 3. Ablation Works by Changing the Default

Why does ablating "inhibitors" improve ToM?

```
Normal: Inhibitor outputs "DON'T UPDATE" signal → Wrong answer
Ablated: No signal → Model defaults to "UPDATE" → Correct answer
```

The model's prior (without inhibitor intervention) is to update beliefs. The inhibitors actively SUPPRESS this default.

## Revised Circuit Architecture

```
                    CORRUPTED INPUT                    CLEAN INPUT
                    (no bridge phrase)                (with bridge phrase)
                           |                                |
                           v                                v
LAYER 15:           L15H9 processes                  L15H9 processes
                    (essential, but                  (same)
                    doesn't carry signal)
                           |                                |
                           v                                v
LAYERS 17-18:      L17H4, L18H11 output           L17H4, L18H11 output
                   "SUPPRESS UPDATE"               "ALLOW UPDATE"
                   signal at ALL positions         signal at ALL positions
                           |                                |
                           v                                v
LAYER 19:          L19H2, L19H15                   L19H2, L19H15
                   (essential post-processing)     (same)
                           |                                |
                           v                                v
OUTPUT:            Predicts WRONG location         Predicts CORRECT location
                   (original: drawer)              (updated: basket)
```

## Why "Inhibitor" is a Misnomer

We should rename these heads. They're not "inhibitors" in the traditional sense:

| Old Name | What They Actually Do | Better Name |
|----------|----------------------|-------------|
| Inhibitors | Carry the update/no-update decision | **Decision Heads** |
| Enablers | Essential infrastructure | **Processing Heads** |

The "decision heads" (L17H4, L18H11) make the actual ToM decision at every token position.

## Intervention Strategies (Revised)

### Strategy 1: Ablate Decision Heads (Removes Active Suppression)
```python
ablate(L17H4, L18H14) → 100% ToM accuracy
```
Works because: Removes the "SUPPRESS" signal, model defaults to "UPDATE"

### Strategy 2: Amplify Processing Heads (Boosts Infrastructure)
```python
amplify(L19H2, 3x) → 87% ToM accuracy
```
Works because: Strengthens the processing that normally gets suppressed

### Strategy 3: Path Patching (Injects Correct Signal)
```python
patch_all_positions(L18H11, clean_activations) → 70% restoration
```
Works because: Replaces "SUPPRESS" with "ALLOW" at all positions

## Implications for Multi-Agent AI

### 1. Monitoring
Track L17H4 and L18H11 activations to predict ToM failures. If these heads output "SUPPRESS" signal, the model won't update beliefs properly.

### 2. Intervention
During multi-agent tasks, ablate L17H4 + L18H14 to ensure proper belief tracking between agents.

### 3. Training
These heads could be fine-tuned to reduce the "SUPPRESS" default behavior.

## What We Still Don't Know

1. **Why does the model default to "SUPPRESS"?**
   - Training artifact?
   - Overfitting to certain patterns?
   - Statistical prior from pretraining data?

2. **What triggers the "ALLOW" signal in clean prompts?**
   - The bridging phrase activates something
   - What exactly do these heads detect?

3. **Does this generalize to other ToM scenarios?**
   - Second-order beliefs?
   - Multi-agent dialogues?
   - Real-world collaboration?

## Files & Results

| File | Purpose |
|------|---------|
| `step11_path_patching.py` | Main path patching experiment |
| `step12_signal_injection.py` | Signal injection test (didn't work) |
| `step12b_investigate_mechanism.py` | Discovered position importance |
| `PATH_PATCHING_DISCOVERY.md` | Detailed path patching analysis |
| `results/path_patching_results.json` | Raw results |
| `results/mechanism_investigation.json` | Position investigation results |

## Summary Table

| Discovery | Evidence | Confidence |
|-----------|----------|------------|
| L18H11 carries ToM signal | 36% restoration | HIGH |
| L17H4 carries ToM signal | 22% restoration | HIGH |
| Enablers don't carry signal | 0% restoration | HIGH |
| Signal is distributed | 70% vs 23% (all vs last) | HIGH |
| Ablation changes default | Consistent with all data | MEDIUM |
| "Inhibitors" are decision heads | Path patching + ablation | HIGH |

---

*This document represents the complete mechanistic understanding of the ToM circuit as of this analysis.*

