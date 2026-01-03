# MATS Mission Speedrun Findings

**Date**: December 24, 2025  
**Duration**: ~1 hour speedrun

---

## Executive Summary

Three critical experiments completed in rapid succession:

| Test | Result | Significance |
|------|--------|--------------|
| **Circuit Re-validation** | NOT VALIDATED | Earlier claims were artifacts |
| **Explicit vs Implicit ToM** | +52% CONFIRMED | Major capability gap |
| **Self/Other/User Separation** | 100% VALIDATED | Strong representation separation |

---

## 1. Circuit Re-validation: NOT VALIDATED

### Original Claim
- Ablating L17H4, L15H12, L24H29 achieves 90% ToM accuracy
- These heads form an "inhibitory network"

### Rigorous Test Results

| Condition | Implicit ToM | Explicit ToM |
|-----------|--------------|--------------|
| Baseline | **76.7%** | 100.0% |
| L17H4 ablation | 76.7% | 100.0% |
| 3-Head ablation | **63.3%** | 100.0% |
| Random control | 76.7% | 100.0% |

### Verdict: **CIRCUIT FINDING INVALIDATED**

- 3-head ablation actually *decreased* performance by -13%
- Random control had no effect
- Original findings were likely due to uncontrolled confounds

---

## 2. Explicit vs Implicit ToM: MASSIVE GAP CONFIRMED

### Test Design
Matched pairs testing:
- **Implicit**: "Alice put ball in drawer. Alice left. Bob moved ball to basket. Alice returned. Alice looks in the..."
- **Explicit**: "Alice believes the ball is in the drawer. It's actually in the basket. Alice looks in the..."

### Results

| Condition | Accuracy | Mean Logit Diff |
|-----------|----------|-----------------|
| Implicit | **28.0%** | -3.29 |
| Explicit | **80.0%** | +1.06 |
| Semi-explicit | 16.0% | -1.96 |
| Structured | **84.0%** | +4.28 |

### Verdict: **+52% ADVANTAGE FOR EXPLICIT**

This is the **core finding** for MATS:
- Model has WEAK implicit ToM (28%)
- Model has STRONG explicit belief parsing (80-84%)
- The capability exists but requires explicit framing!

---

## 3. Self/Other/User Separation: 100% VALIDATED

### Test Design
- 50 multi-party dialogues (User + Self/Assistant + Other/Agent B)
- Linear probes to classify entity type from activations

### Results

| Layer | Probe Accuracy | Samples |
|-------|----------------|---------|
| 4 | 100.0% | 125 |
| 8 | 100.0% | 125 |
| 12 | 100.0% | 125 |
| 16 | 100.0% | 125 |
| 20 | 100.0% | 125 |
| 24 | 100.0% | 125 |
| 28 | 100.0% | 125 |
| 32 | 100.0% | 125 |
| 35 | 100.0% | 125 |

### Similarity Analysis
Interesting pattern: User-Self similarity DROPS in late layers (0.23 at L28)

### Verdict: **STRONG SEPARATION CONFIRMED**

The model forms **perfectly distinct** representations for User, Self, and Other agents across ALL layers.

---

## Key Implications for MATS Mission

### 1. Circuit Hunting Was Premature
- We were looking for circuits before validating basic capabilities
- The "inhibitory network" doesn't hold up to rigorous testing

### 2. Explicit vs Implicit Is THE Core Problem
- 28% implicit vs 80% explicit = **massive capability gap**
- This explains why the model works in multi-agent software dev (explicit communication) but fails Sally-Anne (implicit inference)

### 3. Representation Separation Is Not The Bottleneck
- 100% probe accuracy = representations ARE distinct
- The issue is not "entity confusion" but "belief computation"

---

## Revised Understanding

```
┌─────────────────────────────────────────────────────────────┐
│                     WHAT WORKS                               │
│                                                              │
│  [OK] Distinct User/Self/Other representations (100%)        │
│  [OK] Explicit belief parsing (80-84%)                       │
│  [OK] Role tracking across dialogue                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     WHAT FAILS                               │
│                                                              │
│  [X] Implicit belief computation from narrative (28%)        │
│  [X] Inferring beliefs from presence/absence (28%)           │
│  [X] Simulating other agents' knowledge states               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## What This Means for Multi-Agent AI

### Good News
- Models CAN track multiple agents distinctly
- Models CAN understand explicitly stated beliefs
- Multi-agent collaboration WORKS when communication is explicit

### Concerning News
- Models CANNOT reliably compute beliefs from implicit information
- In scenarios requiring implicit reasoning, accuracy drops to ~28%
- This could cause coordination failures in complex scenarios

### Practical Recommendations
1. **Design explicit protocols**: Make belief states explicit in multi-agent communication
2. **Don't assume implicit ToM**: Models won't "figure out" what others know
3. **Use structured formats**: "[AGENT_X BELIEVES]: ..." works much better

---

## Scripts

| Script | Purpose | Result |
|--------|---------|--------|
| step45_circuit_revalidation.py | Test inhibitory circuit claims | NOT VALIDATED |
| step46_explicit_implicit_tom.py | Compare explicit vs implicit | +52% CONFIRMED |
| step47_self_other_user_probing.py | Probe entity representations | 100% SEPARATION |
| step48_speedrun_summary.py | Generate summary figure | Complete |

---

## Conclusion

The speedrun revealed:

1. **Our earlier circuit claims don't hold up** - methodology matters!
2. **Explicit vs Implicit is the real gap** - +52% advantage for explicit
3. **Entity separation is NOT the bottleneck** - representations are distinct

This fundamentally changes the research direction: we should focus on understanding WHY implicit belief computation fails, not on finding "inhibitory circuits."

---

*Speedrun complete. Science corrects itself.*


