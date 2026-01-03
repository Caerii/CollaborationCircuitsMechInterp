# 13_sae_steering_crosslayer.md

## Steps 13-18: SAE, Steering, and MLP Analysis

### Step 13: SAE Feature Analysis

**Trained SAE on L12 MLP outputs to find sparse belief features.**

| Metric | Value |
|--------|-------|
| Sparsity | 0.1% (13/10,240 features active) |
| Reconstruction | 99.77% cosine similarity |
| Top discriminative feature | #1979 (diff=2.12, FB>TB) |

**Key features:**
- Feature #1979: "Agent has outdated belief" (FB=3.6, TB=1.5)
- Feature #4724: "Object was moved" (FB=3.8, TB=2.2)

---

### Step 14: Feature Steering (Didn't Flip!)

**Tried to flip predictions by modifying SAE features at L12.**

| Method | Result |
|--------|--------|
| Suppress FB features | No flip |
| Boost TB features | No flip |
| Both | No flip |

**Why?** L12 is too early - decision happens at L28-32!

---

### Step 15: Cross-Layer Feature Evolution

**Trained SAEs across layers to track discriminability.**

| Layer | Max Discriminability | # Features |
|-------|---------------------|------------|
| L4 | 0.32 | 0 |
| L8 | 1.61 | 3 |
| L12 | 1.54 | 7 |
| L16 | 2.81 | **12** |
| L20 | 2.78 | 8 |
| L24 | 5.72 | 3 |
| **L28** | **10.94** | 4 |
| L32 | 6.24 | 2 |

**Key finding:** Peak discriminability at **L28**, not L12!

---

### Step 16: L18H16 Inhibitor Deep Dive

**Why does ablating L18H16 IMPROVE multi-agent performance?**

| Scenario Type | Entropy | Agent Attention |
|---------------|---------|-----------------|
| Single-Agent | 1.96 | 0.285 |
| Multi-Agent | **2.12** | 0.239 |

**Answer:** L18H16 becomes MORE DIFFUSE in multi-agent scenarios!
- Higher entropy = more spread attention = diluted signal
- Removing it prevents this interference

---

### Step 17: Transcoder Analysis (Training Issues)

Attempted to train transcoders to understand MLP computation.
- L1 penalty too strong caused all-zero features
- Needs more training data/hyperparameter tuning
- Alternative approach in Step 18

---

### Step 18: Direct MLP Computation Analysis

**Passed belief difference direction through MLP weights directly.**

| Layer | Diff Norm | Alignment | Amplification |
|-------|-----------|-----------|---------------|
| L12 | 4.09 | 0.203 | 0.01x |
| L28 | 32.17 | 0.000 | 0.001x |
| L32 | 59.98 | 0.045 | 0.001x |

**MAJOR INSIGHT:** 
- MLPs DON'T amplify belief difference (all ~0.01x)
- Yet belief separability grows 15x (4 -> 60)!
- **Belief information is built through ATTENTION + RESIDUAL STREAM, not MLP!**

---

## Summary: Architecture of ToM

```
INPUT (prompt)
     |
     v
[L0-16] Early layers - feature extraction
     |
     v
[L18H16] INHIBITOR - becomes diffuse with multiple agents
     |
     v
[L28] PEAK belief discriminability (via attention)
     |
     v
[L32-34] ToM HEADS - track WHO (agents)
     |
     v
[MLP L32] BELIEF ENCODING (transforms but doesn't amplify)
     |
     v
OUTPUT (prediction)
```

**Key insight**: 
- ATTENTION does the heavy lifting for belief tracking
- MLPs transform but don't amplify
- Belief separability grows through residual stream accumulation

