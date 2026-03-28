# Revised Understanding: The Model's ToM Capability

## Executive Summary

**The model CAN infer belief updates. It's not an architectural limitation.**

Our initial finding (7-17% on belief updates) was due to prompt sensitivity, not capability gaps.

## Key Experimental Results

### Control Conditions (Experiment 19, Step 1)

| Condition | Accuracy | Change vs Baseline |
|-----------|----------|-------------------|
| Chain of thought | 100% | +82% |
| Few-shot | 100% | +82% |
| ToM framing | 96% | +78% |
| Narrative consequence | 96% | +78% |
| Social context | 82% | +64% |
| Baseline | 18% | -- |

### Decomposition Analysis (Experiment 19, Step 2)

| Minimal Intervention | Accuracy |
|---------------------|----------|
| "Therefore, X now believes..." | 100% |
| "X no longer thinks it's in [old location]" | 100% |
| "so X updated their belief" | 98% |
| "Because X heard this..." | 98% |
| "X now knows" | 92% |
| "X heard this" | 72% |
| Just "let's think step by step" | 54% |
| Baseline (no intervention) | 18% |

## What This Means

### The Capability Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL HAS:                                   │
│                                                                 │
│  1. Communication Detection                                     │
│     "Eve tells Alice: 'moved to basket'"                       │
│     -> Model recognizes this is communication                   │
│                                                                 │
│  2. Belief Representation                                       │
│     Can represent "Alice believes X"                            │
│     -> L12H0, L23H0 circuit processes this                     │
│                                                                 │
│  3. Belief-Based Prediction                                     │
│     "Alice believes X" -> predicts Alice acts on X             │
│     -> Works perfectly when belief is explicit                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ WEAK CONNECTION
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL NEEDS HELP WITH:                       │
│                                                                 │
│  "Communication happened" --> "Belief updated"                  │
│                                                                 │
│  This inference is NOT automatic, but CAN be triggered by:     │
│  - Explicit bridge: "so X updated their belief"                │
│  - Contrastive: "X no longer thinks..."                        │
│  - Causal: "Because X heard this..."                           │
│  - Knowledge state: "X now knows..."                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Multi-Agent Software Dev Works

In software development prompts, the "belief update bridge" is often implicit:

```
DEVELOPER: "Here's my code"
REVIEWER: "I found a bug on line 5"  
DEVELOPER: "I see your point..."  <-- Implicit acknowledgment
```

The phrase "I see your point" serves as the bridge - it signals "I heard and processed this information."

### Why Sally-Anne Fails (Without Scaffolding)

```
"Eve tells Alice: 'I moved the ball to the basket.'"
Where will Alice look?
```

No bridge! The model detects the communication but doesn't automatically infer the belief update.

### Minimal Fix

Add ANY of these:
- "...and Alice now knows the new location"
- "...so Alice updated her belief"
- "Because Alice heard this, Alice will look in the..."

---

## Implications for Our Research

### Circuit Discovery Focus

We should look for:
1. **The "communication detection" circuit** - This exists and works
2. **The "belief update bridge" circuit** - This is WEAK but EXISTS
3. **Why the bridge isn't automatically activated** - Training/architecture question

### For Multi-Agent Systems

**Best practice:** Always include explicit belief state markers

```python
# GOOD: Explicit belief state
"Alice received Bob's message and now believes the file is in /src/utils/"

# BAD: Implicit (relies on automatic inference)
"Bob told Alice the file is in /src/utils/"
```

### What We Originally Called "Architectural Limitation"

Was actually: **A weak but present inference pathway that needs explicit activation.**

This is more nuanced and arguably more interesting than a pure capability gap.

---

## Revised Research Questions

1. **Why is the communication->belief bridge weak?**
   - Training data distribution?
   - Attention pattern issue?
   - Missing circuit connection?

2. **Can we find and strengthen this bridge?**
   - Activation steering?
   - Fine-tuning on bridged examples?
   - Prompt engineering guidelines?

3. **Is this pattern consistent across models?**
   - Does Llama/Mistral show same pattern?
   - Is it universal or model-specific?

---

## Summary

| Original Claim | Revised Understanding |
|----------------|----------------------|
| "Model can't infer belief updates" | Model CAN, but needs explicit bridge |
| "Architectural limitation" | Weak inference pathway, not missing capability |
| "Need different training/architecture" | Can be solved with prompt engineering |
| "Fundamentally lacks ToM" | Has ToM, just needs scaffolding |

**The capability exists. The automatic trigger is weak. Prompting solves it.**


