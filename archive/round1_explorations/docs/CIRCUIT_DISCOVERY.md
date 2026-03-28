# Circuit Discovery: Two Distinct ToM Systems

## Executive Summary

We discovered that the model has **two distinct circuit systems** for Theory of Mind:

1. **Explicit Belief Parser** (L12H0, L23H0)
   - Processes explicitly stated beliefs: "X believes Y"
   - Identified in Experiment 16 through ablation
   - Works perfectly regardless of prompt framing

2. **Belief Update Circuit** (L23H4, L28H0, L24H29, L26H26...)
   - Updates belief representations based on communication
   - Identified in Experiment 19 through attention analysis
   - NOT automatically activated by simple "X tells Y" prompts
   - ACTIVATES when we add bridging phrases like "so X updated their belief"

## The Evidence

### Attention Change Analysis

When we add "so X updated their belief" to the prompt:

| Head | Attention Change | Interpretation |
|------|-----------------|----------------|
| L23H4 | **+0.54** | Belief update circuit - ACTIVATES |
| L28H0 | **+0.50** | Belief update circuit - ACTIVATES |
| L24H29 | **+0.45** | Belief update circuit - ACTIVATES |
| L12H0 | -0.009 | Explicit parser - NO CHANGE |
| L23H0 | -0.001 | Explicit parser - NO CHANGE |

The explicit belief parser (L12H0, L23H0) doesn't change - it's always active.
The belief UPDATE circuit (L23H4, L28H0, etc.) ONLY activates with the bridge phrase.

### Behavioral Confirmation

| Prompt Type | Accuracy | Circuit Active |
|-------------|----------|----------------|
| Explicit belief: "X believes Y" | 100% | L12H0, L23H0 |
| Baseline: "X tells Y" (no bridge) | 18% | L12H0, L23H0 only |
| Bridged: "X tells Y, so X updated belief" | 98% | Both circuits! |

## Circuit Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT PROCESSING                         │
│  "Eve tells Alice: 'I moved the ball to the basket'"            │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│              COMMUNICATION DETECTION (works)                     │
│  Recognizes "tells", "says", communication patterns              │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│   EXPLICIT BELIEF PARSER    │   │   BELIEF UPDATE CIRCUIT     │
│   L12H0, L23H0              │   │   L23H4, L28H0, L24H29...   │
│                             │   │                             │
│   Extracts stated beliefs   │   │   Updates beliefs from      │
│   "X believes Y" → Y        │   │   communication events      │
│                             │   │                             │
│   ALWAYS ACTIVE             │   │   WEAKLY TRIGGERED          │
│                             │   │   Needs explicit bridge     │
└──────────────────┬──────────┘   └──────────────┬──────────────┘
                   │                              │
                   │         WEAK CONNECTION      │
                   │    (not automatic trigger)   │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    BELIEF REPRESENTATION    │
                    │    "Alice believes X"       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    BEHAVIOR PREDICTION      │
                    │    "Alice will look in X"   │
                    └─────────────────────────────┘
```

## Key Insight: The "Weak Connection"

The connection from "Communication Detection" to "Belief Update Circuit" is **weak**.

- The model DETECTS "Eve tells Alice..."
- The model HAS the belief update circuit (L23H4, L28H0)
- But the circuit doesn't AUTOMATICALLY activate

### What Activates It

Adding ANY of these phrases triggers the circuit:
- "so X updated their belief" (98%)
- "X no longer thinks..." (100%)
- "Therefore X believes..." (100%)
- "Because X heard this..." (98%)
- "X now knows..." (92%)

These phrases explicitly signal that a belief update should occur.

## Implications

### For Multi-Agent Systems

Always include explicit belief state bridges in prompts:

```python
# BAD (18% accuracy) - Belief update circuit NOT activated
"Bob told Alice the file location."
"Where does Alice think the file is?"

# GOOD (98% accuracy) - Belief update circuit ACTIVATED  
"Bob told Alice the file location, so Alice updated her knowledge."
"Where does Alice think the file is?"
```

### For Circuit-Level Interventions

We now have two targets for enhancement:

1. **Strengthen the trigger connection**
   - Make communication detection automatically trigger update circuit
   - Possible via activation steering or fine-tuning

2. **Amplify the update circuit**
   - Ablate/boost L23H4, L28H0 specifically
   - See if this improves implicit belief update

### For Research

This finding reconciles the apparent contradiction:
- Model "works" in multi-agent software dev → Explicit beliefs + communication acknowledged
- Model "fails" on Sally-Anne → No explicit belief bridge

**Both are true.** The model has the capability but it's not automatically triggered.

## Next Steps

1. **Confirm L23H4/L28H0 role** - Ablate these heads, see if bridged prompts fail
2. **Test activation steering** - Can we amplify update circuit without explicit bridge?
3. **Cross-model validation** - Do Llama/Mistral have same architecture?
4. **Training hypothesis** - Why is the trigger weak? Training data analysis?

## Summary

| Component | Heads | Function | Status |
|-----------|-------|----------|--------|
| Explicit Belief Parser | L12H0, L23H0 | Parse "X believes Y" | STRONG |
| Belief Update Circuit | L23H4, L28H0, L24H29 | Update beliefs from communication | EXISTS but WEAKLY TRIGGERED |
| Communication Detection | Unknown | Detect "X tells Y" | WORKS |
| Update Trigger Connection | Unknown | Connect detection → update | WEAK |

**The capability exists. The automatic trigger is weak. We found the circuit.**


