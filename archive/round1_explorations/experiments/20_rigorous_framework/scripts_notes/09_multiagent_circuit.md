# 09_multiagent_circuit.md

## Step 10: Multi-Agent Circuit Hunt

**Goal:** Find which heads matter for multi-agent reasoning (since ToM heads L32-34 had no effect).

## BREAKTHROUGH: Multi-Agent Uses DIFFERENT Circuit!

### Multi-Agent Circuit Location

```
ToM CIRCUIT (L32-34):          MULTI-AGENT CIRCUIT (L0-22):
┌─────────────────┐            ┌─────────────────┐
│  L32: Head H0   │            │  L0: H8, H24    │ ← EARLY
│  L33: H4,H16,H28│            │  L6: H0,H16,H24 │
│  L34: Head H0   │            │  L8: H0         │
└─────────────────┘            │  L12: H0,H8     │
     LATE LAYERS               │  L18: H16 ★★★  │ ← MOST IMPACT
                               │  L22: H8,H16   │
                               └─────────────────┘
                                    EARLY-MID LAYERS
```

### Key Findings

| Head | Effect when Ablated | Role |
|------|---------------------|------|
| **L18H16** | **+25%** | INHIBITOR (ablation HELPS!) |
| L0H8 | -12.5% | Enabler |
| L0H24 | -12.5% | Enabler |
| L6H24 | -12.5% | Enabler |
| L8H0 | -12.5% | Enabler |
| L12H0 | -12.5% | Enabler |

### Overlap Analysis

- **Only 1 head overlaps** between ToM and Multi-Agent: L34H0
- **20 heads are Multi-Agent ONLY** (not ToM)
- **4 heads are ToM ONLY** (not Multi-Agent)

### Implication: Separate Social Cognition Circuits

The model doesn't have a unified "social reasoning" system. Instead:

1. **Single-agent ToM** → Late layers (32-34)
   - "Where does Alice think the ball is?"
   
2. **Multi-agent reasoning** → Early-mid layers (0-22)
   - "Alice knows X, Bob knows Y"
   - Deception detection
   - Trust calibration

### L18H16: The Multi-Agent Inhibitor

The most impactful finding is **L18H16** - an INHIBITOR head!

When ablated, multi-agent performance IMPROVES by 25%. This suggests:
- The head may be "overthinking" or adding noise
- It might be competing with correct multi-agent reasoning
- Removing it allows cleaner signal flow

This is similar to how some language model heads have been found to actively suppress certain patterns.

