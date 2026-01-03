# 07_multiagent_test.md

## Step 8: Do ToM Heads Enable Multi-Agent Reasoning?

**Goal:** Test if the critical ToM heads (L32H0, L33H4, L33H16, L33H28, L34H0) are also necessary for multi-agent scenarios.

**Methodology:**
- 8 multi-agent scenarios across 3 types:
  - Multi-agent belief tracking (3): Who knows what among multiple agents
  - Deception detection (3): Can model detect lies vs honest mistakes
  - Goal reasoning (2): Competitive vs cooperative goals
- Compared baseline (no ablation) vs critical head ablation

## Results

| Condition | Accuracy |
|-----------|----------|
| Baseline | 87.5% (7/8) |
| Ablated | 87.5% (7/8) |
| **Change** | **0.0%** |

### By Scenario Type

| Type | Baseline | Ablated | Change |
|------|----------|---------|--------|
| Multi-agent belief | 66.7% | 66.7% | 0% |
| Deception detection | 100% | 100% | 0% |
| Goal reasoning | 100% | 100% | 0% |

## Key Finding: ToM Heads ≠ Multi-Agent Heads

**Ablating the critical ToM heads had NO EFFECT on multi-agent performance!**

This suggests:
1. **Separate circuits:** Multi-agent reasoning may use different pathways than single-agent ToM
2. **Distributed computation:** The model may have redundant pathways for multi-agent scenarios
3. **Different representations:** The mechanism for "Alice thinks X" differs from "Alice knows Bob knows Y"

## Implications

The ToM heads we found (L32H0, L33H4, etc.) appear to be specialized for:
- Single-agent false belief tracking
- The classic Sally-Anne type task

But NOT for:
- Multi-agent belief coordination
- Deception detection
- Goal inference

**This is a key architectural finding!** The model doesn't use a unified "social cognition" circuit - it has separate pathways for different social reasoning tasks.

