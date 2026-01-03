# THE REAL TRUTH: Qwen3-4B Does NOT Have Theory of Mind

**Date**: December 24, 2025  
**Status**: Critical revision of all previous findings

---

## Executive Summary

After rigorous critical analysis, we discovered that:

**The model uses a HEURISTIC (likely first-mention or association), NOT actual Theory of Mind.**

Standard Sally-Anne tests pass BY ACCIDENT because the heuristic gives the correct answer.

---

## The Critical Evidence

### Test: Does the model update beliefs when agent SEES or is TOLD?

| Scenario | Model Predicts | Correct ToM | Result |
|----------|----------------|-------------|--------|
| Alice LEFT (didn't see) | drawer 91.9% | drawer | ✅ |
| Alice STAYED (saw move) | drawer 95.7% | **basket** | ❌ |
| Alice was TOLD | drawer 58.5% | **basket** | ❌ |
| Alice EXPLICITLY told | drawer 66.2% | **basket** | ❌ |

### The Smoking Gun

When Alice **stays and watches Bob move the ball**, she KNOWS where it is now.
True ToM would predict: Alice looks in the **basket** (where she saw it moved).
Model predicts: Alice looks in the **drawer** (where she originally put it).

**This is WRONG. The model ignores that Alice witnessed the event.**

---

## What This Means

### 1. Earlier "ToM Accuracy" Was Measuring the Wrong Thing

Our earlier findings:
- "22.5% baseline" → Wasn't testing ToM, was testing something else
- "76.7% with our prompts" → Also not ToM
- "90% with circuit ablation" → Irrelevant to actual ToM

All these numbers are meaningless because the test doesn't discriminate between heuristics and true ToM.

### 2. Standard Sally-Anne Is a Bad Test

Sally-Anne works by accident:
```
Agent's belief = Original location = First-mentioned location

So a simple first-mention heuristic gives the "correct" answer!
```

Better tests must include:
- Agent witnessed the move (should predict NEW location)
- Agent was told (should predict NEW location)
- These discriminate heuristics from ToM

### 3. "Inhibitory Circuits" Were Probably Irrelevant

We spent days hunting for circuits that "suppress ToM".
But the model doesn't HAVE ToM to suppress.
The circuits were probably doing something else entirely.

---

## The Real Picture

```
┌─────────────────────────────────────────────────────────────┐
│                     WHAT MODEL DOES                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT: "Alice put ball in drawer. [events]. Alice looks..." │
│                                                              │
│  MODEL PROCESS:                                              │
│  1. Parse "Alice put ball in drawer"                         │
│  2. Associate: Alice → drawer (strong)                       │
│  3. Ignore or weakly process intermediate events             │
│  4. Complete: "Alice looks in the drawer"                    │
│                                                              │
│  This is ASSOCIATIVE MEMORY, not ToM!                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 WHAT TRUE ToM WOULD DO                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Track Alice's KNOWLEDGE STATE                            │
│     - What events did Alice witness?                         │
│     - What was Alice told?                                   │
│                                                              │
│  2. Update belief based on knowledge                         │
│     - Alice left → Alice doesn't know about move             │
│     - Alice stayed → Alice DOES know about move              │
│     - Alice told → Alice DOES know about move                │
│                                                              │
│  3. Predict based on Alice's BELIEF, not reality             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Discriminating Test Suite

To properly test ToM, use these scenarios:

### Must-Pass (Agent doesn't know)
```
Alice put ball in drawer. Alice LEFT. Bob moved ball to basket. 
Alice returned. Alice looks in the ___
→ CORRECT: drawer (Alice's belief)
```

### Must-Pass (Agent knows - witnessed)
```
Alice put ball in drawer. Bob moved ball to basket. 
Alice looks in the ___
→ CORRECT: basket (Alice saw the move!)
```

### Must-Pass (Agent knows - told)
```
Alice put ball in drawer. Alice left. Bob moved ball to basket.
Bob told Alice "The ball is in the basket". Alice looks in the ___
→ CORRECT: basket (Alice was informed!)
```

### Scoring
- Heuristic model: ~1/3 correct (passes first, fails second and third)
- True ToM model: 3/3 correct

**Qwen3-4B score: 1/3 = NO ToM**

---

## Implications for MATS Mission

### 1. Revise Research Direction
- Stop hunting for "ToM circuits" - they may not exist
- Instead, investigate what the model ACTUALLY does
- Study the heuristics it uses

### 2. Multi-Agent Collaboration
- Models can coordinate via EXPLICIT communication
- Don't rely on implicit belief tracking
- Make all knowledge states explicit in protocols

### 3. Future Work
- Test other models with discriminating suite
- Find models that pass ALL three scenarios
- Understand what training enables true ToM

---

## Conclusion

**We were fooled by standard Sally-Anne.**

The model doesn't have Theory of Mind. It has associative heuristics that happen to give correct answers on poorly-designed tests.

This is humbling. Science requires:
- Discriminating tests
- Critical self-analysis
- Willingness to abandon cherished findings

---

*The truth is what remains after you've eliminated what you want to believe.*


