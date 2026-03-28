# THE DEEP TRUTH: What Our Analysis Actually Revealed

**Date**: December 24, 2025

---

## The Journey of Discovery

We started with: "Model has ToM circuits that can be ablated"
We then claimed: "Model uses first-mention heuristic, no ToM"
After deeper analysis: "It's more complicated than either"

---

## What The Data Actually Shows

### Test 1: Location Bias Test (simple prompts)
```
"The person put it in X. Then moved it to Y. The person looks in the"

drawer -> basket: basket 68% WINS  (RECENCY)
basket -> drawer: drawer 88% WINS  (RECENCY)
box -> basket: basket 92% WINS     (RECENCY)
basket -> box: box 65% WINS        (RECENCY)
cabinet -> basket: basket 97% WINS (RECENCY)
basket -> cabinet: cabinet 80% WINS (RECENCY)
```
**Finding**: In simple prompts, RECENCY wins consistently.

### Test 2: Sally-Anne Format (complex narrative)
```
"Alice put the ball in X. Alice left. Bob moved to Y. Alice returned. Alice looks in"

drawer -> basket, left: drawer 65%  (NOT recency!)
basket -> drawer, left: drawer 88%  (Drawer always wins?!)
drawer -> basket, stayed: drawer 96%
basket -> drawer, stayed: drawer 83%
```
**Finding**: In Sally-Anne format, "drawer" wins regardless of order!

### Test 3: Nonsense Words
```
zork -> blep: zork 97% (FIRST wins)
blep -> zork: blep 80% (FIRST wins)
alpha -> beta: beta 26% (RECENT wins)
beta -> alpha: alpha 49% (RECENT wins)
```
**Finding**: With pure nonsense, first-mention wins. With familiar patterns, recency wins.

---

## The Complex Truth

The model uses **MULTIPLE COMPETING HEURISTICS** that activate differently based on context:

### Heuristic 1: Recency (Last-Mentioned)
- Dominates in simple, short prompts
- Predicts most recently mentioned location

### Heuristic 2: Narrative Structure / First-Mention
- Dominates with complete nonsense words
- Predicts originally-associated location

### Heuristic 3: Token Familiarity / Prior
- "drawer" has higher prior than "basket" in certain contexts
- This can override both recency and first-mention

### Heuristic 4: Prompt Format Effects
- Sally-Anne narrative structure triggers different behavior than simple prompts
- Specific completion phrases matter ("looks in" vs "will look" vs "searched")

---

## What About ToM?

### Evidence AGAINST ToM:
1. Model predicts same location whether agent "left" or "stayed" in many cases
2. Being "told" doesn't reliably update the prediction
3. Performance is ~50% when controlling for location bias (counterbalanced)

### Evidence FOR Some ToM-Like Processing:
1. "Alice left" does CHANGE predictions (drawer goes from 96% to 65%)
2. The change goes in a direction that could be ToM (less certain about first location)
3. But the magnitude is inconsistent

---

## The Honest Conclusion

```
┌────────────────────────────────────────────────────────────────────┐
│                        THE HONEST TRUTH                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. We CANNOT definitively say "no ToM" or "yes ToM"               │
│     - Evidence is mixed and context-dependent                       │
│                                                                     │
│  2. Standard Sally-Anne is a BAD test                               │
│     - Confounded by location bias                                   │
│     - Confounded by recency effects                                 │
│     - Success doesn't prove ToM                                     │
│                                                                     │
│  3. The model has COMPLEX, CONTEXT-DEPENDENT behavior               │
│     - Multiple heuristics compete                                   │
│     - Prompt format matters enormously                              │
│     - Location names matter                                         │
│                                                                     │
│  4. "Presence tracking" shows WEAK effects                          │
│     - "Alice left" changes behavior by ~30%                         │
│     - But direction is sometimes unexpected                         │
│     - Not the robust tracking we'd expect from true ToM             │
│                                                                     │
│  5. Our earlier circuit findings need RE-EVALUATION                 │
│     - We don't know what the circuits were actually doing           │
│     - They might have been modulating these heuristics              │
│     - Or something else entirely                                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## What We Should Have Done Differently

1. **Use neutral/counterbalanced locations from the start**
2. **Test with multiple completion phrases**
3. **Include "stayed" and "told" conditions as baselines**
4. **Use much larger sample sizes**
5. **Pre-register hypotheses before testing**

---

## The Meta-Lesson

This exploration shows why mechanistic interpretability is HARD:

- Surface behavior can be deceiving
- Multiple mechanisms can produce similar outputs
- Confounds are everywhere
- You need to test your tests

We went from:
- "90% ToM with circuit ablation!" → 
- "No ToM, just heuristics!" → 
- "It's complicated, we don't fully understand"

**The honest answer is: we need better experiments.**

---

## Recommendations for Future Work

1. **Design ToM tests that CANNOT be solved by recency or location bias**
2. **Use made-up location names that are counterbalanced**
3. **Test presence tracking with n>50 per condition**
4. **Compare to null models (recency-only, first-mention-only)**
5. **Look at attention patterns to understand mechanism**

---

*Science is the process of finding out how wrong you were.*


