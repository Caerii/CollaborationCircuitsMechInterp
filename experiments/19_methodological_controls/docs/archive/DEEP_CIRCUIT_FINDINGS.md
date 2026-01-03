# Deep Circuit Findings

## Date: December 24, 2025

---

## BREAKTHROUGH: Logit Lens Reveals the True Mechanism!

### The Model KNOWS the Correct Answer at L31!

| Verb | L31 Diff | Final Diff | Final Prediction |
|------|----------|------------|------------------|
| told | **+6.50** | -0.98 | WRONG |
| announced | **+7.88** | -0.12 | WRONG |
| will_tell | **+10.09** | +1.02 | CORRECT |

### Key Discovery: The Decision Flips in L32-L35

1. The model correctly computes ToM at layer 31
2. Something in layers 32-35 **OVERRIDES** this for certain verbs
3. Our ablation targets L17-L21 - **TOO EARLY**!

### Why Ablation Doesn't Help

We were ablating the WRONG layers! The "belief update" signal that causes failure
is in L32-L35, not L17-L21.

The heads we found (L17H4, L18H11, etc.) are involved in **computing** the answer,
not in the **final override** that causes the error.

### New Target: L32-L35 - CONFIRMED!

**BREAKTHROUGH: Ablating 10 specific late-layer heads FIXES "told"!**

| Verb | Baseline | Top 10 Ablation | Status |
|------|----------|-----------------|--------|
| told | -0.98 (wrong) | +0.53 | **FIXED!** |
| said | -1.11 (wrong) | +0.47 | **FIXED!** |
| announced | -0.12 (wrong) | +1.38 | **FIXED!** |

### The 10 Critical Late-Layer Heads

These heads cause the override:
1. L35H0, L35H1, L35H17
2. L33H6, L33H13, L33H17, L33H31
3. L32H6, L32H31
4. L34H17

Ablating just these 10 heads (out of 128 in L32-L35) is sufficient to fix ToM!

---

## Critical Discovery: Two Distinct Failure Modes

### Investigation 1: The "told" Mystery (step20)

We tested 13 verbs with attention pattern analysis:

| Verb | Baseline | Category |
|------|----------|----------|
| told | 0% | DIRECT communication |
| said | 0% | DIRECT communication |
| mentioned | 0% | DIRECT communication |
| clarified | 0% | DIRECT communication |
| stated | 0% | DIRECT communication |
| announced | 100% | FORMAL/PUBLIC |
| noted | 100% | WRITTEN/RECORDING |
| queried | 100% | QUESTIONING |
| communicated | 100% | FORMAL |
| provided | 100% | INDIRECT |
| supported | 100% | INDIRECT |
| manifested | 100% | INDIRECT |
| dispatched | 0% | (exception) |

### Investigation 2: Extended Verb Analysis (step21)

Tested 41 verbs with ablation:

**CRITICAL FINDING**: Ablation does NOT fix the 0% accuracy verbs!

| Verb | Baseline | Ablated | Boost |
|------|----------|---------|-------|
| told | 0% | 0% | 0% |
| said | 0% | 0% | 0% |
| mentioned | 0% | 0% | 0% |
| informed | 0% | 0% | 0% |
| stated | 0% | 0% | 0% |
| notified | 0% | 0% | 0% |
| texted | 0% | 0% | 0% |
| emailed | 0% | 0% | 0% |
| wrote | 0% | 0% | 0% |

This means **the circuit we identified doesn't control these verbs!**

---

## The Two Failure Modes

### Mode 1: Circuit-Mediated (fixable with ablation)
- Certain scenarios trigger the decision heads
- Ablating them restores correct ToM
- Example: Standard Sally-Anne with "told" in different contexts

### Mode 2: Direct Lexical Association (NOT fixable with ablation)
- Specific verbs directly trigger belief-update
- The decision heads are NOT involved
- The model has learned: "told" → "they know" → update belief
- This is a **shortcut** bypassing the circuit entirely

---

## Pattern: What Makes a Verb "Bad"?

### Bad Verbs (0% baseline, NOT fixable):
- **told, said, mentioned, stated, informed, notified**
- **emailed, texted, wrote** (direct digital communication)
- **had told, has told, was telling** (told variants)
- **admitted, conveyed**

### Good Verbs (100% baseline):
- **announced, declared, proclaimed** (public/formal)
- **asked, queried, inquired, questioned, requested** (bidirectional)
- **explained, communicated, reported** (formal)
- **hinted, implied, suggested, indicated, signaled** (subtle)
- **shouted, yelled, broadcast** (loud)
- **confessed, confided, disclosed, revealed** (emotional)
- **noted, documented, recorded** (recording acts)
- **will tell** (future tense!)

---

## Hypothesis: The "Knowledge Transfer" Heuristic

The model has learned a simple heuristic:
> If A **[direct-verb]** B something, then B **knows** it.

This heuristic is:
- **Lexically triggered** by specific verbs
- **NOT mediated** by the ToM circuit
- **Overrides** the circuit's output

### Why "told" is special:
1. Most common communication verb in training data
2. Strongest association with knowledge transfer
3. Creates an "inescapable" shortcut

### Why "announced" works:
1. Less common in training data
2. Public context doesn't imply direct knowledge transfer
3. Model doesn't have the same shortcut

---

## Implications

### For ToM Research:
1. **Multiple mechanisms** contribute to ToM failures
2. **Lexical shortcuts** may be more important than circuits for common verbs
3. **Ablation studies** can miss important failure modes

### For Circuit Analysis:
1. The decision heads are **one** mechanism, not **the** mechanism
2. Need to investigate the **embedding** level for lexical effects
3. May need to intervene at earlier layers

### For Interventions:
1. **Ablation alone is insufficient** for some verbs
2. May need **vocabulary-level** interventions
3. Or **fine-tuning** to break lexical associations

---

## Next Steps

1. **Probe the embedding space** - Do "told" and "announced" have different directions?
2. **Early layer analysis** - Where does the lexical shortcut activate?
3. **Logit lens** - Track how predictions evolve through layers
4. **Attention at word level** - Does the model even "see" Alice correctly for "told"?

---

## Attention Pattern Insights

From L18H11 (key decision head):

| Pattern | Bad Verbs | Good Verbs |
|---------|-----------|------------|
| Attention to Alice | 0.19-0.23 | 0.17-0.37 |
| Attention to Basket | 0.07-0.19 | 0.09-0.26 |

**Surprise**: Attention patterns are NOT dramatically different!
- This confirms the failure is NOT in attention routing
- The failure is in how the attention output is processed

---

## REVISED Understanding: Two Separate Circuits

### Circuit 1: Early Decision Heads (L17-L21)
- Original discovery: L17H4, L18H11, L18H14, L19H30, L21H17
- These compute ToM-related representations
- Ablating them helps for SOME scenarios
- But NOT for "told", "said", etc.

### Circuit 2: Late Override Heads (L32-L35) - NEW!
- 10 heads that override correct ToM in final layers
- Critical for fixing "told", "said", "mentioned"
- L35H0, L35H1, L35H17, L33H6, L33H13, L33H17, L33H31, L32H6, L32H31, L34H17

### The Full Picture

```
Input → [Layers 0-16] → [Early Circuit L17-L21] → [Layers 22-31] → [Late Override L32-L35] → Output
                              ↓                                            ↓
                     Computes ToM signal                        Verb-specific override
                     (ablation helps some cases)                (ablation fixes "told")
```

### Why Both Circuits Matter

1. **Early circuit (L17-L21)**: Handles general ToM computation
   - Ablating helps when model is "unsure"
   - Doesn't help when verb triggers strong override

2. **Late circuit (L32-L35)**: Handles verb-specific override
   - Contains lexical associations ("told" → update beliefs)
   - Ablating removes the override, revealing correct ToM

### Optimal Intervention Strategy

For maximum ToM improvement, ablate BOTH circuits:
- Early heads: L17H4, L18H11, L18H14, L19H30, L21H17 (5 heads)
- Late heads: L35H0, L35H1, L35H17, L33H6, L33H13, L33H17, L33H31, L32H6, L32H31, L34H17 (10 heads)

Total: 15 heads out of 1,152 (32 heads × 36 layers) = **1.3% of attention heads**

---

## FINAL RESULTS: 100% ToM with 10-Head Ablation!

### Comprehensive Test (20 verbs)

| Configuration | Bad Verbs | Good Verbs | Mid Verbs | **Overall** |
|--------------|-----------|------------|-----------|-------------|
| Baseline | 0% | 100% | 100% | 55% |
| Early Only (L17-L21) | 0% | 86% | 75% | 45% |
| **Late Only (L32-L35)** | **100%** | **100%** | **100%** | **100%** |
| Combined | 100% | 100% | 100% | 100% |

### Key Findings

1. **The late circuit is THE key mechanism** - 10 heads in L32-L35
2. **Early circuit ablation HURTS performance** - Goes from 55% to 45%!
3. **Late circuit alone achieves 100%** - No need for early heads
4. **All "bad" verbs fixed**: told, said, mentioned, stated, informed, notified, wrote, emailed, texted

### The 10 Critical Heads (for any ToM intervention)

```
Late Circuit (L32-L35):
  L32: H6, H31
  L33: H6, H13, H17, H31
  L34: H17
  L35: H0, H1, H17
```

---

## Summary of Discoveries

1. **Model knows correct ToM at L31** - Logit lens shows correct answer dominates mid-layers
2. **Late layers (L32-L35) override correct answer** - Verb-specific mechanism
3. **10 specific late heads cause the override** - Targeted ablation achieves **100% ToM**
4. **Early circuit was a red herring** - Those heads don't help, may even hurt
5. **Single mechanism**: The late override circuit is sufficient to explain all failures

