# Corrected Findings: After Methodological Critique

**Date**: December 24, 2025  
**Status**: Post-critique validation complete

---

## Executive Summary

Our methodological critique (Step 39-43) revealed that:

1. **Original "verb type" claims were CONFOUNDED by syntax**
2. **Effect size is model-dependent** (larger models are robust)
3. **Real patterns exist but are different than initially thought**

---

## Original Claims vs. Corrected Reality

| Original Claim | Statistical Test | Corrected Finding |
|----------------|------------------|-------------------|
| "Action verbs: 100%" | n=50: **58%** | Overclaimed due to small n |
| "Belief verbs: 0%" | n=50: **54%** | Overclaimed due to small n |
| "Verb type is critical" | p=0.84 (NS) | **Syntax is the real factor** |
| "Universal across Qwen" | Cross-model test | **Model size dependent** |

---

## Key Discoveries (Validated)

### 1. Syntax Structure Matters (Step 41)

The failure trigger is: **[explicit noun phrase] + [finite "is"]**

| Pattern | Example | Result |
|---------|---------|--------|
| Pronoun + is | "thinks it is in the" | ✅ +1.30 |
| Explicit noun + is | "thinks the ball is in the" | ❌ -0.50 |
| Explicit + infinitive | "believes the ball to be in the" | ✅ +1.53 |

**Mechanism**: Explicit noun phrases in finite clauses trigger factual interpretation.

### 2. Model Size Dependency (Step 42)

| Model Size | ToM Performance | Syntax Sensitivity |
|------------|-----------------|-------------------|
| 4B (Qwen3) | 100% | None (robust) |
| 1.5B (Qwen2.5) | 60-80% | **+20% for infinitive** |
| 0.5B (Qwen2.5) | 40% | None (too small) |

**Key insight**: Syntax effects only matter for medium-sized models!

### 3. Real Semantic Effects (Step 43)

Statistically significant verb pair differences:

| Comparison | Result | p-value | Interpretation |
|------------|--------|---------|----------------|
| expects vs suspects | +80% | **p=0.0007*** | Expectation beats suspicion |
| sees vs imagines | **-70%** | **p=0.0055** | Perception FAILS, imagination WORKS |
| find vs lose | **-60%** | *p=0.0198* | Finding FAILS, losing WORKS |

**Surprising**: Perceptual verbs (sees, perceives) have WORST performance (20%)!

### 4. Tense Effect (Step 43)

| Tense | Accuracy |
|-------|----------|
| Past ("looked") | 100% |
| Perfect ("has looked") | 100% |
| Present ("looks") | 90% |
| Progressive ("is looking") | 90% |
| **Future ("will look")** | **40%** |

**p = 0.01** for past vs future comparison.

---

## Revised Mechanistic Model

```
Input Prompt
    ↓
┌─────────────────────────────────────────┐
│ SYNTAX PARSING                          │
│ - Explicit noun + finite "is"? → FAIL   │
│ - Pronoun or infinitive? → PASS         │
│ - Future tense? → DEGRADED              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ MODEL CAPACITY CHECK                    │
│ - 4B+: Robust, handles edge cases       │
│ - 1-2B: Syntax sensitive                │
│ - <1B: Generally fails ToM              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ SEMANTIC VERB EFFECTS                   │
│ - Perceptual verbs: BAD (20%)           │
│ - Action verbs: GOOD (77%)              │
│ - Mental state verbs: MIXED (75%)       │
│ - Memory verbs: GOOD (75%)              │
└─────────────────────────────────────────┘
    ↓
Output
```

---

## What We Were Wrong About

1. **"Verb type is the critical factor"** → WRONG
   - Syntax structure (explicit noun + finite clause) is the real issue
   
2. **"Action verbs always work, belief verbs always fail"** → WRONG
   - "sees" (perceptual/action-like) = 10% accuracy
   - "imagines" (mental) = 80% accuracy
   
3. **"100% vs 0% accuracy"** → WRONG
   - Was based on n=3-7 samples
   - Real difference: ~4% (not significant)
   
4. **"Effect is universal"** → WRONG
   - Large models: Robust
   - Small models: Fail anyway
   - Only medium models show syntax sensitivity

---

## What We Were Right About

1. **Prompt format affects ToM** ✓ (just not the factors we thought)
2. **Future tense is problematic** ✓ (40% vs 100%, p=0.01)
3. **Well-structured prompts help** ✓ (but syntax, not "verb type")
4. **Model has functional ToM on benchmarks** ✓ (100% on ToMi for 4B)

---

## Recommendations

### For Researchers

1. **Use n≥50 with proper statistics** - our n=7 claims were worthless
2. **Control for syntax** - verb comparisons need matched structure
3. **Test multiple model sizes** - effects are scale-dependent
4. **Pre-register hypotheses** - we found patterns post-hoc

### For Practitioners

1. **Use larger models (4B+)** for robust ToM
2. **Avoid future tense** in ToM prompts
3. **Prefer pronouns** over explicit nouns in embedded clauses
4. **Use infinitive clauses** ("to be") over finite ("is") when possible

---

## Scripts Reference

| Script | Purpose | Key Finding |
|--------|---------|-------------|
| step39_statistical_validation.py | Proper n=50 test | Original claims NS |
| step40_syntax_controlled.py | Isolate syntax | Syntax confound found |
| step41_syntax_deep_dive.py | Detailed syntax | Explicit+finite = failure |
| step42_cross_architecture.py | Model size | Scale-dependent effect |
| step43_real_patterns.py | Valid effects | Tense, perceptual verbs |

---

## Conclusion

Our original hypothesis that "verb type" (action vs. belief) determines ToM accuracy was **methodologically flawed** and **largely incorrect**.

The actual factors are:
1. **Syntactic structure** (explicit noun + finite clause fails)
2. **Model capacity** (larger models are robust)
3. **Tense** (future fails)
4. **Specific verb semantics** (perceptual verbs fail regardless)

This is a valuable lesson in the importance of:
- Adequate sample sizes
- Controlled comparisons
- Cross-model validation
- Skepticism toward initial results

---

*Science corrects itself. This correction makes our findings more trustworthy.*


