# Methodological Critique & Improvements

**Date**: December 24, 2025  
**Status**: Critical self-assessment of our ToM research

---

## Executive Summary: Issues Identified

| Severity | Issue | Impact |
|----------|-------|--------|
| 🔴 HIGH | Sample size too small | Claims lack statistical power |
| 🔴 HIGH | Prompt confounds not controlled | Can't isolate verb effect |
| 🟡 MEDIUM | No baseline comparisons | Don't know if effects are meaningful |
| 🟡 MEDIUM | Single model family tested | Can't generalize findings |
| 🟡 MEDIUM | Correlation ≠ causation in neuron analysis | Pathway may be spurious |
| 🟢 LOW | Benchmark selection not rigorous | May have cherry-picked |

---

## 1. Statistical Issues

### 1.1 Sample Size Problems

**Current state:**
- ToMi: 7 scenarios → "100% accuracy"
- Verb test: 3-4 verbs per category → "0% vs 100%"
- Cross-model: 3 models × ~10 tests each

**Problems:**
- 7/7 correct has 95% CI of [59%, 100%] using Wilson interval
- 0/3 "belief verb" failures could occur by chance (p ≈ 0.13 for 3 coin flips)
- No statistical tests reported (t-tests, chi-square, etc.)

**What we should have done:**
```python
# Minimum sample sizes needed:
# - 80% power to detect 30% accuracy difference
# - n ≈ 50 per condition minimum
# - With confidence intervals

from scipy import stats
# Test if 0/3 vs 3/3 is significant
result = stats.fisher_exact([[0, 3], [3, 0]])
# p = 0.10 - NOT SIGNIFICANT!
```

**Fix required:** Test with n ≥ 50 scenarios per condition.

### 1.2 No Confidence Intervals

We report "100% accuracy" and "0% accuracy" as if certain.

**Fix required:** Report:
- Point estimates with 95% CIs
- Effect sizes (Cohen's d, odds ratios)
- P-values where applicable

---

## 2. Confound Issues

### 2.1 Verb vs. Syntax Confound

**The problem:**

We compare:
- "Alice searched in the" (action)
- "Alice thinks the ball is in the" (belief)

But these differ in:
1. **Syntax**: direct object vs. embedded clause
2. **Length**: 4 tokens vs. 7 tokens
3. **Structure**: "X [verb] in the" vs. "X [verb] [that] Y is in the"

**We can't conclude it's the VERB causing the difference!**

**What we should test:**
```
CONTROL 1: Matched syntax
- "Alice searched for the ball in the" vs. "Alice looked for the ball in the"
- Same length, same structure, different verb

CONTROL 2: Minimal verb swap
- "Alice found the ball in the" vs. "Alice believes the ball is in the"
- But these have different syntax...

PROPER TEST:
- "Alice [VERB] that the ball is in the ___"
- Fill [VERB] with: says, thinks, knows, remembers, expects
- NOW we're testing verbs with controlled syntax
```

### 2.2 Narrative Context Confound

We found "well-structured prompts" work better, but:
- What specifically makes a prompt "well-structured"?
- Is it length? Explicit return mention? Temporal markers?
- We haven't isolated these factors.

**Fix required:** Systematic ablation of narrative elements.

---

## 3. Causal Claims Without Causal Evidence

### 3.1 Neuron Analysis (Gate 9519 → Neuron 716)

**What we found:**
- Gate neuron 9519: diff = +2.63 between action/belief
- Neuron 716 is highly weighted contributor

**Problems:**
1. diff = 2.63 is SMALL (out of typical range ±30)
2. We found CORRELATION, not causation
3. We didn't ablate neuron 9519 to verify

**What causation would require:**
```python
# 1. Ablate neuron 9519 specifically
# 2. Test if verb-type effect disappears
# 3. If yes, it's causal. If no, it's correlational.
```

### 3.2 Attention Pattern Analysis

**Claim:** "Model attends 4-9x more to action verbs"

**Problems:**
1. Attention is zero-sum (more to X = less to others)
2. Higher attention doesn't mean "more important"
3. We didn't do attention knockout to verify

**Alternative hypothesis:** Action verbs are just later in the sequence, and recency bias drives attention.

---

## 4. Generalizability Issues

### 4.1 Single Model Family

**Tested:** Qwen3-4B, Qwen2.5-1.5B, Qwen2.5-0.5B

**Problem:** All Qwen models share:
- Same tokenizer
- Same training data (likely)
- Same architectural choices

**Can't generalize to:**
- Llama family
- Mistral family  
- GPT family
- Claude family

**Fix required:** Test at least 2-3 different model families.

### 4.2 English Only

All tests in English. Effects might be:
- Language-specific
- Based on English training data distribution

**Fix required:** Test in Chinese (Qwen's other primary language).

### 4.3 Domain Limited

Almost all scenarios use:
- Object location (ball/drawer/basket)
- Same character names (Alice, Bob)
- Standard narrative structure

**Fix required:** Diverse domains, characters, structures.

---

## 5. Benchmark Issues

### 5.1 Not Using Official Benchmarks

We created "ToMi-style" scenarios but didn't use:
- Official ToMi dataset
- FANToM full benchmark
- Hi-ToM benchmark

**Risk:** Cherry-picked "easy" scenarios that match our hypothesis.

### 5.2 No Baseline Comparisons

We don't know:
- How does Qwen compare to GPT-4 on same tasks?
- Is 100% on 7 scenarios good or expected?
- What's the random baseline?

---

## 6. Reproducibility Issues

### 6.1 Missing Details

Not documented:
- Random seeds
- Exact model version/checkpoint
- Temperature setting (we assume 0, but not stated)
- Inference settings (top_k, top_p, etc.)

### 6.2 No Replication

Each experiment run once. No:
- Multiple runs to check variance
- Cross-validation
- Bootstrap confidence intervals

---

## 7. Interpretational Issues

### 7.1 "Model Has ToM" Overclaim

We say: "100% on ToMi → Model has functional ToM"

**Problems:**
- ToMi tests one narrow aspect of ToM
- Pattern matching could achieve same result
- We haven't shown the model REASONS about beliefs

**More accurate claim:** "Model performs correctly on simple false belief tasks when prompts are structured appropriately."

### 7.2 Mechanism vs. Behavior Conflation

We found BEHAVIORAL patterns (verb type affects accuracy).
We claim to have found MECHANISTIC explanation (neuron pathway).

But the mechanistic evidence is weak (correlation only).

---

## Required Fixes

### Priority 1: Statistical Rigor
```python
# TODO: step39_statistical_validation.py
# - Test with n=50 scenarios per condition
# - Report confidence intervals
# - Run statistical tests
# - Calculate effect sizes
```

### Priority 2: Control Confounds
```python
# TODO: step40_syntax_controlled_test.py
# - Matched-syntax comparisons
# - Isolate verb effect from structure effect
```

### Priority 3: Causal Verification
```python
# TODO: step41_neuron_ablation_causal.py
# - Ablate specific neurons (9519, 716)
# - Verify causal role in verb-type effect
```

### Priority 4: Cross-Architecture Validation
```python
# TODO: step42_cross_architecture.py
# - Test Llama-3.2-3B
# - Test Mistral-7B-Instruct
# - Compare patterns
```

### Priority 5: Official Benchmarks
```python
# TODO: step43_official_benchmarks.py
# - Use official ToMi dataset
# - Full FANToM evaluation
# - Compare to published results
```

---

## Revised Claims (More Accurate)

| Original Claim | Revised Claim |
|----------------|---------------|
| "100% ToM accuracy" | "7/7 correct on ToMi-style scenarios (95% CI: 59-100%)" |
| "Verb type is critical factor" | "Verb type correlates with accuracy (confounds not fully controlled)" |
| "Found mechanistic pathway" | "Found correlated neurons (causal role not verified)" |
| "Universal across Qwen" | "Consistent in 3 Qwen models (family-specific, not universal)" |
| "Model has functional ToM" | "Model performs correctly on simple false belief tasks with appropriate prompting" |

---

## What This Research Actually Shows

**Strong evidence for:**
1. Prompt format affects ToM performance in Qwen3-4B
2. "thinks/believes" type completions correlate with failures in minimal formats
3. Well-structured prompts generally work
4. Pattern is consistent within Qwen family

**Weak/unclear evidence for:**
1. The CAUSE of verb-type differences
2. Mechanistic pathway specifics
3. Generalizability to other models
4. Whether this is "real" ToM

---

## Recommended Next Steps

1. **Statistical validation** with proper sample sizes
2. **Controlled experiments** isolating verb vs. syntax
3. **Causal neuron ablation** to verify pathway
4. **Cross-architecture testing** (Llama, Mistral)
5. **Official benchmark evaluation** for comparability
6. **Revised publication** with tempered claims

---

## UPDATE: Validation Tests Run (Step 39-40)

### Statistical Validation (Step 39) - CLAIMS REFUTED

Testing 50 scenarios per condition with proper controls:

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| Action verbs | 58% | [44%, 71%] |
| Belief verbs | 54% | [40%, 67%] |

**Fisher's exact p-value: 0.84 (NOT SIGNIFICANT)**  
**Effect size (Cohen's h): 0.08 (TINY)**

⚠️ **Our original "100% vs 0%" claims were based on n=3-7 samples and were statistically meaningless!**

### Syntax-Controlled Test (Step 40) - CONFOUND IDENTIFIED

When syntax is held constant:

| Frame | Accuracy |
|-------|----------|
| Action frame ("expects", "remembers") | 73% |
| Belief frame ("believes", "assumes") | **80%** |

**The belief verbs actually performed BETTER when syntax was controlled!**

Key findings:
- "believes it to be in the" = **100%** accuracy
- "thinks it is in the" = **0%** accuracy  
- The difference is **syntax structure**, not verb type!

### What We Actually Learned

1. **The "verb type" effect is a confound** - syntax/structure matters more
2. **"thinks it is in the" fails due to embedded clause syntax**, not "thinks"
3. **Our n=7 "100%" claims were statistically meaningless**
4. **Proper science requires n≥50 with controlled comparisons**

### Revised Understanding

The model's ToM performance depends on:
1. **Syntactic structure** (direct object vs. embedded clause) - PRIMARY
2. **Prompt length and context** - SECONDARY  
3. **Verb semantics** - MINOR/NONE after controlling for syntax

This is a **major correction** to our earlier findings.

---

*Self-critique generated to maintain scientific rigor.*

