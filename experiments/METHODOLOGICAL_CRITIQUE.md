# Methodological Critique & Improvements

## Critical Analysis of Our Current Approach

### Problem 1: Sample Size is Too Small

| Experiment | N Samples | Statistical Power |
|------------|-----------|-------------------|
| Baseline Probing | 1,186 | Adequate |
| Naturalistic Transfer | 303 | Marginal |
| Perspective Taking | 27 | **Insufficient** |
| Knowledge Tracking | 25 | **Insufficient** |

**Issue**: With 25 samples and 5 classes, we have ~5 samples per class. A linear classifier with 2,560 features can easily overfit. Even leave-one-out CV can give misleading results with such small N.

**Standard**: Interpretability studies typically use 1,000+ samples minimum.

---

### Problem 2: Lexical Confounds

Our scenario types have distinctive lexical patterns:

| Type | Distinctive Phrase |
|------|-------------------|
| secret_kept | "don't tell anyone", "keep it confidential" |
| info_shared | "share with helper", "let helper know" |
| false_belief | "don't tell yet", "preparing for [wrong thing]" |
| conflict | "I think we should", "but the user requested" |
| collaboration | "I think", "Good point" |

**Issue**: The model might be detecting these surface patterns, not abstract "knowledge state" concepts.

**Solution Needed**: Control task with scrambled labels (random baseline).

---

### Problem 3: No Control Tasks / Random Baseline

A fundamental principle from Hewitt & Liang (2019) "Designing and Interpreting Probes":

> "A probe that achieves high accuracy on a task might do so because the representations contain the relevant information, OR because the probe itself has enough capacity to memorize the training data."

**Solution**: We need:
1. **Random baseline**: Shuffle labels and measure probe accuracy
2. **Selectivity**: Probe_accuracy - Random_accuracy
3. Only claim encoding if selectivity > threshold

---

### Problem 4: Multiple Comparisons

We test 10 layers without correction:
- At α = 0.05, we expect 0.5 false positives by chance
- With small samples, spurious "findings" are likely

**Solution**: Apply Bonferroni or FDR correction.

---

### Problem 5: Effect Size Not Reported

Statistical significance ≠ meaningful effect. We should report:
- Cohen's d for effect size
- Confidence intervals
- Not just point estimates

---

### Problem 6: Generalization Claims Without Proper Test

Our "findings" are:
1. Trained on synthetic data
2. Tested on synthetic data from same distribution
3. Single model (Qwen3-4B)

**Issue**: No evidence findings generalize to:
- Other models
- Real conversations
- Different prompting styles

---

## What We Should Do

### Immediate Fixes

1. **Add random baseline control**
   - Shuffle labels, train same probe
   - Report selectivity = real_acc - random_acc

2. **Increase sample size**
   - Generate at least 200 samples per condition
   - Use different wordings for same scenario type

3. **Statistical tests**
   - Permutation tests for significance
   - Bootstrap confidence intervals
   - Multiple comparison correction

4. **Report effect sizes**
   - Cohen's d between conditions
   - Explained variance

### Rigorous Experiment Design

```
For each hypothesis:
1. State null hypothesis (H0)
2. Define minimum effect size of interest
3. Calculate required sample size for 80% power
4. Pre-register analysis plan
5. Run experiment
6. Compare to random baseline
7. Report confidence intervals
```

---

## Revised Interpretation of Our Findings

### Experiment 01-04: Entity Label Encoding
- **Claim**: Model encodes entity labels
- **Actual**: Model encodes "User:", "Self:", "Other:" TOKENS
- **Evidence strength**: STRONG (large N, transfer test as control)

### Experiment 05: Naturalistic Transfer
- **Claim**: Encoding is lexical, not semantic
- **Actual**: Probes fail without explicit labels
- **Evidence strength**: MODERATE (needs replication)

### Experiment 06: Perspective/Knowledge
- **Claim**: Model tracks perspectives/knowledge states
- **Actual**: INSUFFICIENT EVIDENCE (too few samples, no control task)
- **Evidence strength**: WEAK - needs proper replication

---

## Best Practices from Literature

1. **Belinkov (2022)**: "Probing Classifiers: Promises, Shortcomings, and Advances"
   - Use control tasks
   - Report selectivity
   - Consider probe complexity

2. **Hewitt & Liang (2019)**: "Designing and Interpreting Probes"
   - Random baselines essential
   - High probe accuracy ≠ feature encoding

3. **Voita & Titov (2020)**: "Information-Theoretic Probing"
   - Mutual information instead of accuracy
   - Controls for probe capacity

4. **General ML**: 
   - Train/test split (not just CV on small data)
   - Holdout set for final evaluation
   - Pre-registration of hypotheses
























