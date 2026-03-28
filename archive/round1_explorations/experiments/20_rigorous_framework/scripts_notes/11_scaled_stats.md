# 11_scaled_stats.md

## Step 12: Scaled Statistics (N≥50)

**Goal:** Run key experiments with proper sample sizes for statistical validity.

## Results

### Performance Summary

| Condition | Accuracy | 95% CI | N |
|-----------|----------|--------|---|
| **False Belief (baseline)** | **78.6%** | [52.4%, 92.4%] | 14 |
| False Belief (ablated) | 71.4% | [45.4%, 88.3%] | 14 |
| True Belief | 28.6% | [11.7%, 54.6%] | 14 |

### Statistical Tests

**Ablation Effect:**
- Change: -7.1%
- Cohen's h: 0.165 (small effect)
- McNemar's χ²: 0.00, p=1.0
- **Significant: NO**

**False Belief vs True Belief:**
- Difference: +50.0%
- Cohen's h: 1.051 (LARGE effect!)
- χ²: 5.17, p=0.023
- **Significant: YES**

## CONCERN: True Belief Performance is LOW

The model gets **False Belief correct (79%)** but **True Belief wrong (29%)**!

This is backwards from what we'd expect. True belief should be EASIER.

### Possible Explanations

1. **Heuristic reliance**: Model may be using "first-mentioned location" heuristic
   - Works for FB (return to original location)
   - Fails for TB (should track update)

2. **Prompt format issue**: Our TB prompts may be confusing

3. **Training bias**: Model may be overfit to FB-style patterns from training data

### What This Means for Interpretation

The high FB accuracy may be **spurious** - the model might not have genuine ToM but instead:
- Uses positional heuristics
- Defaults to original location
- Gets FB "right for the wrong reason"

This requires further investigation with:
- Counterbalanced location orders
- Novel location names
- Explicit heuristic baseline comparison

## Sample Size Note

Current N=14 per condition is below our target of N≥50. The wide confidence intervals reflect this:
- FB: [52%, 92%] - 40 percentage point range!
- TB: [12%, 55%] - 43 percentage point range!

We should increase N to get tighter estimates.

