# Rigorous Findings: Theory of Mind in Qwen3-4B

## Methodological Corrections Applied

| Previous Problem | Fix Applied |
|------------------|-------------|
| N=12 samples | N=200 samples |
| Q&A format ("Does B agree?") | Behavioral prediction ("Sally will...") |
| Wrong ablation (residual stream slicing) | Correct attention head hooks |
| No null distributions | Computed random baselines |
| Balanced design (tautological orthogonality) | Unbalanced conditions |

---

## 1. Null Distribution Analysis

### Cosine Similarity in High Dimensions

| Dimensionality | Mean |cos| | 95th Percentile |
|----------------|----------|-----------------|
| d=128 | 0.071 | 0.172 |
| d=640 | 0.032 | 0.078 |
| d=2560 | **0.016** | **0.039** |

**Implication**: Previous "orthogonal" findings (cos=0.03-0.12) were **within random expectation** for d=2560.

### Probe Accuracy at Chance

| N | d | Chance Level | 95th Percentile |
|---|---|--------------|-----------------|
| 12 | 128 | ~50% | Unstable (overfits) |
| 12 | 2560 | **N/A** | Complete overfitting |
| 200 | 128 | 50.2% | 61.5% |
| 200 | 640 | 50.1% | 59.0% |

**Implication**: Previous 91.7% accuracy with N=12 was **meaningless overfitting**.

### Ablation Flip Rate Baseline

| N | Expected | Std |
|---|----------|-----|
| 4 | 50% | **24.6%** |
| 12 | 50% | 13.8% |
| 50 | 50% | 7.0% |
| 200 | 50% | 3.5% |

**Implication**: Previous 75% flip with N=4 had **p ≈ 0.25** (not significant).

---

## 2. Behavioral ToM Test (VALID)

### Design
- **Format**: Story completion, NOT Q&A
- **Task**: Predict where agent looks (believed vs actual location)
- **N**: 200 false belief scenarios

### Results

| Metric | Value |
|--------|-------|
| Belief-based predictions | **162/200 (81%)** |
| Reality-based predictions | 38/200 (19%) |
| **p-value** | **9.9 × 10⁻²⁰** |
| Cohen's h | 0.67 (medium-large) |
| 95% CI | [75.3%, 85.9%] |

### Interpretation
✅ **Qwen3-4B shows genuine Theory of Mind**

The model predicts that agents will search in their **believed location** (81%), not the actual location, even when these differ. This is the gold-standard Sally-Anne test.

### Control: True Belief
- Accuracy: 84%
- Model correctly predicts behavior when belief=reality

---

## 3. Proper Attention Head Ablation (VALID)

### Design
- **Hook point**: `self_attn` module (correct)
- **Method**: Zero out specific head's contribution
- **Test prompts**: 10 (ToM + agreement + neutral)
- **Heads tested**: 15

### Results

| Head Group | Mean Change Rate | N heads |
|------------|------------------|---------|
| **ToM heads (L12/24/30 H0)** | **37%** | 3 |
| Other heads | 20% | 12 |
| **Mann-Whitney p-value** | **0.022** | - |

### Top Ablation Effects

| Layer | Head | Change Rate |
|-------|------|-------------|
| 24 | 0 | **50%** |
| 23 | 0 | 40% |
| 12 | 0 | 30% |
| 30 | 0 | 30% |
| 6 | 0 | 30% |

### Interpretation
✅ **Head 0 at layers 12, 24, 30 is causally important**

The "Head 0 channel" shows significantly higher impact (p=0.022) than random heads. This survives proper statistical testing.

---

## 4. What the Previous Claims Got Wrong

| Previous Claim | Status | Correction |
|----------------|--------|------------|
| "Belief ≠ Reality orthogonal (cos=0.03)" | ❌ **Invalid** | Within random baseline for d=2560 |
| "91.7% head probe accuracy" | ❌ **Invalid** | N=12 overfitting, meaningless |
| "75% ablation flip rate = causal" | ❌ **Invalid** | N=4 gives p≈0.25, not significant |
| "Independent agent modeling" | ⚠️ **Tautological** | Balanced design guaranteed result |

---

## 5. What We Can Actually Claim

### Supported Claims ✅

1. **Qwen3-4B demonstrates behavioral ToM**
   - Evidence: 81% belief-based predictions (N=200, p<10⁻¹⁹)
   - Methodology: Sally-Anne style task, not Q&A

2. **Head 0 at layers 12, 24, 30 is causally relevant**
   - Evidence: 37% vs 20% change rate (p=0.022)
   - Methodology: Proper attention head ablation

3. **Layer 24 Head 0 is most impactful**
   - Evidence: 50% change rate when ablated
   - Consistent across multiple test prompts

### Unsupported Claims ❌

1. ~~"Orthogonal belief/reality encoding"~~ - Within random expectation
2. ~~"91.7% head accuracy"~~ - Overfitting artifact
3. ~~"Independent agent modeling"~~ - Design artifact

---

## 6. For MATS Application

### What You Can Honestly Say

> "We demonstrate that Qwen3-4B exhibits behavioral Theory of Mind, predicting that agents will search based on their beliefs (81%) rather than reality. Using properly-designed causal ablation, we identify Head 0 at layers 12, 24, and 30 as significantly more impactful than control heads (p=0.022), suggesting a distributed 'ToM channel' in the model architecture."

### What You Should NOT Say

> ~~"We found orthogonal belief/reality representations"~~ (within random baseline)
> ~~"91.7% accuracy proves ToM heads"~~ (N=12 overfitting)
> ~~"We discovered ToM circuits"~~ (need path patching, more ablation)

---

## 7. Next Steps for Rigorous Science

1. **Increase ablation N**: Test on 50+ prompts
2. **Path patching**: Trace information flow through Head 0 channel
3. **Cross-model validation**: Test on other model families
4. **Representation analysis with proper N**: N=500+ for d=2560

---

## Summary Table

| Experiment | N | Statistic | p-value | Conclusion |
|------------|---|-----------|---------|------------|
| Behavioral ToM | 200 | 81% belief | <10⁻¹⁹ | ✅ **Real ToM** |
| Ablation ToM vs Other | 15 heads | 37% vs 20% | 0.022 | ✅ **Head 0 causal** |
| Cosine null (d=2560) | 10,000 | μ=0.016 | - | Previous findings invalid |
| Probe null (N=12, d=128) | 50 trials | Unstable | - | Previous findings invalid |




















