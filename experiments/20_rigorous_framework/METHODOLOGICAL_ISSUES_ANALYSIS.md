# Methodological Issues Analysis: Experiment 20 Framework

## Executive Summary

While the framework demonstrates strong awareness of methodological best practices, several critical issues exist in **implementation** that could lead to false discoveries and inflated claims. The framework has good **intentions** (n≥50, statistical tests, counterbalancing) but **execution** falls short in several areas.

---

## Critical Issues

### 1. **Multiple Comparisons Not Corrected in Practice** ⚠️ CRITICAL

**Issue**: Circuit discovery experiments test many heads/layers simultaneously without correcting for multiple comparisons.

**Evidence**:
- `step35_real_circuit_hunt.py` tests multiple layers × multiple heads (e.g., 5 layers × 4 heads = 20 tests)
- No Bonferroni or FDR correction applied in the analysis
- Functions exist (`bonferroni_correct`, `benjamini_hochberg`) but are not used in actual experiments

**Impact**: 
- With 20 tests at α=0.05, expect ~1 false positive by chance
- No correction means inflated Type I error rate
- Circuit claims may be spurious

**Recommendation**:
```python
# In step35_real_circuit_hunt.py, after collecting all ablation results:
from analysis.controls import bonferroni_correct

p_values = [compute_p_value(head_result) for head_result in all_heads]
corrected = bonferroni_correct(p_values, alpha=0.05)
# Only report heads that pass corrected threshold
```

---

### 2. **Sample Size Inconsistencies** ⚠️ CRITICAL

**Issue**: Despite requiring n≥50, many experiments use much smaller samples.

**Evidence**:
- `scripts_notes/11_scaled_stats.md`: Reports n=14 per condition (target was n≥50)
- `FINAL_RESEARCH_SUMMARY.md`: Reports n=20 for FB/TB (below requirement)
- `step35_real_circuit_hunt.py`: Uses only 4 scenarios (2 FB, 2 TB) - far below n≥50

**Impact**:
- Wide confidence intervals (e.g., [52%, 92%] for FB with n=14)
- Underpowered to detect real effects
- Cannot make reliable claims

**Recommendation**:
- Enforce minimum sample size at runtime
- Add validation checks that prevent experiments from running with n<50
- Document when exploratory work uses smaller n (and don't make claims)

---

### 3. **No Statistical Tests in Circuit Discovery** ⚠️ CRITICAL

**Issue**: Circuit ablation experiments report accuracy differences without statistical significance tests.

**Evidence**:
- `step35_real_circuit_hunt.py` (lines 247-266): Only reports accuracy differences, no p-values
- No tests for whether ablation effect is significantly different from zero
- No tests comparing ablation vs baseline

**Impact**:
- Cannot distinguish real effects from noise
- May report "helpful" heads that are actually random

**Recommendation**:
```python
# Add statistical tests:
from scipy import stats

# Test if ablation effect is significantly different from zero
baseline_acc = results['baseline']
ablation_acc = head_data['accuracy']
n = len(scenarios)

# McNemar's test for paired data (same scenarios, baseline vs ablated)
# Or binomial test if independent samples
p_value = stats.binom_test(
    int(ablation_acc * n), n, baseline_acc, 
    alternative='two-sided'
)
```

---

### 4. **Activation Patching in Chat Mode** ⚠️ MAJOR

**Issue**: Multiple notes document that activation patching corrupts generation in chat mode, yet experiments continue to attempt it.

**Evidence**:
- `scripts_notes/28_patching_corrupts_generation.md`: Documents corruption
- `scripts_notes/30_final_summary.md`: Concludes "activation patching doesn't work in chat mode"
- Yet `step36_causal_patching.py` and related scripts still attempt patching

**Impact**:
- Wasted computational resources
- Potentially misleading results if corruption is not detected
- Should use alternative methods (ablation, probing) instead

**Recommendation**:
- Document as known limitation
- Use ablation studies (like step35) instead of patching
- Focus on probing and transcoder analysis for chat mode

---

### 5. **Multiple Hypothesis Testing Across Experiments** ⚠️ MAJOR

**Issue**: Testing many hypotheses across 35+ steps without correction.

**Evidence**:
- 35+ experimental steps, each potentially testing multiple hypotheses
- No pre-registration of which hypotheses are primary vs exploratory
- No correction for testing across multiple experiments

**Impact**:
- High risk of false discoveries
- Cannot distinguish pre-planned vs post-hoc analyses
- P-hacking potential (testing many things, reporting only significant ones)

**Recommendation**:
- Pre-register primary hypotheses (already have `EXPERIMENT_PLAN.md` but need to mark which are primary)
- Use stricter thresholds for exploratory analyses
- Clearly label exploratory vs confirmatory results

---

### 6. **Inconsistent Methodology Application** ⚠️ MODERATE

**Issue**: Some experiments follow rigorous methodology, others don't.

**Evidence**:
- `step33_proper_retest.py`: Uses proper chat mode, sufficient tokens
- `step12_scale_up.py`: Claims n≥50 but actually uses n=14
- `step35_real_circuit_hunt.py`: Uses only 4 scenarios

**Impact**:
- Inconsistent quality across experiments
- Difficult to know which results are reliable
- Some claims may be based on weak evidence

**Recommendation**:
- Add runtime validation that enforces methodology
- Use `ResultValidator` in all experiments
- Fail fast if requirements not met

---

### 7. **No Power Analysis Before Experiments** ⚠️ MODERATE

**Issue**: Power analysis function exists but is not used to plan experiments.

**Evidence**:
- `analysis/controls.py` has `power_analysis()` function
- No evidence it's used to determine sample sizes
- Sample sizes seem arbitrary (n=14, n=20, n=50)

**Impact**:
- May be underpowered to detect real effects
- May be overpowered (wasteful) for large effects
- Cannot justify sample size choices

**Recommendation**:
- Run power analysis before each experiment
- Document required n for target effect size
- Adjust n based on preliminary results

---

### 8. **Probe Accuracy Interpretation** ⚠️ MODERATE

**Issue**: High probe accuracy (95%) may reflect lexical separability rather than semantic understanding.

**Evidence**:
- `scripts_notes/10_mlp_probing.md`: Reports 95% probe accuracy
- `FINAL_RESEARCH_SUMMARY.md` (line 195): Notes "100% probe accuracy reflects token separability, not semantic understanding"
- But probing results are still used as evidence of ToM

**Impact**:
- May overinterpret probe results
- Need to distinguish lexical vs semantic encoding

**Recommendation**:
- Test probes on novel names/scenarios (generalization test)
- Report both in-distribution and out-of-distribution probe accuracy
- Don't claim ToM based on probe accuracy alone

---

### 9. **True Belief Control Issues** ⚠️ MODERATE

**Issue**: Early experiments showed TB failing (29% accuracy), which was later corrected but raises questions about controls.

**Evidence**:
- `scripts_notes/11_scaled_stats.md`: TB at 28.6% (worse than chance!)
- Later corrected in `scripts_notes/17_critical_corrections.md`
- Suggests controls weren't working properly initially

**Impact**:
- If controls fail, cannot interpret FB results
- Need to ensure TB works before making FB claims

**Recommendation**:
- Always test TB first and ensure it works
- If TB fails, investigate before testing FB
- Document TB performance in all experiments

---

### 10. **No Replication Across Models** ⚠️ MODERATE

**Issue**: All experiments use Qwen3-4B only, no cross-model validation.

**Evidence**:
- `EXPERIMENT_PLAN.md` mentions cross-model validation (Step 13)
- No evidence it was done
- All results are model-specific

**Impact**:
- Cannot generalize findings
- May be Qwen3-4B-specific artifacts
- Limited scientific value

**Recommendation**:
- Test on at least 2-3 models
- Report which findings generalize vs model-specific
- Prioritize generalizable findings

---

## Positive Aspects

The framework does many things well:

1. ✅ **Awareness of issues**: Documents many methodological requirements
2. ✅ **Statistical tools**: Has functions for CIs, effect sizes, corrections
3. ✅ **Validation framework**: `ResultValidator` enforces requirements
4. ✅ **Self-correction**: Notes document corrections (e.g., step 17, 18)
5. ✅ **Proper baselines**: Compares to heuristics, not just chance
6. ✅ **Counterbalancing**: 8-scenario design when used
7. ✅ **Effect sizes**: Reports Cohen's h, not just p-values

---

## Priority Recommendations

### Immediate (Critical):
1. **Add multiple comparisons correction** to all circuit discovery experiments
2. **Enforce minimum n≥50** at runtime (fail if not met)
3. **Add statistical tests** to ablation experiments (test if effect ≠ 0)

### Short-term (Major):
4. **Stop activation patching in chat mode** - use ablation instead
5. **Pre-register primary hypotheses** - mark exploratory vs confirmatory
6. **Use power analysis** to plan sample sizes

### Long-term (Moderate):
7. **Test probe generalization** - novel names, out-of-distribution
8. **Cross-model validation** - test on multiple models
9. **Replication** - key findings should be replicated

---

## Example: How to Fix Step 35

```python
# Current (problematic):
results = analyzer.ablation_sweep(scenarios, layers_to_test, heads_per_layer=4)
all_heads.sort(key=lambda x: x['diff'])
print("Most HELPFUL heads:", all_heads[:5])  # No stats!

# Fixed version:
from scipy import stats
from analysis.controls import bonferroni_correct

# 1. Validate sample size
assert len(scenarios) >= 50, f"Need n≥50, got {len(scenarios)}"

# 2. Compute p-values for each head
p_values = []
for head_data in all_heads:
    # Test if ablation effect is significantly different from baseline
    baseline_correct = sum(baseline_results)
    ablation_correct = sum(head_data['individual'])
    n = len(scenarios)
    
    # McNemar's test (paired: same scenarios, baseline vs ablated)
    # Or binomial test if independent
    p = stats.binom_test(
        ablation_correct, n, baseline_acc,
        alternative='two-sided'
    )
    p_values.append(p)

# 3. Apply multiple comparisons correction
corrected = bonferroni_correct(p_values, alpha=0.05)
significant_heads = [
    (head, p, sig) 
    for head, p, sig in zip(all_heads, p_values, corrected['significant_after_correction'])
    if sig
]

# 4. Only report significant heads
print(f"Significant heads (after Bonferroni correction): {len(significant_heads)}/{len(all_heads)}")
for head, p, _ in significant_heads:
    print(f"  L{head['layer']}H{head['head']}: diff={head['diff']:+.0%}, p={p:.4f}")
```

---

## Conclusion

The framework shows **strong methodological awareness** but **inconsistent implementation**. The most critical issues are:

1. Multiple comparisons not corrected
2. Sample sizes below requirements
3. Missing statistical tests in circuit discovery

These can lead to **false discoveries** and **inflated claims**. However, the framework is well-designed to fix these issues - the tools exist, they just need to be used consistently.

**Recommendation**: Add a "strict mode" that enforces all requirements at runtime, preventing experiments from running if methodology is violated.

