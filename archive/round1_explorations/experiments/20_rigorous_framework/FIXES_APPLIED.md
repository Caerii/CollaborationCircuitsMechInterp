# Methodological Fixes Applied

## Summary

Fixed critical methodological issues in `step35_real_circuit_hunt.py` to ensure proper statistical rigor.

## Fixes Applied

### 1. ✅ Sample Size Enforcement
- **Before**: Used only 4 hardcoded scenarios
- **After**: Generates n≥50 scenarios using `generate_n_scenarios()` from scenario generator
- **Validation**: Warns if n < 50 (methodology requirement)
- **Implementation**: Uses `config.min_samples_per_condition` (default 50)

### 2. ✅ Statistical Tests Added
- **Before**: Only reported accuracy differences, no p-values
- **After**: Computes McNemar's test for each ablation (paired data: same scenarios, baseline vs ablated)
- **Output**: Reports p-value for each head ablation
- **Interpretation**: Marks significant results (p < 0.05)

### 3. ✅ Multiple Comparisons Correction
- **Before**: No correction for testing many heads simultaneously
- **After**: Applies Bonferroni correction to all p-values
- **Output**: 
  - Reports corrected alpha threshold
  - Marks which heads are significant after correction
  - Separates uncorrected vs corrected significance

### 4. ✅ Enhanced Scenario Handling
- **Before**: Hardcoded 4 scenarios with specific format
- **After**: 
  - Uses scenario generator for proper counterbalancing
  - Handles multiple scenario formats (story+question, question-only, legacy)
  - Filters to only FB/TB scenarios (excludes reality controls)
  - Counts scenarios by type

### 5. ✅ Improved Reporting
- **Before**: Simple list of "most helpful" heads
- **After**:
  - Reports significant heads (after correction) separately
  - Shows both uncorrected and corrected significance
  - Includes statistical details in saved JSON
  - Better visualization of results

## Code Changes

### Key Additions:
1. Import statistical functions:
   ```python
   from scipy import stats
   from analysis.controls import bonferroni_correct
   from scenarios.templates import generate_n_scenarios
   ```

2. Scenario generation:
   ```python
   scenarios = generate_n_scenarios(
       n=min_n,
       use_novel_names=config.require_novel_names,
       seed=42
   )
   ```

3. Statistical testing in ablation loop:
   ```python
   # McNemar's test for paired data
   both_correct = sum(...)
   baseline_only = sum(...)
   ablation_only = sum(...)
   both_wrong = sum(...)
   mcnemar_stat = (abs(baseline_only - ablation_only) - 1) ** 2 / (baseline_only + ablation_only)
   p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
   ```

4. Multiple comparisons correction:
   ```python
   correction_result = bonferroni_correct(p_values, alpha=0.05)
   corrected_alpha = correction_result['corrected_alpha']
   significant_after_correction = correction_result['significant_after_correction']
   ```

## Expected Output

The script now produces:
- Baseline accuracy with n≥50
- For each head ablation:
  - Accuracy and difference from baseline
  - **p-value** (McNemar's test)
  - Significance markers (* = uncorrected, *** = corrected)
- Summary of multiple comparisons correction
- **Only significant heads** (after correction) are highlighted as "Key ToM heads"

## Running the Fixed Script

```bash
cd experiments/20_rigorous_framework
python scripts/step35_real_circuit_hunt.py
```

**Note**: This will now take longer because:
- Tests n≥50 scenarios (instead of 4)
- Each scenario is tested twice (baseline + ablation)
- Total: ~50 scenarios × (1 baseline + ~20 ablations) = ~1050 model calls

## Remaining Considerations

1. **Runtime**: With n=50 and 20 head tests, expect ~30-60 minutes runtime
2. **Memory**: May need to reduce `heads_per_layer` if memory constrained
3. **Power**: With n=50, can detect medium effects (h≥0.5) with 80% power
4. **Correction**: Bonferroni is conservative - may find fewer significant heads, but they're more reliable

## Next Steps

1. Run the fixed script and verify it works
2. Check if any heads pass Bonferroni correction
3. If none pass, consider:
   - Larger sample size (n=100+)
   - More targeted layer selection
   - Less conservative correction (FDR instead of Bonferroni)

