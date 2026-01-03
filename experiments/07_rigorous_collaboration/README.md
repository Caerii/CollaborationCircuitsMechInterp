# Experiment 07: Rigorous Multi-Agent Collaboration Analysis

## Purpose

A **methodologically rigorous** test of whether the model encodes knowledge states (who knows what) in multi-agent conversations.

---

## Methodological Improvements Over Previous Experiments

| Issue | Previous | Now |
|-------|----------|-----|
| Sample size | 8-25 | **100** |
| Controls | None | **Permutation tests** |
| Multiple comparisons | Uncorrected | **Bonferroni correction** |
| Effect size | Not reported | **Cohen's d** |
| Confidence intervals | None | **95% bootstrap CI** |
| Varied phrasings | Same templates | **Multiple variations** |

---

## Experimental Design

### Conditions

| Condition | Description | Self Knows | Other Knows |
|-----------|-------------|------------|-------------|
| **Private** | User tells Self something confidentially | ✓ | ✗ |
| **Shared** | Information shared with everyone | ✓ | ✓ |

### Hypothesis

**H0 (null)**: Knowledge states are NOT encoded - classifier performs at chance  
**H1 (alternative)**: Knowledge states ARE encoded - classifier performs above chance

### Statistical Tests

1. **Stratified 5-fold cross-validation** for accuracy
2. **Permutation test** (100 iterations) for significance
3. **Bonferroni correction** for multiple comparisons
4. **Cohen's d** for effect size
5. **Bootstrap 95% CI** for accuracy

---

## Pipeline

```
Step 1: Generate Data
  → scenarios.json (100 conversations)

Step 2: Extract Activations  
  → activations.pt (model representations)

Step 3: Statistical Analysis
  → analysis_results.json (all statistics)

Step 4: Visualize
  → rigorous_analysis.png
```

---

## How to Run

```bash
cd experiments/07_rigorous_collaboration

# Run each step separately (easier to debug)
python scripts/step1_generate_data.py
python scripts/step2_extract_activations.py
python scripts/step3_analyze.py
python scripts/step4_visualize.py

# Or all at once
python scripts/run_all.py
```

---

## Interpreting Results

### Significance Thresholds

| Threshold | Meaning |
|-----------|---------|
| p < 0.05/7 = 0.007 | Significant after Bonferroni correction |
| p < 0.05 | Nominally significant (inflated false positive rate) |
| Selectivity > 0.1 | Meaningful above-chance performance |
| Cohen's d > 0.2 | Small effect |
| Cohen's d > 0.5 | Medium effect |

### Verdict Criteria

- **ROBUST**: Selectivity > 0.1 AND majority of layers significant (corrected)
- **WEAK**: Some signal but not robust across layers
- **NONE**: Results indistinguishable from chance

---

## Files

```
07_rigorous_collaboration/
├── README.md                     # This file
├── scripts/
│   ├── step1_generate_data.py    # Generate scenarios
│   ├── step2_extract_activations.py  # Extract from model
│   ├── step3_analyze.py          # Statistical analysis
│   ├── step4_visualize.py        # Create plots
│   └── run_all.py                # Run everything
├── data/
│   └── scenarios.json            # Generated scenarios
├── results/
│   ├── activations.pt            # Model activations
│   └── analysis_results.json     # Statistical results
└── figures/
    └── rigorous_analysis.png     # Main visualization
```
























