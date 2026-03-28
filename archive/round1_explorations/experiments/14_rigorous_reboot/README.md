# Experiment 14: Rigorous Reboot

## Critical Fixes Required

### 1. Sample Size Crisis
- Previous: N=12-36
- Required: N=200+ per condition
- Rationale: Need N >> d for meaningful probing

### 2. Proper Null Distributions
- Random vector cosines in 2560d
- Permutation tests with proper power
- Bootstrap CIs that aren't trivial

### 3. Fix Ablation Architecture
- Hook `self_attn` output, not layer output
- Access actual attention head outputs before projection

### 4. Behavioral Tests (Not Q&A)
- Predict agent's NEXT ACTION
- No explicit "Does B agree?" questions
- Sally-Anne style: Where will Sally look?

### 5. Unbalanced Design
- Varying ratios of conditions
- Not perfectly orthogonal by construction

## New Experiment Structure
1. `step1_generate_large_dataset.py` - 200+ samples
2. `step2_compute_null_distributions.py` - Statistical baselines
3. `step3_behavioral_tom_test.py` - Action prediction, not Q&A
4. `step4_proper_ablation.py` - Correct attention head hooks
5. `step5_rigorous_analysis.py` - With proper statistics























