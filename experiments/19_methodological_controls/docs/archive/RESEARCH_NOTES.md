# Experiment 19: Research Notes & Next Steps

## Summary of What We've Done

### Complete Systematic Head Search (step8d)
- **Coverage**: ALL 1152 heads (36 layers × 32 heads)
- **Method**: Ablate each head individually, measure belief update accuracy
- **Phase 1**: N=15 scenarios (fast scan), baseline 40%
- **Phase 2**: N=30 scenarios (validation), baseline 13.3%

### Key Findings

#### 1. Inhibitory Network Location (Layers 14-21)
```
Layer Distribution of Inhibitors (>10% boost when ablated):
  L14: ███████ (7 heads)
  L15: ███████ (7 heads)
  L17: ████ (4 heads)
  L18: ██████ (6 heads)  ← STRONGEST
  L19: ██████ (6 heads)  ← STRONGEST
  L20: ████ (4 heads)
  L21: █ (1 head)
  
Almost NO inhibitors in layers 0-13 or 22-35!
```

#### 2. Top Validated Inhibitors
| Rank | Head | Phase 2 Boost | Consistency |
|------|------|---------------|-------------|
| 1 | L18H11 | +43.3% | CONSISTENT |
| 2 | L17H4 | +43.3% | CONSISTENT |
| 3 | L18H14 | +43.3% | CONSISTENT |
| 4 | L21H17 | +43.3% | varies |
| 5 | L19H30 | +40.0% | CONSISTENT |

#### 3. Critical Enabling Heads (ablation → 0% accuracy)
| Head | Function |
|------|----------|
| L15H9 | ESSENTIAL - pre-ToM processing |
| L19H2 | ESSENTIAL - post-inhibitor processing |
| L19H15 | ESSENTIAL - post-inhibitor processing |

---

## Methodological Issues Identified

### Issue 1: Baseline Variance
- Phase 1 baseline: 40%
- Phase 2 baseline: 13.3%
- **This is a 27% difference!**

**Cause**: Different random seeds produce different scenarios
**Implication**: Results are sensitive to specific scenario wording
**Solution**: Use FIXED scenario set for all comparisons, or much larger N

### Issue 2: Single Ablation Limitation
- We tested each head in isolation
- Heads may have **redundant** or **compensatory** functions
- The network may exhibit **self-repair**

**Next Step**: Test combined ablations of top inhibitors (we did this earlier with OLD top inhibitors, need to redo with NEW ones)

### Issue 3: No Path Patching
- We know WHICH heads matter, but not HOW they communicate
- Information flow between L15H9 → L17-18 inhibitors → L19 enablers is unknown

**Standard Approach** (from literature):
1. **Activation patching**: Replace activations from one run with another
2. **Path patching**: Trace specific pathways through the network
3. **Causal tracing**: Use multiple "corrupted" vs "clean" runs

### Issue 4: No Attention Pattern Analysis
- We don't know WHAT the inhibitory heads attend to
- Do they attend to original location? Agent name? Communicative verb?

**From Zhu et al. 2024**: They analyzed attention to belief-relevant tokens
**Our gap**: We haven't done this systematically

### Issue 5: Model-Specific Findings
- All findings are on Qwen3-4B only
- Unknown if this generalizes to other models

---

## What The Literature Tells Us

### From Zhu et al. 2024 (Belief State Encoding)
- **Method**: Linear probing on residual stream at each layer
- **Key finding**: Belief states are linearly decodable from mid-to-late layers
- **Relevance**: Our inhibitory zone (L14-21) aligns with where they found belief encoding

### From Lee et al. 2025 (Multi-Agent MI Research Agenda)
- **Method recommendation**: Trace attention heads mediating inter-agent communication
- **Key insight**: Toxic agreement has identifiable circuit origins
- **Gap in our work**: We found inhibitors but haven't traced their mechanism

### From ARENA Tutorial (Path Patching)
- **Standard approach**: 
  1. Run model on "clean" input (correct answer expected)
  2. Run model on "corrupted" input (wrong answer expected)
  3. Patch activations from corrupted → clean at specific components
  4. Measure how much patching "breaks" the clean output
  
- **We haven't done this**: Our ablation is "zero out" not "patch from alternative"

### From Li et al. 2023 (Multi-Agent ToM)
- **Key finding**: Explicit belief representation improves multi-agent performance
- **Relevance**: Our "bridging phrases" work because they make belief states explicit
- **Connection**: The inhibitory network may be blocking implicit → explicit belief conversion

---

## Proper Next Steps (Prioritized)

### HIGH PRIORITY

#### 1. Retest Combined Ablation with CORRECT Top Inhibitors
**Why**: Our earlier combined ablation used OLD top inhibitors (L17H4, L15H12, L24H29)
**New top 3**: L18H11, L17H4, L18H14 (from complete search)
**Expected**: Should achieve even higher accuracy than 92.5%

```python
# Test this combination
NEW_TOP_INHIBITORS = [(18, 11), (17, 4), (18, 14)]
```

#### 2. Attention Pattern Analysis on L18H11
**Why**: It's the strongest inhibitor, we need to know what it looks at
**Method**: 
- Get attention weights for L18H11 on ToM prompts
- Identify which tokens get highest attention FROM the critical final tokens
- Hypothesis: It attends to original location, anchoring belief there

#### 3. Test Enabling Head Amplification
**Why**: We've ablated inhibitors; can we BOOST enablers instead?
**Method**: Multiply L15H9, L19H2, L19H15 output by 1.5x, 2x
**Expected**: If these are "ToM processors", amplifying should improve accuracy

#### 4. Fix Baseline Variance Issue
**Why**: 40% vs 13.3% baseline makes comparisons unreliable
**Method**: 
- Create FIXED scenario set (N=100) saved to file
- Use identical scenarios for ALL future experiments
- Report confidence intervals

### MEDIUM PRIORITY

#### 5. Path Patching: L15H9 → L17-18 → L19
**Why**: Understand information flow through the ToM zone
**Method**:
- Run clean (belief update expected) and corrupted (no update expected)
- Patch L15H9 output from corrupted → clean
- Measure if this breaks belief update
- Repeat for L17H4, L18H11, etc.

#### 6. Cross-Model Validation
**Why**: Are these circuits universal or Qwen-specific?
**Method**: Run complete head search on Llama-3-8B or Mistral-7B
**Challenge**: Need enough VRAM or use NDIF

#### 7. Scenario Feature Analysis
**Why**: Understand what makes some scenarios "hard" vs "easy"
**Method**:
- Log all scenario features (word order, agent names, object types)
- Correlate with success/failure
- Identify systematic biases

### LOWER PRIORITY (Future Work)

#### 8. Linear Probing for Belief States
**Why**: Zhu et al. method - directly decode belief from activations
**Method**: Train linear probe at each layer to predict agent belief
**Expected**: Should show strongest encoding in L15-19 zone

#### 9. SAE Feature Analysis
**Why**: Find interpretable features in inhibitory heads
**Method**: Use SAELens/Neuronpedia to identify ToM-related features
**Challenge**: May not have pre-trained SAEs for Qwen3-4B

---

## Key Open Questions

1. **Why do inhibitors exist?**
   - Training artifact? Overgeneralization prevention? Statistical bias?
   
2. **Is the inhibitory network learned or architectural?**
   - Does fine-tuning change it? Does RLHF affect it?
   
3. **Do enablers and inhibitors "compete" or "gate"?**
   - Is it a voting mechanism? Or sequential processing?
   
4. **Does this generalize to real multi-agent scenarios?**
   - Our test is synthetic Sally-Anne; real collaboration may differ

---

## Experimental Rigor Checklist

For any next experiment, ensure:

- [ ] **Fixed scenario set** (not random each run)
- [ ] **Sufficient N** (minimum 50, ideally 100)
- [ ] **Validation with different seed** (Phase 1 + Phase 2 approach)
- [ ] **Statistical significance** (report p-values or confidence intervals)
- [ ] **Comparison to baseline** (always report baseline accuracy)
- [ ] **Document all parameters** (model, seed, N, scenario structure)

---

## Files in This Experiment

| Script | Purpose | Status |
|--------|---------|--------|
| step1_comprehensive_controls.py | Test prompting interventions | COMPLETE |
| step2_what_specifically_helps.py | Decompose what helps | COMPLETE |
| step3_attention_analysis.py | Compare attention patterns | COMPLETE |
| step4_ablate_update_circuit.py | Test update circuit ablation | COMPLETE |
| step5_systematic_head_search.py | Partial head search | OBSOLETE (replaced by step8d) |
| step6_combined_ablation.py | Combined head ablation | NEEDS REDO with new top inhibitors |
| step7_inhibitory_head_discovery.py | L24H29 discovery | COMPLETE |
| step8_extreme_conditions.py | Amplification tests | COMPLETE |
| step8b_inhibitory_search.py | Partial search | OBSOLETE (replaced by step8d) |
| step8c_combined_inhibitor_ablation.py | Combined OLD top inhibitors | NEEDS REDO |
| step8d_complete_head_search.py | FULL systematic search | COMPLETE - CANONICAL |

---

## Next Script to Write

**step9_validated_combined_ablation.py**

Purpose: Test combined ablation with the CORRECT top inhibitors from complete search

```python
# Key test conditions:
1. Baseline (no ablation)
2. L18H11 alone (strongest)
3. L18H11 + L17H4
4. L18H11 + L17H4 + L18H14 (new top 3)
5. All top 5 validated heads
6. Also test AMPLIFYING L15H9 (critical enabler)

# Use FIXED scenario set of N=100
# Report confidence intervals
```

---

*Last Updated: Session of complete head search and analysis*

