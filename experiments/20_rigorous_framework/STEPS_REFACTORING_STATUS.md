# Steps Refactoring Status

## ✅ Fully Refactored (Using Library Properly)

These steps have been refactored to use library components:

1. **step1_baseline_tom.py** ✅
   - Uses `ChatExperimentRunner.run_batch()` for chat mode
   - Uses `HeuristicBaselines.evaluate()` for heuristic comparison
   - Still has completion mode test (intentional baseline)

2. **step5_head_ablation_sweep.py** ✅
   - Uses `ChatModeCircuitAnalyzer.ablation_sweep()`
   - Uses chat mode with proper methodology
   - Includes statistical tests and multiple comparisons correction

3. **step32_chat_proper.py** ✅
   - Uses `ChatExperimentRunner.run_scenario()`
   - Uses `ResponseParser` automatically

4. **step33_proper_retest.py** ✅
   - Uses `ChatExperimentRunner.run_scenario()` for all tests
   - Removed custom `test_chat()` function

5. **step35_real_circuit_hunt.py** ✅
   - Uses `ChatModeCircuitAnalyzer`
   - Uses `generate_n_scenarios`
   - Uses `bonferroni_correct`

6. **step36_causal_patching.py** ✅
   - Uses `ResponseParser` for answer extraction
   - Uses `ActivationPatcher` (already in library)

7. **step6_activation_patching.py** ✅
   - Uses `ActivationPatcher` from library for caching activations
   - Uses completion mode (logit-based) intentionally for precise evaluation
   - Removed duplicate `LayerPatcher` class

8. **step12_scale_up.py** ✅
   - Uses `ChatExperimentRunner` for evaluation (chat mode - proper methodology)
   - Uses `wilson_ci` and `cohens_h` from `core.cross_model`
   - Uses `ChatModeCircuitAnalyzer` for ablation
   - Removed custom functions that duplicated library functionality

9. **step4_logit_lens.py** ✅
   - Already uses `LogitLens` from library (correct!)
   - Uses completion mode intentionally (logit lens requires completion mode)
   - No changes needed - this is the correct approach for logit lens

---

## 🔴 High Priority - Should Be Refactored

These steps have significant duplicate code and should use the library:

---

## 🟡 Medium Priority - Could Be Refactored

These steps may have some duplicate code but might be intentionally exploratory:

### step7_fine_grained_analysis.py
- Uses specialized tools (`SignalExtractor`, `MLPAnalyzer`, `HeadAmplifier`)
- May be intentionally custom for specific analysis
- **Check:** Do these tools exist in library?

### step8_multiagent_tom_heads.py
- Multi-agent specific code
- May need custom handling
- **Check:** Does `MultiAgentInteraction` cover this?

### step10_multiagent_circuit_hunt.py
- Multi-agent circuit discovery
- **Check:** Can this use `ChatModeCircuitAnalyzer`?

### step11_mlp_probing.py
- MLP-specific probing
- **Check:** Does `ProbingPipeline` cover this?

### step13_sae_feature_analysis.py
- SAE-specific analysis
- May be intentionally custom

### step14_feature_steering.py
- Feature steering experiments
- **Check:** Does `CausalSteering` cover this?

### step15_crosslayer_features.py
- Cross-layer feature analysis
- May be intentionally custom

### step16_inhibitor_deep_dive.py
- Deep dive on inhibitory heads
- Could potentially use `ChatModeCircuitAnalyzer`

### step17_transcoder_analysis.py
- Transcoder-specific analysis
- May be intentionally custom

### step18_mlp_computation.py
- MLP computation analysis
- May be intentionally custom

### step19_late_layer_steering.py
- Late layer steering
- **Check:** Does `CausalSteering` cover this?

### step20_higher_order_tom.py
- Higher-order ToM tests
- Could use `ChatExperimentRunner`

### step21_true_belief_investigation.py
- True belief specific investigation
- Could use `ChatExperimentRunner`

### step22_transcoder_more_data.py
- More transcoder analysis
- May be intentionally custom

### step23_explicit_belief_scale.py
- Explicit belief scale tests
- Could use `ChatExperimentRunner`

### step24_first_mention_circuit.py
- First mention circuit discovery
- Could use `ChatModeCircuitAnalyzer`

### step25_attention_head_steering.py
- Attention head steering
- **Check:** Does `CausalSteering` cover this?

### step26_multiagent_explicit.py
- Multi-agent explicit tests
- Could use `ChatExperimentRunner` or `MultiAgentInteraction`

### step27_ablate_first_mention.py
- Ablation of first mention
- Could use `ChatModeCircuitAnalyzer`

### step28_comprehensive_summary.py
- Summary script (may not need refactoring)
- Just aggregates results

### step29_belief_probing.py
- Belief probing
- **Check:** Does `ProbingPipeline` cover this?

### step30_entity_types.py
- Entity type tests
- Could use `ChatExperimentRunner`

### step31_critical_baselines.py
- Baseline tests
- Could use `ChatExperimentRunner` and `HeuristicBaselines`

### step34_deep_reasoning_analysis.py
- Deep reasoning analysis
- Could use `ChatExperimentRunner`

---

## Summary

### Refactored: 9 steps ✅
- step1, step4, step5, step6, step12, step32, step33, step35, step36

### High Priority (Should Refactor): 0 steps 🔴
- All high-priority steps have been refactored!

### Medium Priority (Could Refactor): 25 steps 🟡
- step7, step8, step10, step11, step13, step14, step15, step16, step17, step18, step19, step20, step21, step22, step23, step24, step25, step26, step27, step28, step29, step30, step31, step34

### Total Steps: 34
### Refactored: 9 (26%)
### Need Refactoring: 25 (74%)

---

## Quick Wins (Easiest to Refactor)

1. **step12_scale_up.py** - Just replace custom functions with library equivalents
2. **step6_activation_patching.py** - Replace `LayerPatcher` with `ActivationPatcher`
3. **step4_logit_lens.py** - Just add `ChatExperimentRunner` for consistency
4. **step20_higher_order_tom.py** - Likely just needs `ChatExperimentRunner`
5. **step30_entity_types.py** - Likely just needs `ChatExperimentRunner`
6. **step31_critical_baselines.py** - Likely just needs `ChatExperimentRunner` + `HeuristicBaselines`

