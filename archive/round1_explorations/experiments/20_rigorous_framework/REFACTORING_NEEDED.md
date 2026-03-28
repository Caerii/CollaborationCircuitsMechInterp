# Steps That Need Library Refactoring

## Summary
Many steps have duplicate code instead of using the library. This document tracks what needs to be fixed.

## ✅ Already Using Library Properly
- **step1_baseline_tom.py**: ✅ REFACTORED - Uses `ChatExperimentRunner` + `HeuristicBaselines`
- **step5_head_ablation_sweep.py**: ✅ REFACTORED - Uses `ChatModeCircuitAnalyzer` with proper methodology
- **step32_chat_proper.py**: ✅ REFACTORED - Uses `ChatExperimentRunner`
- **step33_proper_retest.py**: ✅ REFACTORED - Uses `ChatExperimentRunner` for all tests
- **step35_real_circuit_hunt.py**: ✅ REFACTORED - Uses `ChatModeCircuitAnalyzer`, `generate_n_scenarios`, `bonferroni_correct`
- **step36_causal_patching.py**: ✅ REFACTORED - Uses `ResponseParser` + `ActivationPatcher`

## 🔴 Critical - Needs Immediate Refactoring

### step6_activation_patching.py
**Status:** ✅ REFACTORED
- Now uses `ActivationPatcher` from library for caching activations
- Uses completion mode (logit-based) intentionally for precise evaluation
- Removed duplicate `LayerPatcher` class
- Uses library's patching mechanics while keeping logit-based evaluation

### step12_scale_up.py
**Status:** ✅ REFACTORED
- Now uses `ChatExperimentRunner` for evaluation (chat mode - proper methodology)
- Uses `wilson_ci` and `cohens_h` from `core.cross_model`
- Uses `ChatModeCircuitAnalyzer` for ablation
- Removed custom functions that duplicated library functionality

## 🟡 Medium Priority

### step4_logit_lens.py
**Status:** ✅ REFACTORED (intentionally uses completion mode)
- Already uses `LogitLens` from library (correct!)
- Uses completion mode intentionally (logit lens requires completion mode)
- Manual prompt formatting is appropriate for logit lens analysis
- No changes needed - this is the correct approach for logit lens

### step20_higher_order_tom.py
**Problems:**
- Likely has custom test functions
- Should use `ChatExperimentRunner`

**Priority:** MEDIUM - Should be straightforward refactor

### step30_entity_types.py
**Problems:**
- Likely has custom test functions
- Should use `ChatExperimentRunner`

**Priority:** MEDIUM - Should be straightforward refactor

### step31_critical_baselines.py
**Problems:**
- Likely has custom baseline functions
- Should use `ChatExperimentRunner` + `HeuristicBaselines`

**Priority:** MEDIUM - Should be straightforward refactor

## 🟢 Low Priority (May Be Exploratory)

These steps may be intentionally exploratory or testing specific techniques:
- step4_logit_lens.py
- step7_fine_grained_analysis.py
- step8_multiagent_tom_heads.py
- step9_attention_patterns.py
- step10_multiagent_circuit_hunt.py
- step11_mlp_probing.py
- step12_scale_up.py
- step13_sae_feature_analysis.py
- step14_feature_steering.py
- step15_crosslayer_features.py
- step16_inhibitor_deep_dive.py
- step17_transcoder_analysis.py
- step18_mlp_computation.py
- step19_late_layer_steering.py
- step20_higher_order_tom.py
- step21_true_belief_investigation.py
- step22_transcoder_more_data.py
- step23_explicit_belief_scale.py
- step24_first_mention_circuit.py
- step25_attention_head_steering.py
- step26_multiagent_explicit.py
- step27_ablate_first_mention.py
- step28_comprehensive_summary.py
- step29_belief_probing.py
- step30_entity_types.py
- step31_critical_baselines.py
- step34_deep_reasoning_analysis.py

## Refactoring Template

When refactoring a step, follow this pattern:

```python
"""
Step N: Description

Uses library components for proper methodology.
"""

import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import ChatExperimentRunner
from scenarios.templates import generate_n_scenarios
from analysis.controls import accuracy_with_ci, bonferroni_correct

def main():
    print("STEP N: DESCRIPTION")
    
    config = ExperimentConfig()
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Use library!
    runner = ChatExperimentRunner(model, tokenizer, config)
    scenarios = generate_n_scenarios(n=config.min_samples_per_condition)
    
    # Run experiments
    results = runner.run_batch(scenarios)
    
    # Analyze
    acc = accuracy_with_ci([r.is_correct for r in results.results])
    
    # Save results
    # ...

if __name__ == "__main__":
    main()
```

## Benefits of Refactoring

1. **Consistency**: All steps use same methodology
2. **Maintainability**: Fix bugs once in library, not in 30 scripts
3. **Correctness**: Library has been tested and validated
4. **Performance**: Library has optimizations (caching, batching)
5. **Readability**: Scripts become much shorter and clearer

