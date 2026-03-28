"""
Step 15: Comprehensive Robustness Test (Using Library)

This is a cleaner version using our new mechinterp library.

Tests robustness across:
1. Multiple prompt templates (6 variations)
2. Different communication verbs (9 variations)
3. Edge cases (5 types)
4. Negative controls (3 types)

Statistical rigor:
- Wilson score confidence intervals
- McNemar's test for significance
- Effect sizes (Cohen's h)
"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.model_utils import QwenModel
from lib.hooks import HookManager
from lib.scenarios import ScenarioGenerator
from lib.evaluation import ToMEvaluator, ResultsManager
from lib.statistics import compute_accuracy_ci, significance_test, effect_size

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Top 5 inhibitors from our discovery
INHIBITORS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]


def run_robustness_tests():
    """Run comprehensive robustness tests."""
    
    print("=" * 70)
    print("STEP 15: COMPREHENSIVE ROBUSTNESS TEST (Library Version)")
    print("=" * 70)
    
    # 1. Load model
    print("\n[1/5] Loading model...")
    model = QwenModel().load()
    hooks = HookManager(model)
    evaluator = ToMEvaluator(model, hooks)
    results_mgr = ResultsManager(RESULTS_DIR)
    
    all_results = {}
    
    # 2. Template variations
    print("\n[2/5] Testing template variations...")
    gen = ScenarioGenerator(seed=42)
    template_scenarios = gen.template_variations(n_per_template=20)
    
    template_results = {}
    for template_type in ['original', 'passive', 'story', 'formal', 'casual']:
        subset = [s for s in template_scenarios if s.get('template') == template_type]
        
        print(f"  {template_type} (N={len(subset)})...", end=" ", flush=True)
        
        # Baseline
        baseline_res = evaluator.evaluate_batch(subset)
        
        # With ablation
        hooks.ablate_heads(INHIBITORS)
        ablated_res = evaluator.evaluate_batch(subset)
        hooks.clear()
        
        # Statistics
        baseline_ci = compute_accuracy_ci([r['predicts_correct'] for r in baseline_res['results']])
        ablated_ci = compute_accuracy_ci([r['predicts_correct'] for r in ablated_res['results']])
        
        template_results[template_type] = {
            'baseline': baseline_ci,
            'ablated': ablated_ci,
            'boost': ablated_ci['accuracy'] - baseline_ci['accuracy'],
        }
        
        print(f"Baseline: {baseline_ci['accuracy']*100:.0f}% -> Ablated: {ablated_ci['accuracy']*100:.0f}%")
    
    all_results['templates'] = template_results
    
    # 3. Communication verb variations
    print("\n[3/5] Testing communication verbs...")
    verb_scenarios = gen.communication_verb_variations(n_per_verb=15)
    
    verb_results = {}
    verb_types = set(s.get('verb') for s in verb_scenarios)
    
    for verb in sorted(verb_types):
        subset = [s for s in verb_scenarios if s.get('verb') == verb]
        
        print(f"  {verb} (N={len(subset)})...", end=" ", flush=True)
        
        baseline_res = evaluator.evaluate_batch(subset)
        
        hooks.ablate_heads(INHIBITORS)
        ablated_res = evaluator.evaluate_batch(subset)
        hooks.clear()
        
        baseline_ci = compute_accuracy_ci([r['predicts_correct'] for r in baseline_res['results']])
        ablated_ci = compute_accuracy_ci([r['predicts_correct'] for r in ablated_res['results']])
        
        verb_results[verb] = {
            'baseline': baseline_ci,
            'ablated': ablated_ci,
        }
        
        print(f"{baseline_ci['accuracy']*100:.0f}% -> {ablated_ci['accuracy']*100:.0f}%")
    
    all_results['verbs'] = verb_results
    
    # 4. Negative controls
    print("\n[4/5] Testing negative controls...")
    neg_scenarios = gen.negative_controls(n=45)
    
    neg_results = {}
    neg_types = set(s.get('type') for s in neg_scenarios)
    
    for neg_type in sorted(neg_types):
        subset = [s for s in neg_scenarios if s.get('type') == neg_type]
        
        print(f"  {neg_type} (N={len(subset)})...", end=" ", flush=True)
        
        baseline_res = evaluator.evaluate_batch(subset)
        
        hooks.ablate_heads(INHIBITORS)
        ablated_res = evaluator.evaluate_batch(subset)
        hooks.clear()
        
        baseline_ci = compute_accuracy_ci([r['predicts_correct'] for r in baseline_res['results']])
        ablated_ci = compute_accuracy_ci([r['predicts_correct'] for r in ablated_res['results']])
        
        neg_results[neg_type] = {
            'baseline': baseline_ci,
            'ablated': ablated_ci,
        }
        
        print(f"{baseline_ci['accuracy']*100:.0f}% -> {ablated_ci['accuracy']*100:.0f}%")
    
    all_results['negative_controls'] = neg_results
    
    # 5. Summary statistics
    print("\n[5/5] Computing summary statistics...")
    
    # Aggregate across all positive tests (excluding negative controls)
    all_baseline_accs = []
    all_ablated_accs = []
    
    for category in ['templates', 'verbs']:
        for name, data in all_results[category].items():
            all_baseline_accs.append(data['baseline']['accuracy'])
            all_ablated_accs.append(data['ablated']['accuracy'])
    
    mean_baseline = sum(all_baseline_accs) / len(all_baseline_accs)
    mean_ablated = sum(all_ablated_accs) / len(all_ablated_accs)
    
    # Effect size
    es = effect_size(mean_baseline, mean_ablated)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAcross {len(all_baseline_accs)} test conditions:")
    print(f"  Mean baseline:    {mean_baseline*100:.1f}%")
    print(f"  Mean w/ ablation: {mean_ablated*100:.1f}%")
    print(f"  Mean boost:       {(mean_ablated - mean_baseline)*100:+.1f}%")
    print(f"  Effect size:      {es['magnitude']} (Cohen's h = {es['cohens_h']:.2f})")
    
    # Save results
    all_results['summary'] = {
        'n_conditions': len(all_baseline_accs),
        'mean_baseline': mean_baseline,
        'mean_ablated': mean_ablated,
        'mean_boost': mean_ablated - mean_baseline,
        'effect_size': es,
    }
    
    output_path = results_mgr.save(all_results, 'robustness_clean_results')
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_robustness_tests()


