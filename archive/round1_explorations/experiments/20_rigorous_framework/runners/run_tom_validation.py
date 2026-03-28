"""
Run ToM Validation with Proper Methodology

This is the gold-standard ToM test with all methodological requirements:
- n >= 50 per condition (200+ total scenarios)
- 8-scenario counterbalancing
- Novel names to break priors
- Heuristic baseline comparison
- Statistical validation
- 1000 token budget for reasoning
"""

import sys
from pathlib import Path
import json
import time

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT.parent.parent))

try:
    from ..config import ExperimentConfig, DEFAULT_CONFIG
    from ..core.chat_runner import ChatExperimentRunner, load_model_for_chat
    from ..scenarios.tom_extended import ToMScenarioGenerator
    from ..scenarios.counterbalancing import generate_counterbalanced_set, SALLY_ANNE_TEMPLATE
    from ..analysis.heuristics import HeuristicBaselines
    from ..analysis.validator import ResultValidator
except ImportError:
    # For direct execution
    sys.path.insert(0, str(FRAMEWORK_ROOT))
    from config import ExperimentConfig, DEFAULT_CONFIG
    from core.chat_runner import ChatExperimentRunner, load_model_for_chat
    from scenarios.tom_extended import ToMScenarioGenerator
    from scenarios.counterbalancing import generate_counterbalanced_set, SALLY_ANNE_TEMPLATE
    from analysis.heuristics import HeuristicBaselines
    from analysis.validator import ResultValidator


def run_tom_validation(
    config: ExperimentConfig = DEFAULT_CONFIG,
    n_tasks: int = 25,  # 25 tasks * 8 scenarios = 200 scenarios
    verbose: bool = True
):
    """
    Run complete ToM validation with proper methodology.
    
    Args:
        config: Experiment configuration
        n_tasks: Number of unique tasks (each generates 8 scenarios)
        verbose: Print progress
        
    Returns:
        Complete results dictionary
    """
    print("=" * 70)
    print("TOM VALIDATION WITH PROPER METHODOLOGY")
    print("=" * 70)
    
    start_time = time.time()
    
    # Validate configuration
    violations = config.validate_for_publication()
    if violations:
        print("\n[WARNING] Config violations:")
        for v in violations:
            print(f"  - {v}")
    
    # Generate scenarios
    print(f"\n[1/5] Generating {n_tasks * 8} counterbalanced scenarios...")
    scenarios = generate_counterbalanced_set(
        SALLY_ANNE_TEMPLATE,
        n_tasks=n_tasks,
        use_novel_names=config.require_novel_names,
    )
    print(f"  Generated {len(scenarios)} scenarios")
    print(f"  Types: {set(s['type'] for s in scenarios)}")
    
    # Load model
    print(f"\n[2/5] Loading model: {config.model_name}...")
    model, tokenizer = load_model_for_chat(
        config.model_name,
        dtype=config.dtype,
    )
    print("  Model loaded!")
    
    # Create runner
    runner = ChatExperimentRunner(model, tokenizer, config)
    
    # Run evaluation
    print(f"\n[3/5] Running evaluation (n={len(scenarios)})...")
    results = runner.run_batch(scenarios, verbose=verbose)
    
    # Compute heuristic baselines
    print("\n[4/5] Computing heuristic baselines...")
    baselines = HeuristicBaselines()
    model_predictions = [r.predicted_answer or "" for r in results.results]
    heuristic_eval = baselines.evaluate(scenarios, model_predictions)
    
    # Validate results
    print("\n[5/5] Validating methodology...")
    validator = ResultValidator(config)
    
    # Combine results for validation
    validation_input = {
        "n_total": results.n_total,
        "accuracy": results.accuracy,
        "n_options": 2,
        "by_type": results.by_type,
        "first_mention_accuracy": heuristic_eval["first_mention_accuracy"],
        "recency_accuracy": heuristic_eval["recency_accuracy"],
        "reality_accuracy": heuristic_eval["reality_accuracy"],
    }
    
    report = validator.validate_tom_results(validation_input)
    report.print_report()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results.accuracy:.1%} ({results.n_correct}/{results.n_total})")
    print(f"\nBy Type:")
    for stype, data in sorted(results.by_type.items()):
        print(f"  {stype}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    
    print(f"\nHeuristic Comparison:")
    print(f"  Model:         {results.accuracy:.1%}")
    print(f"  First-mention: {heuristic_eval['first_mention_accuracy']:.1%}")
    print(f"  Recency:       {heuristic_eval['recency_accuracy']:.1%}")
    print(f"  Reality:       {heuristic_eval['reality_accuracy']:.1%}")
    print(f"  Model beats heuristics: {heuristic_eval['model_beats_heuristics']}")
    
    print(f"\nTime: {elapsed:.1f}s ({elapsed/len(scenarios):.2f}s per scenario)")
    
    # Full results
    full_results = {
        "config": config.to_dict(),
        "n_scenarios": len(scenarios),
        "accuracy": results.accuracy,
        "by_type": results.by_type,
        "heuristic_comparison": heuristic_eval,
        "validation": report.to_dict(),
        "elapsed_seconds": elapsed,
    }
    
    return full_results


if __name__ == "__main__":
    # Run with default config
    results = run_tom_validation(
        n_tasks=25,  # 200 scenarios
        verbose=True
    )
    
    # Save results
    output_path = FRAMEWORK_ROOT / "results" / "tom_validation.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")

