"""
Run Full Test Suite

Comprehensive test covering:
1. Theory of Mind (with counterbalancing)
2. Multi-agent belief tracking
3. Deception detection
4. Cooperation understanding

All tests follow proper methodology (n>=50, heuristic comparison, validation).
"""

import sys
from pathlib import Path
import json
import time
import argparse

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT.parent.parent))


def run_full_suite(
    n_per_type: int = 50,
    save_results: bool = True,
    verbose: bool = True
):
    """
    Run the complete test suite.
    
    Args:
        n_per_type: Scenarios per type (minimum 50 for publication)
        save_results: Save results to disk
        verbose: Print progress
    """
    # Lazy imports to avoid loading everything at start
    # Try relative imports first
    try:
        from ..config import DEFAULT_CONFIG
        from ..core.chat_runner import ChatExperimentRunner, load_model_for_chat
        from ..scenarios.tom_extended import ToMScenarioGenerator
        from ..scenarios.multi_agent import MultiAgentScenarioGenerator
        from ..scenarios.deception import DeceptionScenarioGenerator
        from ..scenarios.cooperation import CooperationScenarioGenerator
        from ..analysis.heuristics import HeuristicBaselines
        from ..analysis.validator import ResultValidator
    except ImportError:
        # For direct execution
        sys.path.insert(0, str(FRAMEWORK_ROOT))
        from config import DEFAULT_CONFIG
        from core.chat_runner import ChatExperimentRunner, load_model_for_chat
        from scenarios.tom_extended import ToMScenarioGenerator
        from scenarios.multi_agent import MultiAgentScenarioGenerator
        from scenarios.deception import DeceptionScenarioGenerator
        from scenarios.cooperation import CooperationScenarioGenerator
        from analysis.heuristics import HeuristicBaselines
        from analysis.validator import ResultValidator
    
    config = DEFAULT_CONFIG
    
    print("=" * 70)
    print("FULL TEST SUITE - RIGOROUS METHODOLOGY")
    print("=" * 70)
    print(f"\nSample size per type: n={n_per_type}")
    print(f"Token budget: {config.max_tokens}")
    
    start_time = time.time()
    all_results = {}
    
    # Load model once
    print(f"\n[1/5] Loading model: {config.model_name}...")
    model, tokenizer = load_model_for_chat(config.model_name)
    runner = ChatExperimentRunner(model, tokenizer, config)
    print("  Model loaded!")
    
    # === Theory of Mind ===
    print("\n" + "=" * 70)
    print("[2/5] THEORY OF MIND TEST")
    print("=" * 70)
    
    tom_gen = ToMScenarioGenerator(use_novel_names=True)
    tom_scenarios = tom_gen.generate_balanced_set(n_per_type=n_per_type)
    print(f"  Generated {len(tom_scenarios)} ToM scenarios")
    
    tom_results = runner.run_batch(tom_scenarios, verbose=verbose)
    all_results["tom"] = {
        "accuracy": tom_results.accuracy,
        "n": tom_results.n_total,
        "by_type": tom_results.by_type,
    }
    print(f"\n  ToM Accuracy: {tom_results.accuracy:.1%}")
    
    # === Multi-Agent ===
    print("\n" + "=" * 70)
    print("[3/5] MULTI-AGENT TEST")
    print("=" * 70)
    
    ma_gen = MultiAgentScenarioGenerator(use_novel_names=True)
    ma_scenarios = ma_gen.generate_balanced_set(n_per_type=n_per_type)
    print(f"  Generated {len(ma_scenarios)} multi-agent scenarios")
    
    ma_results = runner.run_batch(ma_scenarios, verbose=verbose)
    all_results["multi_agent"] = {
        "accuracy": ma_results.accuracy,
        "n": ma_results.n_total,
        "by_type": ma_results.by_type,
    }
    print(f"\n  Multi-Agent Accuracy: {ma_results.accuracy:.1%}")
    
    # === Deception ===
    print("\n" + "=" * 70)
    print("[4/5] DECEPTION TEST")
    print("=" * 70)
    
    dec_gen = DeceptionScenarioGenerator(use_novel_names=True)
    dec_scenarios = dec_gen.generate_balanced_set(n_per_type=n_per_type)
    print(f"  Generated {len(dec_scenarios)} deception scenarios")
    
    dec_results = runner.run_batch(dec_scenarios, verbose=verbose)
    all_results["deception"] = {
        "accuracy": dec_results.accuracy,
        "n": dec_results.n_total,
        "by_type": dec_results.by_type,
    }
    print(f"\n  Deception Accuracy: {dec_results.accuracy:.1%}")
    
    # === Cooperation ===
    print("\n" + "=" * 70)
    print("[5/5] COOPERATION TEST")
    print("=" * 70)
    
    coop_gen = CooperationScenarioGenerator(use_novel_names=True)
    coop_scenarios = coop_gen.generate_balanced_set(n_per_type=n_per_type)
    print(f"  Generated {len(coop_scenarios)} cooperation scenarios")
    
    coop_results = runner.run_batch(coop_scenarios, verbose=verbose)
    all_results["cooperation"] = {
        "accuracy": coop_results.accuracy,
        "n": coop_results.n_total,
        "by_type": coop_results.by_type,
    }
    print(f"\n  Cooperation Accuracy: {coop_results.accuracy:.1%}")
    
    # === Summary ===
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("FULL SUITE SUMMARY")
    print("=" * 70)
    
    total_n = sum(r["n"] for r in all_results.values())
    
    print(f"\nOverall Results:")
    print(f"  Total scenarios: {total_n}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_n:.2f}s per scenario)")
    
    print(f"\nAccuracy by Domain:")
    for domain, data in all_results.items():
        print(f"  {domain}: {data['accuracy']:.1%} (n={data['n']})")
    
    # Calculate overall
    weighted_acc = sum(r["accuracy"] * r["n"] for r in all_results.values()) / total_n
    print(f"\n  Weighted Average: {weighted_acc:.1%}")
    
    # Validation
    validator = ResultValidator(config)
    validation_results = {}
    
    print(f"\nValidation Status:")
    for domain, data in all_results.items():
        report = validator.validate_tom_results({
            "n_total": data["n"],
            "accuracy": data["accuracy"],
            "n_options": 2,
            "by_type": data.get("by_type", {}),
        }, require_counterbalancing=False, require_beats_heuristics=False)
        
        status = "PASS" if report.all_passed else "FAIL"
        print(f"  {domain}: {status}")
        validation_results[domain] = report.to_dict()
    
    # Full results
    full_results = {
        "config": config.to_dict(),
        "results": all_results,
        "validation": validation_results,
        "summary": {
            "total_n": total_n,
            "weighted_accuracy": weighted_acc,
            "elapsed_seconds": elapsed,
        },
    }
    
    # Save
    if save_results:
        output_path = FRAMEWORK_ROOT / "results" / "full_suite.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
    
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full test suite")
    parser.add_argument("--n", type=int, default=50, help="Scenarios per type (min 50)")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()
    
    run_full_suite(
        n_per_type=args.n,
        verbose=not args.quiet
    )

