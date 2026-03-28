"""
Step 35: Real Circuit Hunt (Chat Mode) - FIXED VERSION

NOW that we know proper methodology, find the ACTUAL ToM circuit.

Previous findings (L28H5, L18H16, etc.) were under flawed conditions.
This script:
1. Ablates heads in CHAT MODE with proper tokens
2. Tests on both FB and TB scenarios
3. Identifies heads that CAUSALLY affect ToM performance
4. Uses proper statistical tests and multiple comparisons correction

OUTPUT: results/step35_circuit.json, figures/step35_circuit.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from scenarios.templates import generate_n_scenarios
from analysis.circuits import ChatModeCircuitAnalyzer
from analysis.controls import bonferroni_correct

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 35: REAL CIRCUIT HUNT (CHAT MODE) - FIXED")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nFinding the ACTUAL ToM circuit with proper methodology!")
    print("✅ Using correct head ablation (pre-hook on o_proj input)")
    print("✅ Using library components (ChatModeCircuitAnalyzer)")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded! {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    sys.stdout.flush()
    
    # Initialize analyzer (uses library!)
    analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
    
    # Generate scenarios with proper methodology
    print("\nGenerating scenarios...")
    sys.stdout.flush()
    
    # Use scenario generator to get n≥50 scenarios
    min_n = config.min_samples_per_condition
    print(f"  Target: n≥{min_n} scenarios (methodology requirement)")
    sys.stdout.flush()
    
    # Generate balanced set of FB and TB scenarios
    scenarios = generate_n_scenarios(
        n=min_n,
        use_novel_names=config.require_novel_names,
        seed=42  # For reproducibility
    )
    
    # Filter to only FB and TB scenarios (exclude communication and reality control scenarios)
    filtered_scenarios = []
    for s in scenarios:
        s_type = s.get('type', s.get('scenario_type', ''))
        # Include false_belief and true_belief, exclude reality_control and communication
        if s_type in ['false_belief', 'true_belief']:
            filtered_scenarios.append(s)
    
    # If we don't have enough, generate more
    while len(filtered_scenarios) < min_n:
        additional = generate_n_scenarios(n=min_n, use_novel_names=config.require_novel_names, seed=None)
        for s in additional:
            s_type = s.get('type', s.get('scenario_type', ''))
            if s_type in ['false_belief', 'true_belief']:
                filtered_scenarios.append(s)
                if len(filtered_scenarios) >= min_n:
                    break
    
    scenarios = filtered_scenarios[:min_n]
    
    # Count by type
    fb_count = sum(1 for s in scenarios if s.get('type', s.get('scenario_type', '')) == 'false_belief')
    tb_count = sum(1 for s in scenarios if s.get('type', s.get('scenario_type', '')) == 'true_belief')
    
    print(f"  Generated: {len(scenarios)} scenarios ({fb_count} FB, {tb_count} TB)")
    sys.stdout.flush()
    
    # Focus on late layers (where decisions form)
    n_layers = model.config.num_hidden_layers
    layers_to_test = [l for l in [20, 24, 28, 32, 36] if l < n_layers]
    
    print(f"\nTesting layers: {layers_to_test}")
    print(f"Scenarios: {len(scenarios)} (n≥{min_n} required by methodology)")
    sys.stdout.flush()
    
    # Run ablation sweep (uses library - correct implementation!)
    print(f"\n{'='*60}")
    print("ABLATION SWEEP")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    results = analyzer.ablation_sweep(
        scenarios=scenarios,
        layers_to_test=layers_to_test,
        heads_per_layer=4,
        max_tokens=250  # Slightly shorter to save time
    )
    
    # Get significant heads with multiple comparisons correction
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    all_heads, correction_result = analyzer.get_significant_heads(
        results,
        alpha=0.05,
        correction="bonferroni"
    )
    
    corrected_alpha = correction_result['corrected_alpha']
    significant_heads = [h for h in all_heads if h['significant']]
    
    print(f"\nMultiple comparisons correction:")
    print(f"  Tests performed: {len(all_heads)}")
    print(f"  Original alpha: 0.05")
    print(f"  Corrected alpha (Bonferroni): {corrected_alpha:.4f}")
    print(f"  Significant (uncorrected): {sum(1 for h in all_heads if h['significant_uncorrected'])}")
    print(f"  Significant (corrected): {len(significant_heads)}")
    sys.stdout.flush()
    
    # Sort by effect
    all_heads.sort(key=lambda x: x['effect'])
    helpful_heads = [h for h in all_heads if h['effect'] > 0]
    inhibitory_heads = [h for h in all_heads if h['effect'] < 0]
    
    print("\n" + "="*60)
    print("SIGNIFICANT HEADS (after Bonferroni correction):")
    print("="*60)
    if significant_heads:
        for h in significant_heads[:10]:
            direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
            print(f"  L{h['layer']}H{h['head']} ({direction}): effect={h['effect']:+.1%}, p={h['p_value']:.4f}, acc={h['accuracy']:.1%}")
    else:
        print("  None (no heads pass Bonferroni correction)")
    
    print("\n" + "="*60)
    print("MOST HELPFUL HEADS (ablation hurts performance):")
    print("="*60)
    for h in helpful_heads[:5]:
        sig_marker = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: effect={h['effect']:+.1%}, acc={h['accuracy']:.1%}, p={h['p_value']:.4f}{sig_marker}")
    
    print("\n" + "="*60)
    print("MOST INHIBITORY HEADS (ablation helps performance):")
    print("="*60)
    for h in inhibitory_heads[-5:]:
        sig_marker = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: effect={h['effect']:+.1%}, acc={h['accuracy']:.1%}, p={h['p_value']:.4f}{sig_marker}")
    
    print("\nLegend: *** = significant after correction, * = significant uncorrected")
    sys.stdout.flush()
    
    # Visualize
    print("\nGenerating visualization...")
    sys.stdout.flush()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Heatmap of ablation effects
    ax1 = axes[0]
    layers = sorted(results['ablations'].keys())
    heads = sorted(list(results['ablations'][layers[0]].keys()))
    
    grid = np.zeros((len(layers), len(heads)))
    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            if head in results['ablations'][layer]:
                grid[i, j] = results['ablations'][layer][head]['effect']
    
    im = ax1.imshow(grid, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax1.set_xticks(range(len(heads)))
    ax1.set_xticklabels([f"H{h}" for h in heads])
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels([f"L{l}" for l in layers])
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Ablation Effect on ToM (Blue=Helpful, Red=Inhibitory)")
    plt.colorbar(im, ax=ax1, label="Accuracy Change")
    
    # Plot 2: Bar chart of most impactful
    ax2 = axes[1]
    top_10 = helpful_heads[:5] + inhibitory_heads[-5:]
    names = [f"L{h['layer']}H{h['head']}" for h in top_10]
    effects = [h['effect'] * 100 for h in top_10]
    colors = ['steelblue' if e > 0 else 'coral' for e in effects]
    
    ax2.barh(names, effects, color=colors, edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel("Effect Size (%)")
    ax2.set_title("Most Impactful Heads")
    ax2.invert_yaxis()
    
    plt.suptitle("Step 35: Real ToM Circuit (Chat Mode) - FIXED", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "step35_real_circuit.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": "chat",
            "max_tokens": 250,
            "n_scenarios": len(scenarios),
            "min_n_required": min_n,
            "methodology": "rigorous",
            "ablation_method": "pre_hook_on_o_proj_input",  # Correct method!
        },
        "baseline": {
            "accuracy": results['baseline']['accuracy'],
            "n": results['n'],
            "correct": results['baseline']['n_correct'],
        },
        "multiple_comparisons": {
            "n_tests": len(all_heads),
            "original_alpha": 0.05,
            "corrected_alpha": corrected_alpha,
            "n_significant_uncorrected": sum(1 for h in all_heads if h['significant_uncorrected']),
            "n_significant_corrected": len(significant_heads),
        },
        "ablations": results['ablations'],
        "significant_heads": [
            {
                "layer": h['layer'],
                "head": h['head'],
                "effect": h['effect'],
                "accuracy": h['accuracy'],
                "p_value": h['p_value'],
            }
            for h in significant_heads
        ],
        "top_helpful": [
            {
                "layer": h['layer'],
                "head": h['head'],
                "effect": h['effect'],
                "accuracy": h['accuracy'],
                "p_value": h['p_value'],
                "significant": h['significant'],
            }
            for h in helpful_heads[:5]
        ],
        "top_inhibitory": [
            {
                "layer": h['layer'],
                "head": h['head'],
                "effect": h['effect'],
                "accuracy": h['accuracy'],
                "p_value": h['p_value'],
                "significant": h['significant'],
            }
            for h in inhibitory_heads[-5:]
        ],
    }
    
    output_path = RESULTS_DIR / "step35_circuit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CIRCUIT HUNT COMPLETE")
    print(f"{'='*60}")
    print(f"\nBaseline accuracy: {results['baseline']['accuracy']:.1%} (n={results['n']})")
    print(f"\nMultiple comparisons:")
    print(f"  Tests performed: {len(all_heads)}")
    print(f"  Significant (uncorrected): {sum(1 for h in all_heads if h['significant_uncorrected'])}")
    print(f"  Significant (Bonferroni corrected): {len(significant_heads)}")
    
    if significant_heads:
        print(f"\nSignificant ToM heads (after correction):")
        for h in significant_heads[:5]:
            direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
            print(f"  L{h['layer']}H{h['head']} ({direction}): {h['effect']:+.1%}, p={h['p_value']:.4f}")
    else:
        print(f"\n⚠️  No heads pass Bonferroni correction - results may be spurious")
        print(f"   Consider: larger sample size, more targeted testing, or different layers")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
