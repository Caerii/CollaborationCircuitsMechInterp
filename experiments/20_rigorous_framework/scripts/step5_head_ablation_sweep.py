"""
Step 5: Head Ablation Sweep - WHICH Heads Matter for ToM?

NOW USING LIBRARY: ChatModeCircuitAnalyzer for proper methodology!

HYPOTHESIS: Ablating critical heads will drop accuracy by >20%,
            while ablating random heads will change accuracy <5%.

METHODOLOGY:
- Test heads in layers 25-35 (UPDATED based on Step 4 Logit Lens findings!)
- For each head: ablate and measure accuracy change
- Use N≥50 scenarios (methodology requirement)
- Use CHAT MODE (not completion mode) - proper for reasoning model
- Statistical tests and multiple comparisons correction

OUTPUT: results/step5_head_ablation.json, figures/step5_head_importance_heatmap.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from scenarios.templates import generate_n_scenarios
from analysis.circuits import ChatModeCircuitAnalyzer
from analysis.controls import bonferroni_correct, accuracy_with_ci

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 5: HEAD ABLATION SWEEP (USING LIBRARY)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\n✅ Using ChatModeCircuitAnalyzer (proper methodology)")
    print("✅ Using chat mode (not completion mode)")
    print("✅ Using statistical tests and multiple comparisons correction")
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
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Model loaded! {n_layers} layers, {n_heads} heads")
    sys.stdout.flush()
    
    # Use library!
    analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
    
    # Generate scenarios with proper methodology (n≥50)
    min_n = config.min_samples_per_condition
    print(f"\nGenerating scenarios (n≥{min_n} required)...")
    sys.stdout.flush()
    
    all_scenarios = generate_n_scenarios(n=min_n, use_novel_names=config.require_novel_names)
    
    # Filter to false belief for ablation testing
    scenarios = [s for s in all_scenarios if s.get("type") == "false_belief"]
    
    # If we don't have enough FB, generate more
    while len(scenarios) < min_n:
        additional = generate_n_scenarios(n=min_n, use_novel_names=config.require_novel_names, seed=None)
        scenarios.extend([s for s in additional if s.get("type") == "false_belief"])
        if len(scenarios) >= min_n:
            break
    
    scenarios = scenarios[:min_n]
    print(f"Using {len(scenarios)} false belief scenarios")
    sys.stdout.flush()
    
    # Focus on layers 25-35 based on Step 4 Logit Lens analysis
    layers_to_test = [l for l in range(25, min(36, n_layers))]
    
    # Test every 4th head for speed
    heads_per_layer = 8  # 32 heads / 4 = 8 heads per layer
    
    print(f"\nTesting layers: {layers_to_test}")
    print(f"Testing {heads_per_layer} heads per layer")
    print(f"Total ablations: {len(layers_to_test) * heads_per_layer}")
    sys.stdout.flush()
    
    # Run ablation sweep using library!
    print(f"\n{'='*60}")
    print("ABLATION SWEEP (using library)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    results = analyzer.ablation_sweep(
        scenarios=scenarios,
        layers_to_test=layers_to_test,
        heads_per_layer=heads_per_layer,
        max_tokens=config.max_tokens
    )
    
    baseline_acc = results['baseline']['accuracy']
    
    # Get significant heads with multiple comparisons correction
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    sys.stdout.flush()
    
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
    all_heads.sort(key=lambda x: abs(x['effect']), reverse=True)
    helpful_heads = [h for h in all_heads if h['effect'] > 0]  # Ablation hurts = head is helpful
    inhibitory_heads = [h for h in all_heads if h['effect'] < 0]  # Ablation helps = head is inhibitory
    
    print(f"\n{'='*60}")
    print("TOP IMPACT HEADS")
    print(f"{'='*60}")
    
    print("\nMost impactful heads (by absolute effect):")
    for i, h in enumerate(all_heads[:10]):
        direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
        sig = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  {i+1}. L{h['layer']}H{h['head']} ({direction}): {h['effect']:+.1%}, p={h['p_value']:.4f}{sig}")
    
    print(f"\nHELPFUL HEADS (ablation hurts performance): {len(helpful_heads)}")
    for h in helpful_heads[:5]:
        sig = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: {h['effect']:+.1%}, p={h['p_value']:.4f}{sig}")
    
    print(f"\nINHIBITORY HEADS (ablation helps performance): {len(inhibitory_heads)}")
    for h in inhibitory_heads[:5]:
        sig = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: {h['effect']:+.1%}, p={h['p_value']:.4f}{sig}")
    
    print("\nLegend: *** = significant after correction, * = significant uncorrected")
    sys.stdout.flush()
    
    # ========================================
    # HYPOTHESIS TEST
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}")
    
    max_impact = max(abs(h['effect']) for h in all_heads) if all_heads else 0
    
    h3_critical_drop = max_impact > 0.20
    h3_has_significant = len(significant_heads) > 0
    
    print(f"\nH3a: Critical heads cause >20% change: {'SUPPORTED' if h3_critical_drop else 'NOT SUPPORTED'}")
    print(f"     (Max impact: {max_impact:.1%})")
    
    print(f"\nH3b: Significant heads found (after correction): {'SUPPORTED' if h3_has_significant else 'NOT SUPPORTED'}")
    print(f"     (Significant heads: {len(significant_heads)})")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "n_scenarios": len(scenarios),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "layers_tested": layers_to_test,
            "heads_per_layer": heads_per_layer,
            "using_library": True,
            "methodology": "rigorous",
        },
        "baseline": {
            "accuracy": baseline_acc,
            "n": results['n'],
            "n_correct": results['baseline']['n_correct'],
        },
        "multiple_comparisons": {
            "n_tests": len(all_heads),
            "original_alpha": 0.05,
            "corrected_alpha": corrected_alpha,
            "n_significant_uncorrected": sum(1 for h in all_heads if h['significant_uncorrected']),
            "n_significant_corrected": len(significant_heads),
        },
        "ablation_results": results['ablations'],
        "all_heads": all_heads,
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
            for h in helpful_heads[:10]
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
            for h in inhibitory_heads[:10]
        ],
        "hypothesis_tests": {
            "H3a_critical_causes_20pct_change": h3_critical_drop,
            "H3b_has_significant_heads": h3_has_significant,
        },
    }
    
    output_path = RESULTS_DIR / "step5_head_ablation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    sys.stdout.flush()
    
    # Figure: Head importance heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create matrix from results
    layers = sorted(results['ablations'].keys())
    heads = sorted(list(results['ablations'][layers[0]].keys()))
    
    matrix = np.zeros((len(layers), len(heads)))
    for i, layer_idx in enumerate(layers):
        for j, head_idx in enumerate(heads):
            if layer_idx in results['ablations'] and head_idx in results['ablations'][layer_idx]:
                matrix[i, j] = results['ablations'][layer_idx][head_idx]['effect']
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdBu', vmin=-0.3, vmax=0.3, aspect='auto')
    
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([f'H{h}' for h in heads], rotation=45)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'L{l}' for l in layers])
    
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Head Ablation Impact on ToM Accuracy\n(Blue = helpful, Red = inhibitory)", 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Accuracy Change", shrink=0.8)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step5_head_importance_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Bar chart of top heads
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Helpful heads (ablation hurts)
    ax1 = axes[0]
    if helpful_heads:
        top_helpful = helpful_heads[:8]
        labels = [f"L{h['layer']}H{h['head']}" for h in top_helpful]
        values = [h['effect'] * 100 for h in top_helpful]  # Convert to percentage
        colors = ['steelblue' if h['significant'] else 'lightblue' for h in top_helpful]
        ax1.barh(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Accuracy Change (%)")
        ax1.set_title("HELPFUL HEADS\n(Ablation hurts ToM)", fontweight='bold')
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.invert_yaxis()
    
    # Inhibitory heads (ablation helps)
    ax2 = axes[1]
    if inhibitory_heads:
        top_inhibitory = inhibitory_heads[:8]
        labels = [f"L{h['layer']}H{h['head']}" for h in top_inhibitory]
        values = [h['effect'] * 100 for h in top_inhibitory]  # Convert to percentage
        colors = ['coral' if h['significant'] else 'lightcoral' for h in top_inhibitory]
        ax2.barh(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Accuracy Change (%)")
        ax2.set_title("INHIBITORY HEADS\n(Ablation helps ToM)", fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.invert_yaxis()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step5_top_heads_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 5 COMPLETE")
    print(f"{'='*60}")
    print(f"\nBaseline accuracy: {baseline_acc:.1%}")
    print(f"Significant heads (after correction): {len(significant_heads)}")
    if significant_heads:
        print("\nTop significant heads:")
        for h in significant_heads[:5]:
            direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
            print(f"  L{h['layer']}H{h['head']} ({direction}): {h['effect']:+.1%}, p={h['p_value']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

