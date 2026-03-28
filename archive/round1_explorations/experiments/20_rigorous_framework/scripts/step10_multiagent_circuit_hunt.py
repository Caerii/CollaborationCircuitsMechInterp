"""
Step 10: Multi-Agent Circuit Hunt (FIXED VERSION)

NOW using proper methodology:
- Chat mode (not completion mode)
- n≥50 scenarios with proper counterbalancing
- Statistical tests with multiple comparisons correction
- Library components (ChatModeCircuitAnalyzer)

Goal: Find which heads causally affect multi-agent ToM reasoning.

OUTPUT: results/step10_multiagent_circuit.json, figures/step10_*.png
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
from scenarios.multi_agent import MultiAgentScenarioGenerator
from analysis.circuits import ChatModeCircuitAnalyzer
from analysis.controls import bonferroni_correct

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 10: MULTI-AGENT CIRCUIT HUNT (FIXED)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nFinding multi-agent ToM circuits with proper methodology!")
    print("✅ Using chat mode (not completion mode)")
    print("✅ Using library components (ChatModeCircuitAnalyzer)")
    print("✅ Using proper scenario generation (n≥50)")
    print("✅ Statistical tests with multiple comparisons correction")
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
    
    # Generate multi-agent scenarios with proper methodology
    print("\nGenerating multi-agent scenarios...")
    sys.stdout.flush()
    
    min_n = config.min_samples_per_condition
    print(f"  Target: n≥{min_n} scenarios (methodology requirement)")
    sys.stdout.flush()
    
    # Use multi-agent scenario generator
    generator = MultiAgentScenarioGenerator(
        use_novel_names=config.require_novel_names,
        seed=42
    )
    
    # Generate balanced set of multi-agent scenarios
    scenarios = generator.generate_balanced_set(n_per_type=min_n // 5)  # 5 types, so n_per_type gets us ~min_n total
    
    # If we don't have enough, generate more
    while len(scenarios) < min_n:
        additional = generator.generate_balanced_set(n_per_type=10)
        scenarios.extend(additional)
        if len(scenarios) >= min_n * 2:  # Cap at 2x to avoid infinite loop
            break
    
    scenarios = scenarios[:min_n]
    
    # Count by type
    type_counts = {}
    for s in scenarios:
        s_type = s.get('type', s.get('scenario_type', 'unknown'))
        type_counts[s_type] = type_counts.get(s_type, 0) + 1
    
    print(f"  Generated: {len(scenarios)} multi-agent scenarios")
    print(f"  Types: {dict(type_counts)}")
    sys.stdout.flush()
    
    # Test across all layers (multi-agent was found in early-mid layers before, but that was flawed)
    n_layers = model.config.num_hidden_layers
    # Test every 4th layer for speed, but cover early, mid, and late layers
    layers_to_test = list(range(0, n_layers, 4))  # Every 4th layer
    
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
        heads_per_layer=4,  # Test 4 heads per layer
        max_tokens=500  # Full token budget for multi-agent reasoning
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
        for h in significant_heads[:15]:
            direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
            print(f"  L{h['layer']}H{h['head']} ({direction}): effect={h['effect']:+.1%}, p={h['p_value']:.4f}, acc={h['accuracy']:.1%}")
    else:
        print("  None (no heads pass Bonferroni correction)")
    
    print("\n" + "="*60)
    print("MOST HELPFUL HEADS (ablation hurts performance):")
    print("="*60)
    for h in helpful_heads[:10]:
        sig_marker = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: effect={h['effect']:+.1%}, acc={h['accuracy']:.1%}, p={h['p_value']:.4f}{sig_marker}")
    
    print("\n" + "="*60)
    print("MOST INHIBITORY HEADS (ablation helps performance):")
    print("="*60)
    for h in inhibitory_heads[-10:]:
        sig_marker = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']}H{h['head']}: effect={h['effect']:+.1%}, acc={h['accuracy']:.1%}, p={h['p_value']:.4f}{sig_marker}")
    
    print("\nLegend: *** = significant after correction, * = significant uncorrected")
    sys.stdout.flush()
    
    # Compare to single-agent ToM heads (from step 35 if available)
    print("\n" + "="*60)
    print("COMPARISON TO SINGLE-AGENT ToM CIRCUITS")
    print("="*60)
    
    # Try to load step 35 results for comparison
    step35_path = RESULTS_DIR / "step35_circuit.json"
    if step35_path.exists():
        with open(step35_path, 'r') as f:
            step35_data = json.load(f)
        
        # Extract significant heads from step 35
        tom_heads = []
        if 'significant_heads' in step35_data:
            for h in step35_data['significant_heads']:
                tom_heads.append((h['layer'], h['head']))
        
        multiagent_head_set = set((h['layer'], h['head']) for h in significant_heads)
        tom_head_set = set(tom_heads)
        
        overlap = multiagent_head_set & tom_head_set
        only_multiagent = multiagent_head_set - tom_head_set
        only_tom = tom_head_set - multiagent_head_set
        
        print(f"\nOverlap (both ToM and Multi-Agent): {len(overlap)}")
        for h in overlap:
            print(f"  L{h[0]}H{h[1]}")
        
        print(f"\nMulti-Agent ONLY (not single-agent ToM): {len(only_multiagent)}")
        for h in list(only_multiagent)[:10]:
            print(f"  L{h[0]}H{h[1]}")
        
        print(f"\nSingle-Agent ToM ONLY (not Multi-Agent): {len(only_tom)}")
        for h in only_tom:
            print(f"  L{h[0]}H{h[1]}")
    else:
        print("  Step 35 results not found - run step35 first for comparison")
    
    # Visualize
    print("\nGenerating visualization...")
    sys.stdout.flush()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Heatmap of ablation effects
    ax1 = axes[0]
    layers = sorted(results['ablations'].keys())
    if layers:
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
        ax1.set_title("Multi-Agent Circuit: Ablation Effects\n(Blue=Helpful, Red=Inhibitory)", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label="Accuracy Change")
    
    # Plot 2: Bar chart of most impactful heads
    ax2 = axes[1]
    
    # Top 10 most impactful (by absolute effect)
    top_heads = sorted(all_heads, key=lambda x: abs(x['effect']), reverse=True)[:10]
    
    if top_heads:
        labels = [f"L{h['layer']}H{h['head']}" for h in top_heads]
        effects = [h['effect'] * 100 for h in top_heads]  # Convert to percentage
        colors = ['steelblue' if e > 0 else 'coral' for e in effects]
        
        bars = ax2.barh(range(len(labels)), effects, color=colors, edgecolor='black')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel("Accuracy Change (%)", fontsize=11)
        ax2.set_title("Top 10 Most Impactful Heads", fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add significance markers
        for i, h in enumerate(top_heads):
            if h['significant']:
                ax2.text(effects[i] + (2 if effects[i] > 0 else -2), i, '***', 
                        va='center', fontsize=10, fontweight='bold')
            elif h['significant_uncorrected']:
                ax2.text(effects[i] + (2 if effects[i] > 0 else -2), i, '*', 
                        va='center', fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step10_multiagent_circuit.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_scenarios": len(scenarios),
            "layers_tested": layers_to_test,
            "heads_per_layer": 4,
            "min_samples_per_condition": min_n,
        },
        "baseline": {
            "accuracy": results['baseline']['accuracy'],
            "n_correct": results['baseline']['n_correct'],
            "n_total": results['baseline']['n_total'],
        },
        "scenario_types": type_counts,
        "ablation_results": results['ablations'],
        "all_heads": all_heads,
        "significant_heads": [h for h in significant_heads],
        "helpful_heads": [h for h in helpful_heads[:10]],
        "inhibitory_heads": [h for h in inhibitory_heads[-10:]],
        "statistics": {
            "n_tests": len(all_heads),
            "corrected_alpha": corrected_alpha,
            "n_significant_uncorrected": sum(1 for h in all_heads if h['significant_uncorrected']),
            "n_significant_corrected": len(significant_heads),
        },
    }
    
    output_path = RESULTS_DIR / "step10_multiagent_circuit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print("STEP 10 COMPLETE")
    print(f"{'='*60}")
    print("\nKey findings:")
    if significant_heads:
        print(f"  - Found {len(significant_heads)} heads with significant effects (after correction)")
        print(f"  - Most helpful: {helpful_heads[0]['layer']}H{helpful_heads[0]['head']} ({helpful_heads[0]['effect']:+.1%})" if helpful_heads else "")
        print(f"  - Most inhibitory: {inhibitory_heads[-1]['layer']}H{inhibitory_heads[-1]['head']} ({inhibitory_heads[-1]['effect']:+.1%})" if inhibitory_heads else "")
    else:
        print("  - No heads passed Bonferroni correction")
        print("  - Previous findings (L18H16, etc.) may have been artifacts of flawed methodology")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
