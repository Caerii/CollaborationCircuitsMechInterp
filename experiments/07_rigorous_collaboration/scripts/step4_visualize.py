"""
Step 4: Create Visualizations
=============================

Publication-quality plots with error bars and significance.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("STEP 4: CREATE VISUALIZATIONS")
print("=" * 60)


def main():
    # Load results
    print("\nLoading analysis results...")
    with open(RESULTS_DIR / "analysis_results.json") as f:
        results = json.load(f)
    
    # Convert keys to int
    results = {int(k): v for k, v in results.items()}
    layers = sorted(results.keys())
    
    # Extract data
    accs = [results[l]["cv_accuracy"] for l in layers]
    ci_lows = [results[l]["ci_95_low"] for l in layers]
    ci_highs = [results[l]["ci_95_high"] for l in layers]
    selectivities = [results[l]["selectivity"] for l in layers]
    p_values = [results[l]["p_value"] for l in layers]
    effect_sizes = [results[l]["cohens_d"] for l in layers]
    
    x = np.arange(len(layers))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Accuracy with CI
    ax = axes[0, 0]
    yerr = [np.array(accs) - np.array(ci_lows), np.array(ci_highs) - np.array(accs)]
    ax.bar(x, accs, yerr=yerr, capsize=3, color="#3498db", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance (50%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Classification Accuracy with 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selectivity
    ax = axes[0, 1]
    colors = ["#27ae60" if s > 0.1 else "#e74c3c" for s in selectivities]
    ax.bar(x, selectivities, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Meaningful (>0.1)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity (Acc - Baseline)")
    ax.set_title("Selectivity: Real Signal vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: P-values (log scale)
    ax = axes[1, 0]
    ax.semilogy(x, p_values, 'o-', color="#9b59b6", markersize=8)
    ax.axhline(y=0.05, color="orange", linestyle="--", label="alpha = 0.05")
    ax.axhline(y=0.05/len(layers), color="red", linestyle="--", 
               label=f"Bonferroni = {0.05/len(layers):.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Statistical Significance (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Effect size
    ax = axes[1, 1]
    colors = ["#27ae60" if d > 0.5 else "#f39c12" if d > 0.2 else "#e74c3c" for d in effect_sizes]
    ax.bar(x, effect_sizes, color=colors, alpha=0.8)
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, label="Small (d=0.2)")
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Medium (d=0.5)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size (larger = stronger effect)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Rigorous Analysis: Private vs Shared Knowledge States", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "rigorous_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")
    
    # Summary interpretation
    n_sig = sum(1 for l in layers if results[l]["significant_corrected"])
    n_mean = sum(1 for l in layers if results[l]["meaningful"])
    avg_acc = np.mean(accs)
    avg_sel = np.mean(selectivities)
    
    print("\n" + "=" * 60)
    print("FINAL INTERPRETATION")
    print("=" * 60)
    print(f"\nAverage accuracy: {avg_acc:.1%}")
    print(f"Average selectivity: {avg_sel:+.1%}")
    print(f"Significant layers (Bonferroni): {n_sig}/{len(layers)}")
    print(f"Meaningful effect layers: {n_mean}/{len(layers)}")
    
    avg_d = np.mean(effect_sizes)
    
    # More nuanced interpretation considering all metrics
    if avg_acc >= 0.95 and avg_d > 0.5:
        print(f"\n>>> CONCLUSION: STRONG evidence for knowledge state encoding <<<")
        print(f"    - Near-perfect accuracy ({avg_acc:.1%}) indicates reliable separation")
        print(f"    - Large effect size (d={avg_d:.2f}) indicates meaningful distinction")
    elif avg_sel > 0.1 and n_sig >= len(layers) // 2:
        print("\n>>> CONCLUSION: ROBUST evidence for knowledge state encoding <<<")
    elif avg_sel > 0.05 or avg_d > 0.2:
        print("\n>>> CONCLUSION: MODERATE evidence - some signal detected <<<")
    else:
        print("\n>>> CONCLUSION: NO evidence - results indistinguishable from chance <<<")


if __name__ == "__main__":
    main()


