"""
Step 48: Speedrun Summary - Aggregate all findings into figures

Combines results from:
- step45: Circuit re-validation
- step46: Explicit vs Implicit ToM
- step47: Self/Other/User probing

Generates final summary figures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11


def load_results():
    """Load all speedrun results."""
    results = {}
    
    files = [
        "step45_circuit_revalidation.json",
        "step46_explicit_implicit_tom.json",
        "step47_self_other_user_probing.json"
    ]
    
    for f in files:
        path = RESULTS_DIR / f
        if path.exists():
            with open(path) as fp:
                results[f.replace(".json", "")] = json.load(fp)
    
    return results


def plot_speedrun_summary(results):
    """Create comprehensive speedrun summary figure."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("MATS Mission Speedrun: Key Findings", fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Circuit Re-validation (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if "step45_circuit_revalidation" in results:
        r = results["step45_circuit_revalidation"]["results"]
        conditions = ["Baseline", "L17H4", "3-Head", "Random"]
        implicit = [r["baseline"]["implicit"]["accuracy"],
                   r["L17H4_ablation"]["implicit"]["accuracy"],
                   r["three_head_ablation"]["implicit"]["accuracy"],
                   r["random_ablation"]["implicit"]["accuracy"]]
        explicit = [r["baseline"]["explicit"]["accuracy"],
                   r["L17H4_ablation"]["explicit"]["accuracy"],
                   r["three_head_ablation"]["explicit"]["accuracy"],
                   r["random_ablation"]["explicit"]["accuracy"]]
        
        x = np.arange(len(conditions))
        width = 0.35
        ax1.bar(x - width/2, implicit, width, label='Implicit', color='#e74c3c')
        ax1.bar(x + width/2, explicit, width, label='Explicit', color='#27ae60')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=15)
        ax1.set_ylabel('Accuracy %')
        ax1.set_title('Circuit Re-validation', fontweight='bold')
        ax1.legend()
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    else:
        ax1.text(0.5, 0.5, "Waiting for results...", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Circuit Re-validation', fontweight='bold')
    
    # 2. Explicit vs Implicit (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if "step46_explicit_implicit_tom" in results:
        r = results["step46_explicit_implicit_tom"]["results"]
        conditions = ["Implicit", "Explicit", "Semi-Explicit", "Structured"]
        accuracies = [r["implicit"]["accuracy"], r["explicit"]["accuracy"],
                     r["semi_explicit"]["accuracy"], r["structured"]["accuracy"]]
        colors = ['#e74c3c', '#27ae60', '#f39c12', '#3498db']
        ax2.bar(conditions, accuracies, color=colors, edgecolor='black')
        ax2.set_ylabel('Accuracy %')
        ax2.set_title('Explicit vs Implicit ToM', fontweight='bold')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xticklabels(conditions, rotation=15)
        
        # Add effect annotation
        diff = r["explicit"]["accuracy"] - r["implicit"]["accuracy"]
        ax2.annotate(f'Explicit > Implicit by {diff:+.1f}%', xy=(0.5, 0.95), 
                    xycoords='axes fraction', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "Waiting for results...", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Explicit vs Implicit ToM', fontweight='bold')
    
    # 3. Self/Other/User Probing (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    if "step47_self_other_user_probing" in results:
        r = results["step47_self_other_user_probing"]["probe_results"]
        layers = sorted([int(k) for k in r.keys()])
        accuracies = [r[str(l)]["accuracy"] for l in layers]
        ax3.plot(layers, accuracies, 'o-', color='#9b59b6', linewidth=2, markersize=8)
        ax3.axhline(y=33.3, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Probe Accuracy %')
        ax3.set_title('Entity Probe (User/Self/Other)', fontweight='bold')
        ax3.legend()
        
        best = results["step47_self_other_user_probing"]["best_accuracy"]
        best_l = results["step47_self_other_user_probing"]["best_layer"]
        ax3.annotate(f'Best: L{best_l} = {best:.1f}%', xy=(best_l, best),
                    xytext=(best_l + 2, best - 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "Waiting for results...", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Entity Probe (User/Self/Other)', fontweight='bold')
    
    # 4-6. Key Findings Summary (bottom row)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Build summary text based on available results
    lines = ["KEY FINDINGS FROM SPEEDRUN:", ""]
    
    if "step45_circuit_revalidation" in results:
        eff = results["step45_circuit_revalidation"]["effects"]
        lines.append(f"1. CIRCUIT RE-VALIDATION:")
        lines.append(f"   - 3-Head ablation effect on Implicit ToM: {eff['three_head_implicit']:+.1f}%")
        lines.append(f"   - Random control effect: {eff['random_control']:+.1f}%")
        if eff['three_head_implicit'] > eff['random_control'] + 10:
            lines.append("   -> VALIDATED: Circuit finding holds!")
        else:
            lines.append("   -> NOT VALIDATED: Effect similar to random")
        lines.append("")
    
    if "step46_explicit_implicit_tom" in results:
        diff = results["step46_explicit_implicit_tom"]["explicit_implicit_diff"]
        lines.append(f"2. EXPLICIT vs IMPLICIT ToM:")
        lines.append(f"   - Explicit advantage: {diff:+.1f}%")
        if diff > 20:
            lines.append("   -> CONFIRMED: Model much better with explicit beliefs!")
        else:
            lines.append("   -> Model handles both similarly")
        lines.append("")
    
    if "step47_self_other_user_probing" in results:
        best = results["step47_self_other_user_probing"]["best_accuracy"]
        lines.append(f"3. SELF/OTHER/USER SEPARATION:")
        lines.append(f"   - Best probe accuracy: {best:.1f}% (chance: 33%)")
        if best > 80:
            lines.append("   -> STRONG: Distinct entity representations!")
        elif best > 50:
            lines.append("   -> MODERATE: Some separation exists")
        else:
            lines.append("   -> WEAK: Representations may be entangled")
    
    summary = "\n".join(lines)
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linewidth=2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = FIGURES_DIR / "08_speedrun_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("STEP 48: Speedrun Summary")
    print("="*70)
    
    results = load_results()
    print(f"\nLoaded results from {len(results)} experiments")
    
    for name in results:
        print(f"  - {name}")
    
    plot_speedrun_summary(results)
    
    print("\n[Summary figure generated]")


if __name__ == "__main__":
    main()


