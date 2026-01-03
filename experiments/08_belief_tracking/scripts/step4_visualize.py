"""
Step 4: Visualize Belief Tracking Results
==========================================
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 4: VISUALIZE RESULTS")
print("=" * 60)


def main():
    # Load results
    with open(RESULTS_DIR / "belief_analysis.json") as f:
        results = json.load(f)
    
    layers = results["layers"]
    x = np.arange(len(layers))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Agent Classification
    ax = axes[0, 0]
    accs = [results["minimal_pairs"][str(l)]["agent_classification"]["accuracy"] for l in layers]
    ax.bar(x, accs, color="#3498db", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance (50%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Agent Classification (Alice vs Bob)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    avg_acc = np.mean(accs)
    if avg_acc > 0.7:
        ax.text(0.5, 0.95, f"PASS: Model tracks WHO ({avg_acc:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.95, f"FAIL: No agent tracking ({avg_acc:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='red', fontweight='bold')
    
    # 2. Cross-Content Generalization
    ax = axes[0, 1]
    gen_accs = [results["minimal_pairs"][str(l)]["avg_generalization"] for l in layers]
    ax.bar(x, gen_accs, color="#27ae60", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Strong generalization")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Content Generalization\n(Train: 3 categories, Test: held-out)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    avg_gen = np.mean(gen_accs)
    if avg_gen > 0.6:
        ax.text(0.5, 0.95, f"PASS: Semantic encoding ({avg_gen:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.95, f"FAIL: Lexical confound ({avg_gen:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='red', fontweight='bold')
    
    # 3. Orthogonality
    ax = axes[1, 0]
    cosines = [results["minimal_pairs"][str(l)]["orthogonality"]["mean_cosine"] for l in layers]
    colors = ["#27ae60" if c < 0.3 else "#e74c3c" for c in cosines]
    ax.bar(x, cosines, color=colors, alpha=0.8)
    ax.axhline(y=0.3, color="orange", linestyle="--", label="Orthogonality threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Agent vs Content Orthogonality\n(Lower = more separable)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    n_ortho = sum(1 for c in cosines if c < 0.3)
    if n_ortho >= len(layers) // 2:
        ax.text(0.5, 0.95, f"PASS: {n_ortho}/{len(layers)} layers orthogonal", 
                transform=ax.transAxes, ha='center', fontsize=10, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.95, f"MIXED: {n_ortho}/{len(layers)} layers orthogonal", 
                transform=ax.transAxes, ha='center', fontsize=10, color='orange', fontweight='bold')
    
    # 4. Belief State Decoding
    ax = axes[1, 1]
    state_accs = [results["belief_scenarios"][str(l)]["state_4way_acc"] for l in layers]
    ax.bar(x, state_accs, color="#9b59b6", alpha=0.8)
    ax.axhline(y=0.25, color="red", linestyle="--", label="Chance (25%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("4-Way Belief State Decoding\n(neither/alice_only/bob_only/both)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    avg_state = np.mean(state_accs)
    if avg_state > 0.5:
        ax.text(0.5, 0.95, f"PASS: Belief tracking ({avg_state:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.95, f"WEAK: Some signal ({avg_state:.0%})", 
                transform=ax.transAxes, ha='center', fontsize=10, color='orange', fontweight='bold')
    
    plt.suptitle("Belief Tracking: Does the Model Know WHO Knows WHAT?", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "belief_tracking.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")
    
    # Print final interpretation
    print("\n" + "=" * 60)
    print("FINAL INTERPRETATION")
    print("=" * 60)
    
    passes = 0
    
    if avg_acc > 0.7:
        print("\n[PASS] Agent Classification: Model distinguishes Alice vs Bob")
        passes += 1
    else:
        print("\n[FAIL] Agent Classification: Cannot distinguish agents")
    
    if avg_gen > 0.6:
        print("[PASS] Cross-Content: Generalization suggests semantic encoding")
        passes += 1
    else:
        print("[FAIL] Cross-Content: Poor generalization suggests lexical confound")
    
    if n_ortho >= len(layers) // 2:
        print("[PASS] Orthogonality: Agent and content are separable")
        passes += 1
    else:
        print("[MIXED] Orthogonality: Partial separation only")
    
    if avg_state > 0.5:
        print("[PASS] Belief States: Can decode knowledge configurations")
        passes += 1
    else:
        print("[WEAK] Belief States: Some signal but limited")
    
    print(f"\nOverall: {passes}/4 tests passed")
    
    if passes >= 3:
        print("\n>>> CONCLUSION: EVIDENCE for genuine belief tracking <<<")
    elif passes >= 2:
        print("\n>>> CONCLUSION: PARTIAL evidence - needs more investigation <<<")
    else:
        print("\n>>> CONCLUSION: WEAK/NO evidence for belief tracking <<<")


if __name__ == "__main__":
    main()
























