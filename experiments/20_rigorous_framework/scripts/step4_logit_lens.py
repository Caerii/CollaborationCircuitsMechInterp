"""
Step 4: Logit Lens Analysis - WHERE Does the Model Decide?

HYPOTHESIS: The model's ToM decision crystallizes in middle layers (15-25),
            not at the very beginning or very end.

METHODOLOGY:
- Run logit lens on false-belief scenarios
- Track when correct > wrong logit first appears
- Compare FB vs TB to find where belief tracking happens
- Identify the "decision layer"

OUTPUT: results/step4_logit_lens.json, figures/step4_logit_lens_evolution.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis import LogitLens, plot_logit_lens
from scenarios import generate_n_scenarios

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"


def main():
    print("=" * 70)
    print("STEP 4: LOGIT LENS ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    config = ExperimentConfig()
    
    # Generate scenarios
    n = 20  # Smaller N for detailed analysis
    print(f"\nGenerating {n} scenarios for logit lens analysis...")
    fb_scenarios = generate_n_scenarios(n, "false_belief")
    tb_scenarios = generate_n_scenarios(n // 2, "true_belief")
    
    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Initialize logit lens
    lens = LogitLens(model, tokenizer)
    
    # ========================================
    # Analyze False Belief scenarios
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYZING FALSE BELIEF SCENARIOS")
    print(f"{'='*60}")
    
    fb_results = []
    decision_layers = []
    
    for i, scenario in enumerate(fb_scenarios):
        # Build prompt from scenario fields
        story = scenario["story"]
        question = scenario["question"]
        correct = scenario["correct"]
        options = scenario["options"]
        wrong = [o for o in options if o != correct][0] if len(options) > 1 else options[0]
        
        # Create completion-style prompt for logit lens
        prompt = f"{story}\n{question}\nThe answer is: the"
        
        result = lens.analyze(prompt, " " + correct, " " + wrong)
        fb_results.append(result)
        
        decision_layer = result.decision_layer_idx
        decision_layers.append(decision_layer)
        
        final_pred = result.predictions[-1] if result.predictions else "?"
        print(f"[{i+1}/{len(fb_scenarios)}] Decision at L{decision_layer}, Final: {final_pred}")
    
    # ========================================
    # Analyze True Belief scenarios
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYZING TRUE BELIEF SCENARIOS")
    print(f"{'='*60}")
    
    tb_results = []
    tb_decision_layers = []
    
    for i, scenario in enumerate(tb_scenarios):
        # Build prompt from scenario fields
        story = scenario["story"]
        question = scenario["question"]
        correct = scenario["correct"]
        options = scenario["options"]
        wrong = [o for o in options if o != correct][0] if len(options) > 1 else options[0]
        
        # Create completion-style prompt for logit lens
        prompt = f"{story}\n{question}\nThe answer is: the"
        
        result = lens.analyze(prompt, " " + correct, " " + wrong)
        tb_results.append(result)
        
        decision_layer = result.decision_layer_idx
        tb_decision_layers.append(decision_layer)
        
        final_pred = result.predictions[-1] if result.predictions else "?"
        print(f"[{i+1}/{len(tb_scenarios)}] Decision at L{decision_layer}, Final: {final_pred}")
    
    # ========================================
    # STATISTICS
    # ========================================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    # Decision layer statistics
    fb_mean_decision = np.mean(decision_layers)
    fb_std_decision = np.std(decision_layers)
    tb_mean_decision = np.mean(tb_decision_layers)
    tb_std_decision = np.std(tb_decision_layers)
    
    # Count how many predicted the target (correct) token
    fb_correct = sum(1 for r in fb_results if r.predictions[-1] == "target")
    tb_correct = sum(1 for r in tb_results if r.predictions[-1] == "target")
    
    print(f"\nFALSE BELIEF scenarios:")
    print(f"  Mean decision layer: {fb_mean_decision:.1f} +/- {fb_std_decision:.1f}")
    print(f"  Correct predictions: {fb_correct}/{len(fb_results)}")
    
    print(f"\nTRUE BELIEF scenarios:")
    print(f"  Mean decision layer: {tb_mean_decision:.1f} +/- {tb_std_decision:.1f}")
    print(f"  Correct predictions: {tb_correct}/{len(tb_results)}")
    
    # Hypothesis test
    in_middle = 15 <= fb_mean_decision <= 25
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}")
    print(f"\nH2: Decision in middle layers (15-25): {'SUPPORTED' if in_middle else 'NOT SUPPORTED'}")
    print(f"    (Actual mean: layer {fb_mean_decision:.1f})")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "n_fb_scenarios": len(fb_scenarios),
            "n_tb_scenarios": len(tb_scenarios),
        },
        "false_belief": {
            "mean_decision_layer": float(fb_mean_decision),
            "std_decision_layer": float(fb_std_decision),
            "n_correct": fb_correct,
            "n_total": len(fb_results),
            "decision_layer_histogram": {
                str(i): decision_layers.count(i) for i in range(max(decision_layers) + 1)
            } if decision_layers else {},
        },
        "true_belief": {
            "mean_decision_layer": float(tb_mean_decision),
            "std_decision_layer": float(tb_std_decision),
            "n_correct": tb_correct,
            "n_total": len(tb_results),
        },
        "hypothesis_tests": {
            "H2_decision_in_middle_layers": in_middle,
        },
        # Store detailed results for first 5 scenarios
        "sample_results": [
            {
                "layers": r.layers[:10] + ["..."] + r.layers[-5:],  # Truncate
                "diffs": r.diffs[:10] + r.diffs[-5:],
                "decision_layer": r.decision_layer,
                "decision_layer_idx": r.decision_layer_idx,
            } 
            for r in fb_results[:5]
        ],
    }
    
    output_path = RESULTS_DIR / "step4_logit_lens.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)  # Convert non-serializable to string
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: Logit evolution for sample scenarios
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Logit Lens: How ToM Decision Evolves Through Layers", fontsize=14, fontweight='bold')
    
    for idx, (result, ax) in enumerate(zip(fb_results[:4], axes.flat)):
        layers = list(range(len(result.diffs)))
        diffs = result.diffs
        
        colors = ['green' if d > 0 else 'red' for d in diffs]
        ax.bar(layers, diffs, color=colors, alpha=0.7, width=0.8)
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Mark decision point
        decision_idx = result.decision_layer_idx
        ax.axvline(x=decision_idx, color='blue', linestyle='--', alpha=0.7,
                   label=f'Decision: L{decision_idx}')
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Logit Diff (target - contrast)")
        final_pred = result.predictions[-1] if result.predictions else "?"
        ax.set_title(f"Scenario {idx+1}: Final={final_pred}")
        ax.legend(fontsize=8)
        ax.set_xlim(-1, len(layers))
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step4_logit_lens_evolution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Decision layer histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bins = range(0, max(decision_layers) + 2)
    ax.hist(decision_layers, bins=bins, alpha=0.7, color='steelblue', edgecolor='black',
            label='False Belief')
    ax.hist(tb_decision_layers, bins=bins, alpha=0.5, color='coral', edgecolor='black',
            label='True Belief')
    
    ax.axvline(x=fb_mean_decision, color='blue', linestyle='--', linewidth=2,
               label=f'FB Mean: {fb_mean_decision:.1f}')
    ax.axvline(x=tb_mean_decision, color='red', linestyle='--', linewidth=2,
               label=f'TB Mean: {tb_mean_decision:.1f}')
    
    # Highlight middle layers
    ax.axvspan(15, 25, alpha=0.2, color='green', label='Hypothesized region (15-25)')
    
    ax.set_xlabel("Decision Layer", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Where Does ToM Decision Crystallize?", fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step4_decision_layer_histogram.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 4 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

