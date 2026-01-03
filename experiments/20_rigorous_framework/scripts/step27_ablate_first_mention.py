"""
Step 27: Ablate First-Mention Circuit to Reveal ToM

Step 24 found the first-mention heads: L23H4, L13H10, L31H15
If we ablate these, does True Belief improve WITHOUT explicit beliefs?

This would prove the model HAS ToM but it's obscured by heuristics!

OUTPUT: results/step27_ablate_heuristic.json, figures/step27_*.png
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
from analysis.circuit_analysis import CircuitAnalysis

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_test_scenarios(n=20):
    """Generate FB and TB scenarios."""
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
              "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tina"]
    objects = ["ball", "key", "book", "phone", "wallet", "cup", "pen", "toy", "watch", "ring",
               "coin", "card", "note", "badge", "box", "bag", "hat", "scarf", "glove", "shoe"]
    loc1s = ["drawer", "basket", "shelf", "table", "bed", "chair", "desk", "counter", "bench", "cabinet",
             "closet", "trunk", "case", "bin", "tray", "rack", "hook", "slot", "nook", "corner"]
    loc2s = ["cupboard", "box", "cabinet", "pocket", "bag", "container", "jar", "bucket", "crate", "hamper",
             "pouch", "sack", "envelope", "folder", "binder", "sleeve", "wrapper", "cover", "case", "holder"]
    
    scenarios = []
    
    for i in range(n):
        a1 = agents[i % len(agents)]
        a2 = agents[(i + 1) % len(agents)]
        obj = objects[i % len(objects)]
        l1 = loc1s[i % len(loc1s)]
        l2 = loc2s[i % len(loc2s)]
        
        # False Belief
        fb = {
            "type": "false_belief",
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} left. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l1}",
            "wrong": f" {l2}",
        }
        
        # True Belief (this is what we want to improve!)
        tb = {
            "type": "true_belief",
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} stayed and watched. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l2}",
            "wrong": f" {l1}",
        }
        
        scenarios.extend([fb, tb])
    
    return scenarios


def evaluate_scenario(model, tokenizer, scenario):
    """Evaluate a single scenario."""
    inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
    
    correct_ids = tokenizer.encode(scenario["correct"], add_special_tokens=False)
    wrong_ids = tokenizer.encode(scenario["wrong"], add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_logit = float(logits[correct_ids[0]])
    wrong_logit = float(logits[wrong_ids[0]])
    
    return {
        "correct": correct_logit > wrong_logit,
        "logit_diff": correct_logit - wrong_logit,
    }


def main():
    print("=" * 70)
    print("STEP 27: ABLATE FIRST-MENTION CIRCUIT TO REVEAL ToM")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nHypothesis: Ablating first-mention heads will improve True Belief")
    print("without needing explicit belief statements!")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # First-mention heads from Step 24
    first_mention_heads = [(23, 4), (31, 15), (13, 10), (23, 5), (18, 15)]
    
    # Generate scenarios
    scenarios = generate_test_scenarios(20)
    print(f"\nGenerated {len(scenarios)} scenarios (20 FB, 20 TB)")
    sys.stdout.flush()
    
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
    print("Model loaded!")
    sys.stdout.flush()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    circuit = CircuitAnalysis(model, tokenizer, n_layers, n_heads)
    
    # ========================================
    # BASELINE EVALUATION
    # ========================================
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION")
    print(f"{'='*60}")
    
    baseline_results = {"false_belief": [], "true_belief": []}
    for scenario in scenarios:
        result = evaluate_scenario(model, tokenizer, scenario)
        baseline_results[scenario["type"]].append(result)
        print(".", end="")
        sys.stdout.flush()
    print(" done!")
    
    fb_baseline = sum(1 for r in baseline_results["false_belief"] if r["correct"]) / len(baseline_results["false_belief"])
    tb_baseline = sum(1 for r in baseline_results["true_belief"] if r["correct"]) / len(baseline_results["true_belief"])
    
    print(f"\nFalse Belief baseline: {fb_baseline:.0%}")
    print(f"True Belief baseline: {tb_baseline:.0%}")
    
    # ========================================
    # ABLATED EVALUATION
    # ========================================
    print(f"\n{'='*60}")
    print("ABLATED EVALUATION (removing first-mention heads)")
    print(f"Heads: {first_mention_heads}")
    print(f"{'='*60}")
    
    # Ablate the first-mention heads
    circuit.ablate_heads(first_mention_heads)
    
    ablated_results = {"false_belief": [], "true_belief": []}
    for scenario in scenarios:
        result = evaluate_scenario(model, tokenizer, scenario)
        ablated_results[scenario["type"]].append(result)
        print(".", end="")
        sys.stdout.flush()
    print(" done!")
    
    circuit._clear_hooks()
    
    fb_ablated = sum(1 for r in ablated_results["false_belief"] if r["correct"]) / len(ablated_results["false_belief"])
    tb_ablated = sum(1 for r in ablated_results["true_belief"] if r["correct"]) / len(ablated_results["true_belief"])
    
    print(f"\nFalse Belief ablated: {fb_ablated:.0%}")
    print(f"True Belief ablated: {tb_ablated:.0%}")
    
    # ========================================
    # ANALYSIS
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    fb_change = fb_ablated - fb_baseline
    tb_change = tb_ablated - tb_baseline
    
    print(f"\nFalse Belief: {fb_baseline:.0%} -> {fb_ablated:.0%} ({fb_change:+.0%})")
    print(f"True Belief: {tb_baseline:.0%} -> {tb_ablated:.0%} ({tb_change:+.0%})")
    
    if tb_change > 0.1:
        print("\n*** SUCCESS: Ablating first-mention heads IMPROVES True Belief! ***")
        print("This proves the model HAS ToM but it's obscured by heuristics!")
    elif fb_change < -0.1 and tb_change > -0.1:
        print("\n*** PARTIAL: Ablation disrupts FB more than TB ***")
        print("First-mention circuit is critical for FB but not TB.")
    else:
        print("\n*** Ablation had minimal effect ***")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "first_mention_heads": [{"layer": l, "head": h} for l, h in first_mention_heads],
        "baseline": {
            "false_belief": fb_baseline,
            "true_belief": tb_baseline,
        },
        "ablated": {
            "false_belief": fb_ablated,
            "true_belief": tb_ablated,
        },
        "changes": {
            "false_belief": fb_change,
            "true_belief": tb_change,
        },
    }
    
    output_path = RESULTS_DIR / "step27_ablate_heuristic.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    baseline_vals = [fb_baseline * 100, tb_baseline * 100]
    ablated_vals = [fb_ablated * 100, tb_ablated * 100]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='coral')
    ax.bar(x + width/2, ablated_vals, width, label='After Ablating\nFirst-Mention Heads', color='seagreen')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(['False Belief', 'True Belief'], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Effect of Ablating First-Mention Circuit on ToM", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add change annotations
    for i, (b, a) in enumerate(zip(baseline_vals, ablated_vals)):
        change = a - b
        color = 'green' if change > 0 else 'red'
        ax.annotate(f'{change:+.0f}%', xy=(i + width/2, a + 2),
                   ha='center', fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step27_ablate_heuristic.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 27 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

