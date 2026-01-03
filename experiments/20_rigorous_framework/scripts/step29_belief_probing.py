"""
Step 29: Linear Probing for Belief States

Train a linear probe to classify belief states from activations.
This gives us a quantitative measure of how well beliefs are encoded.

OUTPUT: results/step29_probing.json, figures/step29_*.png
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_scenarios(n=50):
    """Generate FB and TB scenarios."""
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
              "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tina",
              "Uma", "Victor", "Wendy", "Xena", "Yuki", "Zoe", "Adam", "Beth", "Carl", "Dana",
              "Evan", "Fay", "Greg", "Helen", "Ian", "Jane", "Kyle", "Lily", "Max", "Nina",
              "Owen", "Penny", "Quinn", "Rose", "Steve", "Tara", "Uma", "Vera", "Will", "Xia"]
    objects = ["ball", "key", "book", "phone", "wallet", "cup", "pen", "toy", "watch", "ring",
               "coin", "card", "note", "badge", "box", "bag", "hat", "scarf", "glove", "shoe",
               "letter", "photo", "map", "ticket", "stamp", "tool", "gift", "fruit", "candy", "drink"]
    loc1s = ["drawer", "basket", "shelf", "table", "bed", "desk", "box", "case", "bin", "tray"]
    loc2s = ["cupboard", "cabinet", "pocket", "bag", "container", "jar", "bucket", "crate", "pouch", "sack"]
    
    scenarios = []
    for i in range(n):
        a1 = agents[i % len(agents)]
        a2 = agents[(i + 1) % len(agents)]
        obj = objects[i % len(objects)]
        l1 = loc1s[i % len(loc1s)]
        l2 = loc2s[i % len(loc2s)]
        
        # False Belief - agent has outdated belief
        fb = {
            "type": "false_belief",
            "label": 0,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} left. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is?",
        }
        
        # True Belief - agent has correct belief
        tb = {
            "type": "true_belief",
            "label": 1,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} watched {a2} move the {obj} to the {l2}. Where does {a1} think the {obj} is?",
        }
        
        scenarios.extend([fb, tb])
    
    return scenarios


def collect_activations(model, tokenizer, scenarios, layer):
    """Collect activations at a specific layer."""
    activations = []
    labels = []
    
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations.append(hidden[0, -1, :].detach().cpu())
    
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(hook)
    
    for scenario in scenarios:
        inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        labels.append(scenario["label"])
    
    handle.remove()
    
    return torch.stack(activations).numpy(), np.array(labels)


def main():
    print("=" * 70)
    print("STEP 29: LINEAR PROBING FOR BELIEF STATES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTraining linear probes to classify FB vs TB from activations")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    scenarios = generate_scenarios(50)
    print(f"\nGenerated {len(scenarios)} scenarios (50 FB, 50 TB)")
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
    
    # ========================================
    # PROBE ACROSS LAYERS
    # ========================================
    print(f"\n{'='*60}")
    print("PROBING ACROSS LAYERS")
    print(f"{'='*60}")
    
    layer_accuracies = []
    layers_to_probe = [0, 6, 12, 16, 20, 24, 28, 32, 35]
    
    for layer in layers_to_probe:
        print(f"\nLayer {layer}...", end="")
        sys.stdout.flush()
        
        X, y = collect_activations(model, tokenizer, scenarios, layer)
        
        # Train linear probe with 5-fold cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X, y, cv=5)
        
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        layer_accuracies.append({
            "layer": layer,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "cv_scores": scores.tolist(),
        })
        
        print(f" Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")
    
    # Find best layer
    best_layer = max(layer_accuracies, key=lambda x: x["mean_accuracy"])
    print(f"\nBest layer: L{best_layer['layer']} ({best_layer['mean_accuracy']:.1%})")
    
    # ========================================
    # DETAILED ANALYSIS AT BEST LAYER
    # ========================================
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS AT LAYER {best_layer['layer']}")
    print(f"{'='*60}")
    
    X, y = collect_activations(model, tokenizer, scenarios, best_layer['layer'])
    
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X, y)
    
    # Get probe weights
    weights = probe.coef_[0]
    
    # Top positive dimensions (TB direction)
    top_tb_dims = np.argsort(weights)[-10:][::-1]
    # Top negative dimensions (FB direction)
    top_fb_dims = np.argsort(weights)[:10]
    
    print(f"\nTop dimensions for True Belief (agent saw):")
    for dim in top_tb_dims:
        print(f"  Dim {dim}: weight={weights[dim]:.3f}")
    
    print(f"\nTop dimensions for False Belief (agent left):")
    for dim in top_fb_dims:
        print(f"  Dim {dim}: weight={weights[dim]:.3f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name, "n_scenarios": len(scenarios)},
        "layer_accuracies": layer_accuracies,
        "best_layer": {
            "layer": best_layer["layer"],
            "accuracy": best_layer["mean_accuracy"],
        },
        "probe_analysis": {
            "top_tb_dimensions": top_tb_dims.tolist(),
            "top_fb_dimensions": top_fb_dims.tolist(),
            "weight_norm": float(np.linalg.norm(weights)),
        },
    }
    
    output_path = RESULTS_DIR / "step29_probing.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Figure 1: Probe accuracy across layers
    ax1 = axes[0]
    layers = [r["layer"] for r in layer_accuracies]
    accs = [r["mean_accuracy"] * 100 for r in layer_accuracies]
    stds = [r["std_accuracy"] * 100 for r in layer_accuracies]
    
    ax1.errorbar(layers, accs, yerr=stds, marker='o', capsize=5, linewidth=2, color='purple')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax1.axvline(x=best_layer["layer"], color='green', linestyle='--', alpha=0.5, label=f'Best (L{best_layer["layer"]})')
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Probe Accuracy (%)", fontsize=12)
    ax1.set_title("Belief State Probing Across Layers", fontsize=14, fontweight='bold')
    ax1.set_ylim(40, 100)
    ax1.legend()
    
    # Figure 2: Weight distribution at best layer
    ax2 = axes[1]
    ax2.hist(weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Probe Weight", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Probe Weight Distribution (L{best_layer['layer']})", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step29_probing.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 29 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

