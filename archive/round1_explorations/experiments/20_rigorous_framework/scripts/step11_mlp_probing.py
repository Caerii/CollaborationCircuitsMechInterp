"""
Step 11: MLP Probing - Do MLPs Encode Belief States?

Hypothesis: While attention heads track AGENTS, MLPs encode BELIEF STATES.

METHOD:
- Collect MLP activations for true-belief vs false-belief scenarios
- Train linear probes to classify belief state from MLP activations
- See which layers/neurons are most predictive

OUTPUT: results/step11_mlp_probing.json, figures/step11_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def create_belief_scenarios():
    """Create scenarios with clear true/false belief distinction."""
    scenarios = []
    
    # FALSE BELIEF scenarios (agent has outdated belief)
    false_belief = [
        "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef left. Waiter moved ingredients to cabinet B. Where does Chef think ingredients are?",
        "Sally placed her toy in the basket. Sally went outside. Anne moved the toy to the box. Where does Sally think the toy is?",
        "Dad put cookies in the jar. Dad left for work. Mom moved cookies to the cupboard. Where does Dad think the cookies are?",
        "Teacher put chalk in drawer 1. Teacher left. Student moved chalk to drawer 2. Where does Teacher think chalk is?",
        "Mark put his phone on the table. Mark went to shower. Lisa moved phone to the shelf. Where does Mark think phone is?",
        "Emma put her keys in her purse. Emma left the room. Jack moved keys to the hook. Where does Emma think keys are?",
    ]
    
    # TRUE BELIEF scenarios (agent knows current state)
    true_belief = [
        "Alice put the ball in the drawer. Alice stayed and watched. Bob moved the ball to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom watched as Jerry moved the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef saw Waiter move ingredients to cabinet B. Where does Chef think ingredients are?",
        "Sally placed her toy in the basket. Sally watched Anne move the toy to the box. Where does Sally think the toy is?",
        "Dad put cookies in the jar. Dad watched Mom move cookies to the cupboard. Where does Dad think the cookies are?",
        "Teacher put chalk in drawer 1. Teacher saw Student move chalk to drawer 2. Where does Teacher think chalk is?",
        "Mark put his phone on the table. Mark watched Lisa move phone to the shelf. Where does Mark think phone is?",
        "Emma put her keys in her purse. Emma watched Jack move keys to the hook. Where does Emma think keys are?",
    ]
    
    for prompt in false_belief:
        scenarios.append({"prompt": prompt, "belief_type": "false", "label": 0})
    
    for prompt in true_belief:
        scenarios.append({"prompt": prompt, "belief_type": "true", "label": 1})
    
    return scenarios


def extract_mlp_activations(model, tokenizer, prompt, layers):
    """Extract MLP hidden states for specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    activations = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # MLP output is the output of the entire MLP block
            activations[layer_idx] = output[0, -1, :].detach().cpu().numpy()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in layers:
        mlp = model.model.layers[layer_idx].mlp
        handle = mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return activations


def main():
    print("=" * 70)
    print("STEP 11: MLP PROBING FOR BELIEF STATES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Create scenarios
    scenarios = create_belief_scenarios()
    print(f"\nCreated {len(scenarios)} belief scenarios")
    print(f"  - False belief: {sum(1 for s in scenarios if s['label'] == 0)}")
    print(f"  - True belief: {sum(1 for s in scenarios if s['label'] == 1)}")
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
    hidden_size = model.config.hidden_size
    print(f"Model has {n_layers} layers, hidden_size={hidden_size}")
    
    # Layers to probe (focus on critical region + sample others)
    layers_to_probe = list(range(0, n_layers, 4)) + [30, 31, 32, 33, 34, 35]
    layers_to_probe = sorted(set(layers_to_probe))
    print(f"Probing {len(layers_to_probe)} layers: {layers_to_probe}")
    sys.stdout.flush()
    
    # ========================================
    # EXTRACT ACTIVATIONS
    # ========================================
    print(f"\n{'='*60}")
    print("EXTRACTING MLP ACTIVATIONS")
    print(f"{'='*60}")
    
    X_by_layer = {layer: [] for layer in layers_to_probe}
    y = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\rProcessing scenario {i+1}/{len(scenarios)}...", end="")
        sys.stdout.flush()
        
        activations = extract_mlp_activations(
            model, tokenizer, scenario["prompt"], layers_to_probe
        )
        
        for layer_idx in layers_to_probe:
            X_by_layer[layer_idx].append(activations[layer_idx])
        
        y.append(scenario["label"])
    
    print("\nActivation extraction complete!")
    
    # Convert to arrays
    y = np.array(y)
    for layer_idx in layers_to_probe:
        X_by_layer[layer_idx] = np.array(X_by_layer[layer_idx])
    
    # ========================================
    # TRAIN PROBES
    # ========================================
    print(f"\n{'='*60}")
    print("TRAINING LINEAR PROBES")
    print(f"{'='*60}")
    
    probe_results = {}
    
    for layer_idx in layers_to_probe:
        X = X_by_layer[layer_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train probe with cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X_scaled, y, cv=min(5, len(y)//2), scoring='accuracy')
        
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        probe_results[layer_idx] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "scores": scores.tolist(),
        }
        
        # Indicator for good probes
        indicator = "***" if mean_acc > 0.7 else ""
        print(f"  Layer {layer_idx:2d}: {mean_acc:.1%} (+/- {std_acc:.1%}) {indicator}")
    
    # ========================================
    # FIND BEST LAYERS
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYSIS: Which Layers Encode Belief State?")
    print(f"{'='*60}")
    
    # Sort by accuracy
    sorted_layers = sorted(probe_results.items(), key=lambda x: x[1]["mean_accuracy"], reverse=True)
    
    print("\nTop layers for belief state probing:")
    for layer_idx, data in sorted_layers[:5]:
        print(f"  Layer {layer_idx}: {data['mean_accuracy']:.1%}")
    
    # Check if critical ToM layers (32-34) are good for belief probing
    critical_layers = [32, 33, 34]
    print(f"\nCritical ToM layers (32-34) probe accuracy:")
    for layer_idx in critical_layers:
        if layer_idx in probe_results:
            print(f"  Layer {layer_idx}: {probe_results[layer_idx]['mean_accuracy']:.1%}")
    
    # ========================================
    # NEURON-LEVEL ANALYSIS (for top layer)
    # ========================================
    print(f"\n{'='*60}")
    print("NEURON-LEVEL ANALYSIS")
    print(f"{'='*60}")
    
    # Find top layer
    top_layer = sorted_layers[0][0]
    print(f"\nAnalyzing individual neurons in layer {top_layer}...")
    
    X_top = X_by_layer[top_layer]
    
    # Find most discriminative neurons
    neuron_scores = []
    for neuron_idx in range(min(100, X_top.shape[1])):  # Check first 100 neurons
        neuron_vals = X_top[:, neuron_idx]
        
        # Simple: difference in means
        false_belief_mean = neuron_vals[y == 0].mean()
        true_belief_mean = neuron_vals[y == 1].mean()
        diff = abs(true_belief_mean - false_belief_mean)
        
        neuron_scores.append({
            "neuron": neuron_idx,
            "diff": diff,
            "false_belief_mean": false_belief_mean,
            "true_belief_mean": true_belief_mean,
        })
    
    # Sort by difference
    neuron_scores.sort(key=lambda x: x["diff"], reverse=True)
    
    print(f"\nTop discriminative neurons in layer {top_layer}:")
    for n in neuron_scores[:10]:
        direction = "T>F" if n["true_belief_mean"] > n["false_belief_mean"] else "F>T"
        print(f"  Neuron {n['neuron']:4d}: diff={n['diff']:.3f} ({direction})")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_scenarios": len(scenarios),
            "layers_probed": layers_to_probe,
        },
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "top_layers": [(l, d["mean_accuracy"]) for l, d in sorted_layers[:5]],
        "top_layer_neurons": neuron_scores[:20],
    }
    
    output_path = RESULTS_DIR / "step11_mlp_probing.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: Probe accuracy by layer
    fig, ax = plt.subplots(figsize=(12, 5))
    
    layers = list(probe_results.keys())
    accuracies = [probe_results[l]["mean_accuracy"] * 100 for l in layers]
    stds = [probe_results[l]["std_accuracy"] * 100 for l in layers]
    
    colors = ['coral' if l in [32, 33, 34] else 'steelblue' for l in layers]
    
    ax.bar(range(len(layers)), accuracies, yerr=stds, color=colors, edgecolor='black', capsize=3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% threshold')
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy (%)", fontsize=12)
    ax.set_title("Can MLPs Distinguish True vs False Belief?", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.legend()
    ax.set_ylim(40, 100)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step11_probe_accuracy.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Top neurons
    fig, ax = plt.subplots(figsize=(10, 5))
    
    neurons = [n["neuron"] for n in neuron_scores[:15]]
    diffs = [n["diff"] for n in neuron_scores[:15]]
    
    ax.barh(range(len(neurons)), diffs, color='purple', edgecolor='black')
    ax.set_yticks(range(len(neurons)))
    ax.set_yticklabels([f"Neuron {n}" for n in neurons])
    ax.set_xlabel("Activation Difference (|True - False|)", fontsize=12)
    ax.set_title(f"Most Discriminative Neurons in Layer {top_layer}", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step11_top_neurons.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 11 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

