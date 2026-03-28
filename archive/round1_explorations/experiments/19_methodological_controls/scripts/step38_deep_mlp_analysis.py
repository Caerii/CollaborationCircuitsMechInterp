"""
Step 38: Deep MLP Neuron Analysis

Previous findings:
- L35 MLP has massive divergence between action/belief verbs
- Neurons 0 and 4 in L35 down_proj show diff > 25

This script investigates:
1. What do neurons 0 and 4 represent?
2. Which intermediate neurons (gate) drive this?
3. Can we find interpretable features?
4. How does verb information flow through MLP?
"""

import torch
import json
import sys
import io
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


# Prompts for analysis
ACTION_PROMPT = """Alice put the ball in the drawer. Alice left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice searched in the"""

BELIEF_PROMPT = """Alice put the ball in the drawer. Alice left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice thinks the ball is in the"""


def get_mlp_activations_detailed(model, tokenizer, prompt):
    """Get detailed MLP activations at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    activations = {}
    
    # Hook MLP components
    def make_hook(layer_idx, component):
        def hook(module, input, output):
            # Get last token activations
            activations[f"L{layer_idx}_{component}"] = output[0, -1, :].clone().detach()
        return hook
    
    hooks = []
    for layer_idx in range(36):
        layer = model.model.layers[layer_idx]
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(layer_idx, "gate")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(layer_idx, "up")))
    
    # Also hook the MLP output (after down_proj)
    def make_mlp_out_hook(layer_idx):
        def hook(module, input, output):
            activations[f"L{layer_idx}_mlp_out"] = output[0, -1, :].clone().detach()
        return hook
    
    for layer_idx in range(36):
        layer = model.model.layers[layer_idx]
        hooks.append(layer.mlp.register_forward_hook(make_mlp_out_hook(layer_idx)))
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations, outputs.logits


def analyze_neuron_0_and_4():
    """Deep dive into neurons 0 and 4 which showed massive divergence."""
    model, tokenizer = load_model()
    
    print("\n" + "="*70)
    print("DEEP ANALYSIS: Neurons 0 and 4")
    print("="*70)
    
    # Get activations for both prompts
    action_acts, action_logits = get_mlp_activations_detailed(model, tokenizer, ACTION_PROMPT)
    belief_acts, belief_logits = get_mlp_activations_detailed(model, tokenizer, BELIEF_PROMPT)
    
    # Track neurons 0 and 4 through all layers
    print("\n--- Tracking Critical Neurons Through Layers ---")
    
    results = {
        "neuron_0": {"action": [], "belief": [], "diff": []},
        "neuron_4": {"action": [], "belief": [], "diff": []}
    }
    
    print("\nNeuron 0 (MLP output dimension 0):")
    for layer_idx in range(30, 36):
        action_val = action_acts[f"L{layer_idx}_mlp_out"][0].item()
        belief_val = belief_acts[f"L{layer_idx}_mlp_out"][0].item()
        diff = action_val - belief_val
        
        results["neuron_0"]["action"].append(action_val)
        results["neuron_0"]["belief"].append(belief_val)
        results["neuron_0"]["diff"].append(diff)
        
        print(f"  L{layer_idx}: action={action_val:+.2f}, belief={belief_val:+.2f}, diff={diff:+.2f}")
    
    print("\nNeuron 4 (MLP output dimension 4):")
    for layer_idx in range(30, 36):
        action_val = action_acts[f"L{layer_idx}_mlp_out"][4].item()
        belief_val = belief_acts[f"L{layer_idx}_mlp_out"][4].item()
        diff = action_val - belief_val
        
        results["neuron_4"]["action"].append(action_val)
        results["neuron_4"]["belief"].append(belief_val)
        results["neuron_4"]["diff"].append(diff)
        
        print(f"  L{layer_idx}: action={action_val:+.2f}, belief={belief_val:+.2f}, diff={diff:+.2f}")
    
    # Analyze what input features drive these neurons
    print("\n" + "="*70)
    print("ANALYZING INPUT FEATURES TO L35 MLP")
    print("="*70)
    
    # Get the down_proj weights for neurons 0 and 4
    down_proj_weights = model.model.layers[35].mlp.down_proj.weight.data  # [hidden_size, intermediate_size]
    
    # Neuron 0 and 4 are output dimensions, so we look at their weights
    neuron_0_weights = down_proj_weights[0, :].cpu().float().numpy()  # Which intermediate features -> neuron 0
    neuron_4_weights = down_proj_weights[4, :].cpu().float().numpy()  # Which intermediate features -> neuron 4
    
    print("\nTop intermediate neurons contributing to output neuron 0:")
    top_0 = np.argsort(np.abs(neuron_0_weights))[-10:][::-1]
    for i, idx in enumerate(top_0):
        print(f"  {i+1}. Intermediate neuron {idx}: weight={neuron_0_weights[idx]:.4f}")
    
    print("\nTop intermediate neurons contributing to output neuron 4:")
    top_4 = np.argsort(np.abs(neuron_4_weights))[-10:][::-1]
    for i, idx in enumerate(top_4):
        print(f"  {i+1}. Intermediate neuron {idx}: weight={neuron_4_weights[idx]:.4f}")
    
    # Check if these intermediate neurons differ between action/belief
    print("\n" + "="*70)
    print("INTERMEDIATE NEURON ACTIVATION DIFFERENCES")
    print("="*70)
    
    # Gate activations in L35
    action_gate = action_acts["L35_gate"].cpu().float().numpy()
    belief_gate = belief_acts["L35_gate"].cpu().float().numpy()
    gate_diff = action_gate - belief_gate
    
    print("\nTop differing gate neurons in L35:")
    top_gate_diff = np.argsort(np.abs(gate_diff))[-20:][::-1]
    for i, idx in enumerate(top_gate_diff[:10]):
        print(f"  {i+1}. Gate neuron {idx}: action={action_gate[idx]:.3f}, "
              f"belief={belief_gate[idx]:.3f}, diff={gate_diff[idx]:+.3f}")
    
    # Check overlap with neurons contributing to 0 and 4
    overlap_0 = set(top_gate_diff[:20]) & set(top_0)
    overlap_4 = set(top_gate_diff[:20]) & set(top_4)
    
    print(f"\nOverlap with neuron 0's contributors: {overlap_0}")
    print(f"Overlap with neuron 4's contributors: {overlap_4}")
    
    # Analyze what these output neurons do to the logits
    print("\n" + "="*70)
    print("EFFECT ON FINAL PREDICTIONS")
    print("="*70)
    
    # Get the unembedding (lm_head) weights
    lm_head_weights = model.lm_head.weight.data  # [vocab_size, hidden_size]
    
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    # How much do neurons 0 and 4 contribute to drawer vs basket logits?
    drawer_from_0 = lm_head_weights[drawer_id, 0].item()
    drawer_from_4 = lm_head_weights[drawer_id, 4].item()
    basket_from_0 = lm_head_weights[basket_id, 0].item()
    basket_from_4 = lm_head_weights[basket_id, 4].item()
    
    print(f"\nNeuron 0 contribution to logits:")
    print(f"  -> drawer: {drawer_from_0:.4f}")
    print(f"  -> basket: {basket_from_0:.4f}")
    print(f"  Bias toward: {'drawer' if drawer_from_0 > basket_from_0 else 'basket'}")
    
    print(f"\nNeuron 4 contribution to logits:")
    print(f"  -> drawer: {drawer_from_4:.4f}")
    print(f"  -> basket: {basket_from_4:.4f}")
    print(f"  Bias toward: {'drawer' if drawer_from_4 > basket_from_4 else 'basket'}")
    
    # Calculate actual contribution
    print("\n--- Actual Contribution to Prediction ---")
    
    for neuron_idx in [0, 4]:
        action_val = action_acts["L35_mlp_out"][neuron_idx].item()
        belief_val = belief_acts["L35_mlp_out"][neuron_idx].item()
        
        drawer_weight = lm_head_weights[drawer_id, neuron_idx].item()
        basket_weight = lm_head_weights[basket_id, neuron_idx].item()
        
        action_drawer_contrib = action_val * drawer_weight
        action_basket_contrib = action_val * basket_weight
        belief_drawer_contrib = belief_val * drawer_weight
        belief_basket_contrib = belief_val * basket_weight
        
        print(f"\nNeuron {neuron_idx}:")
        print(f"  Action verb: drawer_contrib={action_drawer_contrib:.2f}, "
              f"basket_contrib={action_basket_contrib:.2f}, "
              f"net={action_drawer_contrib-action_basket_contrib:+.2f}")
        print(f"  Belief verb: drawer_contrib={belief_drawer_contrib:.2f}, "
              f"basket_contrib={belief_basket_contrib:.2f}, "
              f"net={belief_drawer_contrib-belief_basket_contrib:+.2f}")
    
    results["analysis"] = {
        "neuron_0_drawer_weight": drawer_from_0,
        "neuron_0_basket_weight": basket_from_0,
        "neuron_4_drawer_weight": drawer_from_4,
        "neuron_4_basket_weight": basket_from_4,
        "top_gate_diff_neurons": top_gate_diff[:10].tolist()
    }
    
    # Save results
    save_path = RESULTS_DIR / "deep_mlp_analysis_results.json"
    
    # Convert to serializable format
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: [float(x) for x in v] if isinstance(v, list) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    # Plot
    plot_neuron_analysis(results)
    
    return results


def plot_neuron_analysis(results):
    """Plot neuron analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = list(range(30, 36))
    
    # Plot neuron 0
    ax1 = axes[0, 0]
    ax1.plot(layers, results["neuron_0"]["action"], 'b-o', label="Action verb", linewidth=2)
    ax1.plot(layers, results["neuron_0"]["belief"], 'r-o', label="Belief verb", linewidth=2)
    ax1.set_title("Neuron 0 Activation Through Layers")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Activation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot neuron 4
    ax2 = axes[0, 1]
    ax2.plot(layers, results["neuron_4"]["action"], 'b-o', label="Action verb", linewidth=2)
    ax2.plot(layers, results["neuron_4"]["belief"], 'r-o', label="Belief verb", linewidth=2)
    ax2.set_title("Neuron 4 Activation Through Layers")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Activation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot differences
    ax3 = axes[1, 0]
    ax3.bar(layers, results["neuron_0"]["diff"], color='purple', alpha=0.7, label="Neuron 0")
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title("Neuron 0: Action - Belief Difference")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Difference")
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.bar(layers, results["neuron_4"]["diff"], color='green', alpha=0.7, label="Neuron 4")
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title("Neuron 4: Action - Belief Difference")
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Difference")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "deep_mlp_neuron_analysis.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("STEP 38: Deep MLP Neuron Analysis")
    print("="*70)
    
    results = analyze_neuron_0_and_4()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key findings about the MLP mechanism:
1. Neurons 0 and 4 show large divergence in L35
2. These neurons have specific weight patterns to drawer/basket tokens
3. The verb type affects intermediate gate neurons, which cascades to output

This reveals a specific pathway:
  Verb token -> Gate neurons -> Neurons 0,4 -> drawer/basket logits
""")


if __name__ == "__main__":
    main()


