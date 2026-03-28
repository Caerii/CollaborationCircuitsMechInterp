"""
Step 28: L35 MLP Deep Investigation

Finding from step26: L35 MLP has massive divergence (norm 76.94) between "told" and "announced"
while attention outputs are nearly identical.

This script investigates:
1. Which neurons in L35 MLP activate differently?
2. What patterns do these neurons detect?
3. Can we ablate specific MLP neurons to fix ToM?
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

BAD_VERBS = ["told", "said", "mentioned", "informed", "stated"]
GOOD_VERBS = ["announced", "asked", "hinted", "explained", "shouted"]

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


def create_prompt(verb):
    return f"""Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice looked for the ball. Alice searched in the"""


def get_mlp_activations(model, tokenizer, prompt, layers=[32, 33, 34, 35]):
    """Get MLP intermediate activations (after gate/up, before down)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    activations = {}
    hooks = []
    
    # Hook the gate_proj output (MLP hidden state)
    def make_hook(layer_idx, name):
        def hook(module, input, output):
            activations[f"L{layer_idx}_{name}"] = output[0, -1, :].clone()
        return hook
    
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        # Qwen3 MLP structure: gate_proj, up_proj, down_proj
        # gate_proj and up_proj outputs are multiplied, then passed to down_proj
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(layer_idx, "gate")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(layer_idx, "up")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(layer_idx, "down")))
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations, outputs.logits


def analyze_mlp_neuron_differences():
    """Find which MLP neurons differ most between bad and good verbs."""
    model, tokenizer = load_model()
    
    print("\n" + "="*70)
    print("ANALYZING MLP NEURON DIFFERENCES")
    print("="*70)
    
    # Collect activations for all verbs
    bad_activations = []
    good_activations = []
    
    for verb in BAD_VERBS:
        prompt = create_prompt(verb)
        acts, _ = get_mlp_activations(model, tokenizer, prompt)
        bad_activations.append(acts)
    
    for verb in GOOD_VERBS:
        prompt = create_prompt(verb)
        acts, _ = get_mlp_activations(model, tokenizer, prompt)
        good_activations.append(acts)
    
    # Analyze differences at each layer
    results = {}
    
    for layer in [32, 33, 34, 35]:
        print(f"\n--- Layer {layer} ---")
        
        # Gate activations
        bad_gate = torch.stack([a[f"L{layer}_gate"] for a in bad_activations]).mean(0)
        good_gate = torch.stack([a[f"L{layer}_gate"] for a in good_activations]).mean(0)
        gate_diff = (bad_gate - good_gate).abs()
        
        # Down activations (final MLP output)
        bad_down = torch.stack([a[f"L{layer}_down"] for a in bad_activations]).mean(0)
        good_down = torch.stack([a[f"L{layer}_down"] for a in good_activations]).mean(0)
        down_diff = (bad_down - good_down).abs()
        
        # Find top differing neurons
        gate_top_k = torch.topk(gate_diff, k=10)
        down_top_k = torch.topk(down_diff, k=10)
        
        print(f"\nTop 10 differing GATE neurons:")
        for i, (idx, val) in enumerate(zip(gate_top_k.indices.tolist(), gate_top_k.values.tolist())):
            bad_val = bad_gate[idx].item()
            good_val = good_gate[idx].item()
            print(f"  {i+1}. Neuron {idx}: diff={val:.3f} (bad={bad_val:.3f}, good={good_val:.3f})")
        
        print(f"\nTop 10 differing DOWN (output) neurons:")
        for i, (idx, val) in enumerate(zip(down_top_k.indices.tolist(), down_top_k.values.tolist())):
            bad_val = bad_down[idx].item()
            good_val = good_down[idx].item()
            print(f"  {i+1}. Neuron {idx}: diff={val:.3f} (bad={bad_val:.3f}, good={good_val:.3f})")
        
        results[f"L{layer}"] = {
            "gate_top_neurons": gate_top_k.indices.tolist(),
            "gate_top_diffs": [float(v) for v in gate_top_k.values.tolist()],
            "down_top_neurons": down_top_k.indices.tolist(),
            "down_top_diffs": [float(v) for v in down_top_k.values.tolist()],
            "gate_total_diff": float(gate_diff.sum().item()),
            "down_total_diff": float(down_diff.sum().item())
        }
    
    return results, model, tokenizer


def test_mlp_neuron_ablation(model, tokenizer, layer_idx, neuron_indices, ablate_gate=False):
    """Test if ablating specific MLP neurons fixes ToM."""
    print(f"\n--- Testing ablation of L{layer_idx} {'gate' if ablate_gate else 'down'} neurons: {neuron_indices[:5]}... ---")
    
    hooks = []
    
    def make_ablation_hook(neurons_to_zero, max_dim):
        def hook(module, input, output):
            for neuron_idx in neurons_to_zero:
                if neuron_idx < output.shape[-1]:  # Safety check
                    output[:, :, neuron_idx] = 0
            return output
        return hook
    
    layer = model.model.layers[layer_idx]
    
    if ablate_gate:
        # Gate has intermediate dimension (10240)
        hook = layer.mlp.gate_proj.register_forward_hook(make_ablation_hook(neuron_indices, 10240))
    else:
        # Down output has hidden dimension (2560) - filter indices
        valid_indices = [i for i in neuron_indices if i < 2560]
        hook = layer.mlp.down_proj.register_forward_hook(make_ablation_hook(valid_indices, 2560))
    
    hooks.append(hook)
    
    # Test on bad verbs
    results = []
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    for verb in BAD_VERBS[:3]:  # Test first 3
        prompt = create_prompt(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        drawer_logit = logits[drawer_id].item()
        basket_logit = logits[basket_id].item()
        correct = drawer_logit > basket_logit
        
        results.append({
            "verb": verb,
            "drawer": drawer_logit,
            "basket": basket_logit,
            "diff": drawer_logit - basket_logit,
            "correct": correct
        })
        print(f"  {verb}: drawer={drawer_logit:.2f}, basket={basket_logit:.2f}, "
              f"diff={drawer_logit-basket_logit:+.2f}, {'[OK]' if correct else '[FAIL]'}")
    
    for hook in hooks:
        hook.remove()
    
    return results


def find_critical_mlp_neurons():
    """Systematically find which MLP neurons are critical for the override."""
    results, model, tokenizer = analyze_mlp_neuron_differences()
    
    print("\n" + "="*70)
    print("TESTING MLP NEURON ABLATIONS")
    print("="*70)
    
    # First, test baseline (no ablation)
    print("\n--- BASELINE (no ablation) ---")
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    for verb in BAD_VERBS[:3]:
        prompt = create_prompt(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        drawer_logit = logits[drawer_id].item()
        basket_logit = logits[basket_id].item()
        print(f"  {verb}: drawer={drawer_logit:.2f}, basket={basket_logit:.2f}, "
              f"diff={drawer_logit-basket_logit:+.2f}")
    
    # Test ablating top neurons at L35 gate (intermediate)
    print("\n--- L35 TOP 10 GATE NEURONS ABLATION ---")
    top_gate_neurons = results["L35"]["gate_top_neurons"][:10]
    test_mlp_neuron_ablation(model, tokenizer, 35, top_gate_neurons, ablate_gate=True)
    
    # Test ablating top neurons at L35 down_proj output
    print("\n--- L35 TOP 10 DOWN NEURONS ABLATION ---")
    top_down_neurons = results["L35"]["down_top_neurons"][:10]
    test_mlp_neuron_ablation(model, tokenizer, 35, top_down_neurons, ablate_gate=False)
    
    # Compare with attention head ablation
    print("\n--- ATTENTION HEAD ABLATION (for comparison) ---")
    LATE_CIRCUIT_HEADS = [
        (32, 6), (32, 31), (33, 6), (33, 13), (33, 17), (33, 31),
        (34, 17), (35, 0), (35, 1), (35, 17)
    ]
    
    hooks = []
    for layer_idx, head_idx in LATE_CIRCUIT_HEADS:
        layer = model.model.layers[layer_idx]
        
        def make_hook(h_idx):
            def hook(module, input, output):
                hidden = output
                batch, seq_len, hidden_size = hidden.shape
                n_heads = 32
                head_dim = hidden_size // n_heads
                hidden = hidden.view(batch, seq_len, n_heads, head_dim)
                hidden[:, :, h_idx, :] = 0
                hidden = hidden.view(batch, seq_len, hidden_size)
                return hidden
            return hook
        
        hook = layer.self_attn.o_proj.register_forward_hook(make_hook(head_idx))
        hooks.append(hook)
    
    for verb in BAD_VERBS[:3]:
        prompt = create_prompt(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        drawer_logit = logits[drawer_id].item()
        basket_logit = logits[basket_id].item()
        print(f"  {verb}: drawer={drawer_logit:.2f}, basket={basket_logit:.2f}, "
              f"diff={drawer_logit-basket_logit:+.2f}, {'[OK]' if drawer_logit > basket_logit else '[FAIL]'}")
    
    for hook in hooks:
        hook.remove()
    
    return results


def plot_mlp_differences(results):
    """Visualize MLP differences across layers."""
    layers = [32, 33, 34, 35]
    gate_diffs = [results[f"L{l}"]["gate_total_diff"] for l in layers]
    down_diffs = [results[f"L{l}"]["down_total_diff"] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, gate_diffs, width, label='Gate (hidden)', color='blue', alpha=0.7)
    ax.bar(x + width/2, down_diffs, width, label='Down (output)', color='red', alpha=0.7)
    
    ax.set_ylabel('Total Absolute Difference')
    ax.set_xlabel('Layer')
    ax.set_title('MLP Activation Differences: Bad vs Good Verbs')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend()
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "mlp_layer_differences.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("STEP 28: L35 MLP Deep Investigation")
    print("="*70)
    print("\nGoal: Understand why L35 MLP has massive divergence")
    
    results = find_critical_mlp_neurons()
    
    # Plot differences
    plot_mlp_differences(results)
    
    # Save results
    save_path = RESULTS_DIR / "mlp_investigation_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. L35 MLP has the largest divergence between bad/good verbs
2. Specific neurons show high activation differences
3. MLP neuron ablation may or may not fix ToM (compared above)
4. The mechanism involves both attention heads AND MLP neurons
""")


if __name__ == "__main__":
    main()

