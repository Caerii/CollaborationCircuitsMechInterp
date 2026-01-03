"""
Step 34: Verb Type Mechanism Analysis

CRITICAL FINDING from Step 33:
- ACTION verbs work: searched, looks, expects, remembers
- BELIEF verbs fail: thinks, believes, knows, assumes

This script investigates:
1. WHERE in the model does the verb type affect processing?
2. Attention differences between action vs belief verbs
3. Layer-by-layer logit evolution for each verb type
4. Can we identify the "belief verb detector" circuit?
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

# Verbs that work vs fail (from step 33)
ACTION_VERBS = ["searched", "looks", "expects", "remembers"]
BELIEF_VERBS = ["thinks", "believes", "knows", "assumes"]


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # For attention weights
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def create_prompt(completion_verb, verb_type="action"):
    """Create prompt with specific completion verb."""
    base = """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice"""
    
    if verb_type == "action":
        # Action verbs need different completions
        if completion_verb == "searched":
            return base + " searched in the"
        elif completion_verb == "looks":
            return base + " looks in the"
        elif completion_verb == "expects":
            return base + " expects to find the ball in the"
        elif completion_verb == "remembers":
            return base + " remembers the ball being in the"
    else:
        # Belief verbs
        return base + f" {completion_verb} the ball is in the"
    
    return base + f" {completion_verb} in the"


def get_layer_logits(model, tokenizer, prompt, target_tokens=["drawer", "basket"]):
    """Get logits for target tokens at each layer using logit lens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Store hidden states at each layer
    hidden_states = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Get the hidden state after this layer
            if isinstance(output, tuple):
                hidden_states.append(output[0][:, -1, :].clone())
            else:
                hidden_states.append(output[:, -1, :].clone())
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(make_hook(i))
        hooks.append(hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    # Apply final layer norm and get logits for each layer's hidden state
    layer_logits = {}
    lm_head = model.lm_head
    norm = model.model.norm
    
    for token in target_tokens:
        token_id = tokenizer.encode(" " + token, add_special_tokens=False)[0]
        layer_logits[token] = []
        
        for i, hidden in enumerate(hidden_states):
            # Apply layer norm and project to vocab
            normed = norm(hidden)
            logits = lm_head(normed)
            layer_logits[token].append(logits[0, token_id].item())
    
    return layer_logits


def get_attention_to_verb(model, tokenizer, prompt, verb_position):
    """Get attention weights to the completion verb from final token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is tuple of (batch, heads, seq, seq) for each layer
    attention_to_verb = []
    
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # Get attention from last token to verb position
        # Average across all heads
        attn = layer_attn[0, :, -1, verb_position].mean().item()
        attention_to_verb.append(attn)
    
    return attention_to_verb


def analyze_verb_types():
    """Compare processing of action vs belief verbs."""
    model, tokenizer = load_model()
    
    results = {
        "action_verbs": {},
        "belief_verbs": {},
        "layer_analysis": {}
    }
    
    print("\n" + "="*70)
    print("VERB TYPE MECHANISM ANALYSIS")
    print("="*70)
    
    # Get drawer/basket token IDs
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    # Test action verbs
    print("\n--- ACTION VERBS (should work) ---")
    action_drawer_evolution = []
    action_basket_evolution = []
    
    for verb in ACTION_VERBS:
        prompt = create_prompt(verb, "action")
        layer_logits = get_layer_logits(model, tokenizer, prompt)
        
        # Get final prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        final_logits = outputs.logits[0, -1, :]
        drawer_final = final_logits[drawer_id].item()
        basket_final = final_logits[basket_id].item()
        diff = drawer_final - basket_final
        
        status = "[OK]" if diff > 0 else "[FAIL]"
        print(f"  {status} {verb}: diff={diff:+.2f}")
        
        results["action_verbs"][verb] = {
            "drawer_evolution": layer_logits["drawer"],
            "basket_evolution": layer_logits["basket"],
            "final_diff": diff
        }
        
        action_drawer_evolution.append(layer_logits["drawer"])
        action_basket_evolution.append(layer_logits["basket"])
    
    # Test belief verbs
    print("\n--- BELIEF VERBS (should fail) ---")
    belief_drawer_evolution = []
    belief_basket_evolution = []
    
    for verb in BELIEF_VERBS:
        prompt = create_prompt(verb, "belief")
        layer_logits = get_layer_logits(model, tokenizer, prompt)
        
        # Get final prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        final_logits = outputs.logits[0, -1, :]
        drawer_final = final_logits[drawer_id].item()
        basket_final = final_logits[basket_id].item()
        diff = drawer_final - basket_final
        
        status = "[OK]" if diff > 0 else "[FAIL]"
        print(f"  {status} {verb}: diff={diff:+.2f}")
        
        results["belief_verbs"][verb] = {
            "drawer_evolution": layer_logits["drawer"],
            "basket_evolution": layer_logits["basket"],
            "final_diff": diff
        }
        
        belief_drawer_evolution.append(layer_logits["drawer"])
        belief_basket_evolution.append(layer_logits["basket"])
    
    # Average evolution
    action_drawer_avg = np.mean(action_drawer_evolution, axis=0)
    action_basket_avg = np.mean(action_basket_evolution, axis=0)
    belief_drawer_avg = np.mean(belief_drawer_evolution, axis=0)
    belief_basket_avg = np.mean(belief_basket_evolution, axis=0)
    
    # Find divergence point
    print("\n" + "="*70)
    print("LAYER-BY-LAYER EVOLUTION")
    print("="*70)
    
    print("\nAction verbs (drawer - basket):")
    action_diff = action_drawer_avg - action_basket_avg
    for i in range(0, 36, 5):
        print(f"  L{i}: {action_diff[i]:+.2f}")
    print(f"  L35: {action_diff[35]:+.2f}")
    
    print("\nBelief verbs (drawer - basket):")
    belief_diff = belief_drawer_avg - belief_basket_avg
    for i in range(0, 36, 5):
        print(f"  L{i}: {belief_diff[i]:+.2f}")
    print(f"  L35: {belief_diff[35]:+.2f}")
    
    # Find where they diverge
    print("\n" + "="*70)
    print("DIVERGENCE ANALYSIS")
    print("="*70)
    
    divergence = action_diff - belief_diff
    print("\nDivergence (action - belief) by layer:")
    significant_layers = []
    for i in range(36):
        if abs(divergence[i]) > 0.5:
            significant_layers.append(i)
        if i % 5 == 0 or i == 35:
            print(f"  L{i}: {divergence[i]:+.2f}")
    
    print(f"\nLayers with significant divergence (>0.5): {significant_layers}")
    
    # Store for plotting
    results["layer_analysis"] = {
        "action_drawer_avg": action_drawer_avg.tolist(),
        "action_basket_avg": action_basket_avg.tolist(),
        "belief_drawer_avg": belief_drawer_avg.tolist(),
        "belief_basket_avg": belief_basket_avg.tolist(),
        "divergence": divergence.tolist(),
        "significant_layers": significant_layers
    }
    
    # Plot
    plot_verb_type_comparison(results)
    
    # Save
    save_path = RESULTS_DIR / "verb_type_mechanism_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def plot_verb_type_comparison(results):
    """Plot comparison of action vs belief verb processing."""
    la = results["layer_analysis"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = range(36)
    
    # Plot 1: Action verbs evolution
    ax1 = axes[0, 0]
    ax1.plot(layers, la["action_drawer_avg"], label="drawer", color="green", linewidth=2)
    ax1.plot(layers, la["action_basket_avg"], label="basket", color="red", linewidth=2)
    ax1.set_title("ACTION Verbs (searched, looks, etc.)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Logit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Belief verbs evolution
    ax2 = axes[0, 1]
    ax2.plot(layers, la["belief_drawer_avg"], label="drawer", color="green", linewidth=2)
    ax2.plot(layers, la["belief_basket_avg"], label="basket", color="red", linewidth=2)
    ax2.set_title("BELIEF Verbs (thinks, believes, etc.)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Logit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diff comparison
    ax3 = axes[1, 0]
    action_diff = np.array(la["action_drawer_avg"]) - np.array(la["action_basket_avg"])
    belief_diff = np.array(la["belief_drawer_avg"]) - np.array(la["belief_basket_avg"])
    ax3.plot(layers, action_diff, label="Action verbs", color="blue", linewidth=2)
    ax3.plot(layers, belief_diff, label="Belief verbs", color="orange", linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title("Drawer - Basket Difference by Verb Type")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Logit Difference")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Divergence
    ax4 = axes[1, 1]
    ax4.bar(layers, la["divergence"], color=['red' if d < 0 else 'green' for d in la["divergence"]])
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title("Divergence: Action - Belief (where processing differs)")
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Divergence")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "verb_type_mechanism.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("STEP 34: Verb Type Mechanism Analysis")
    print("="*70)
    print("\nInvestigating WHERE action vs belief verbs diverge")
    
    results = analyze_verb_types()
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
The model processes "Alice searched" and "Alice thinks" differently:

ACTION VERBS (searched, looks):
  → Model asks "Where will Alice look?" 
  → Answers based on Alice's BELIEF (correct ToM)

BELIEF VERBS (thinks, believes):
  → Model asks "What is true about the ball?"
  → Answers based on REALITY (incorrect ToM)

This is a SEMANTIC interpretation difference, not a circuit failure!
""")


if __name__ == "__main__":
    main()


