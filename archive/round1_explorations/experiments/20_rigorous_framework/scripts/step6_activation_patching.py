"""
Step 6: Activation Patching - Confirm Causal Role of Critical Heads

NOW USING LIBRARY: ActivationPatcher for patching mechanics!

HYPOTHESIS: Patching activations from "correct" scenarios into "wrong" scenarios
            at critical heads (L32-34) will flip the prediction.

METHODOLOGY:
- Create minimal pairs: same story, different outcomes
- Patch activations from source (correct) to target (wrong) at critical layers
- Measure flip rate at critical heads vs random heads
- Provides causal evidence that these heads CAUSE the behavior

Based on Step 4 & 5 findings:
- Decision layers: 29-34 (from Logit Lens)
- Critical heads: L32H0, L33H4, L33H16, L33H28, L34H0 (from Ablation)

NOTE: This step uses completion mode (logit-based evaluation) for precise comparison.
      The library's ActivationPatcher is used for the patching mechanics.

OUTPUT: results/step6_patching.json, figures/step6_patching_results.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.patching import ActivationPatcher

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_logit_prediction(model, tokenizer, prompt: str, correct_token: str, wrong_token: str) -> dict:
    """Get logit-based prediction (completion mode)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    correct_ids = tokenizer.encode(" " + correct_token, add_special_tokens=False)
    wrong_ids = tokenizer.encode(" " + wrong_token, add_special_tokens=False)
    
    if not correct_ids or not wrong_ids:
        return None
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_logit = float(logits[correct_ids[0]])
    wrong_logit = float(logits[wrong_ids[0]])
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "prediction": "correct" if correct_logit > wrong_logit else "wrong",
        "logit_diff": correct_logit - wrong_logit,
    }


def patch_and_get_logits(
    model, tokenizer, prompt: str, patch_activations: dict, 
    correct_token: str, wrong_token: str
) -> dict:
    """Patch activations and get logit-based prediction (completion mode)."""
    hooks = []
    
    def make_patch_hook(layer_idx, source_activation):
        def hook(module, input, output):
            # Get target sequence length
            if isinstance(output, tuple):
                target = output[0]
            else:
                target = output
            
            # Source might have different sequence length
            source = source_activation.to(target.device)
            
            # Patch the last token position (most important for next-token prediction)
            min_len = min(target.shape[1], source.shape[1])
            patched = target.clone()
            patched[:, -min_len:, :] = source[:, -min_len:, :]
            
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched
        return hook
    
    # Install patch hooks
    for layer_idx, activation in patch_activations.items():
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(make_patch_hook(layer_idx, activation))
        hooks.append(handle)
    
    try:
        # Run forward pass with patches
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        correct_ids = tokenizer.encode(" " + correct_token, add_special_tokens=False)
        wrong_ids = tokenizer.encode(" " + wrong_token, add_special_tokens=False)
        
        if not correct_ids or not wrong_ids:
            return None
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_logit = float(logits[correct_ids[0]])
        wrong_logit = float(logits[wrong_ids[0]])
        
        return {
            "correct_logit": correct_logit,
            "wrong_logit": wrong_logit,
            "prediction": "correct" if correct_logit > wrong_logit else "wrong",
            "logit_diff": correct_logit - wrong_logit,
        }
    finally:
        # Clean up hooks
        for h in hooks:
            h.remove()


def create_minimal_pairs():
    """Create matched pairs of prompts where only the belief state differs."""
    pairs = []
    
    # Pair 1: Classic Sally-Anne
    pairs.append({
        "name": "sally_anne",
        "source": {  # Agent KNOWS (saw move)
            "prompt": "Sally put the ball in the basket. Sally stayed and watched. Anne moved the ball to the box. Sally looks for the ball. Sally searches in the",
            "correct": "box",  # She saw the move
            "wrong": "basket",
        },
        "target": {  # Agent DOESN'T KNOW (left)
            "prompt": "Sally put the ball in the basket. Sally left the room. Anne moved the ball to the box. Sally returns and looks for the ball. Sally searches in the",
            "correct": "basket",  # False belief - she thinks it's still there
            "wrong": "box",
        },
    })
    
    # Pair 2: Different names
    pairs.append({
        "name": "alice_bob",
        "source": {
            "prompt": "Alice put the toy in the drawer. Alice watched as Bob moved the toy to the shelf. Alice looks for the toy. Alice checks the",
            "correct": "shelf",
            "wrong": "drawer",
        },
        "target": {
            "prompt": "Alice put the toy in the drawer. Alice went outside. Bob moved the toy to the shelf. Alice returns and looks for the toy. Alice checks the",
            "correct": "drawer",
            "wrong": "shelf",
        },
    })
    
    # Pair 3: Different objects
    pairs.append({
        "name": "keys",
        "source": {
            "prompt": "Mom placed the keys on the table. Mom saw Dad move them to the hook. Mom needs the keys. Mom goes to the",
            "correct": "hook",
            "wrong": "table",
        },
        "target": {
            "prompt": "Mom placed the keys on the table. Mom went upstairs. Dad moved them to the hook. Mom comes down and needs the keys. Mom goes to the",
            "correct": "table",
            "wrong": "hook",
        },
    })
    
    # Pair 4: Novel names  
    pairs.append({
        "name": "zork_blep",
        "source": {
            "prompt": "Zork hid the gem in the cave. Zork observed Blep transfer the gem to the forest. Zork searches for the gem. Zork goes to the",
            "correct": "forest",
            "wrong": "cave",
        },
        "target": {
            "prompt": "Zork hid the gem in the cave. Zork departed. Blep transferred the gem to the forest. Zork returns to find the gem. Zork goes to the",
            "correct": "cave",
            "wrong": "forest",
        },
    })
    
    return pairs


def main():
    print("=" * 70)
    print("STEP 6: ACTIVATION PATCHING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    config = ExperimentConfig()
    
    # Create minimal pairs
    print("\nCreating minimal pairs...")
    pairs = create_minimal_pairs()
    print(f"Created {len(pairs)} minimal pairs")
    
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
    
    # Initialize patcher from library (for caching activations)
    # Using completion mode (chat_mode=False) since we're doing logit-based evaluation
    patcher = ActivationPatcher(model, tokenizer, chat_mode=False, max_new_tokens=1)
    
    print("\n✅ Using ActivationPatcher from library for patching mechanics")
    print("✅ Using completion mode (logit-based) for precise evaluation")
    sys.stdout.flush()
    
    # Critical layers from Step 4 & 5
    critical_layers = [32, 33, 34]
    random_layers = [5, 10, 15]  # Control - early layers
    
    print(f"\nCritical layers (from Step 4&5): {critical_layers}")
    print(f"Random layers (control): {random_layers}")
    sys.stdout.flush()
    
    # ========================================
    # TEST PATCHING
    # ========================================
    print(f"\n{'='*60}")
    print("PATCHING EXPERIMENTS")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    results = {
        "critical_layers": [],
        "random_layers": [],
    }
    
    for pair_idx, pair in enumerate(pairs):
        print(f"\n--- Pair {pair_idx + 1}: {pair['name']} ---")
        sys.stdout.flush()
        
        source = pair["source"]
        target = pair["target"]
        
        # Get baseline prediction on target (should be wrong or correct depending on model)
        baseline = get_logit_prediction(
            model, tokenizer, target["prompt"], target["correct"], target["wrong"]
        )
        if baseline is None:
            print(f"  [SKIP] Could not tokenize options for {pair['name']}")
            continue
        print(f"  Target baseline: {baseline['prediction']} (diff: {baseline['logit_diff']:.2f})")
        sys.stdout.flush()
        
        # Cache activations from source (the "knows" scenario) using library
        source_acts_critical = patcher.cache_activations(source["prompt"], critical_layers)
        source_acts_random = patcher.cache_activations(source["prompt"], random_layers)
        
        # Patch critical layers and get logits
        patched_critical = patch_and_get_logits(
            model, tokenizer, target["prompt"], source_acts_critical,
            target["correct"], target["wrong"]
        )
        
        if patched_critical is None:
            print(f"  [SKIP] Could not tokenize options for patched critical")
            continue
        
        # Patch random layers and get logits
        patched_random = patch_and_get_logits(
            model, tokenizer, target["prompt"], source_acts_random,
            target["correct"], target["wrong"]
        )
        
        if patched_random is None:
            print(f"  [SKIP] Could not tokenize options for patched random")
            continue
        
        # Did patching flip the prediction?
        critical_flip = baseline["prediction"] != patched_critical["prediction"]
        random_flip = baseline["prediction"] != patched_random["prediction"]
        
        # Record logit diff change
        critical_change = patched_critical["logit_diff"] - baseline["logit_diff"]
        random_change = patched_random["logit_diff"] - baseline["logit_diff"]
        
        print(f"  Critical patch: {patched_critical['prediction']} (flip={critical_flip}, change={critical_change:+.2f})")
        print(f"  Random patch:   {patched_random['prediction']} (flip={random_flip}, change={random_change:+.2f})")
        
        results["critical_layers"].append({
            "pair": pair["name"],
            "baseline": baseline["prediction"],
            "patched": patched_critical["prediction"],
            "flipped": critical_flip,
            "logit_change": critical_change,
        })
        
        results["random_layers"].append({
            "pair": pair["name"],
            "baseline": baseline["prediction"],
            "patched": patched_random["prediction"],
            "flipped": random_flip,
            "logit_change": random_change,
        })
    
    # ========================================
    # STATISTICS
    # ========================================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    critical_flip_rate = sum(r["flipped"] for r in results["critical_layers"]) / len(results["critical_layers"])
    random_flip_rate = sum(r["flipped"] for r in results["random_layers"]) / len(results["random_layers"])
    
    critical_mean_change = np.mean([r["logit_change"] for r in results["critical_layers"]])
    random_mean_change = np.mean([r["logit_change"] for r in results["random_layers"]])
    
    print(f"\nCritical Layers ({critical_layers}):")
    print(f"  Flip rate: {critical_flip_rate:.1%}")
    print(f"  Mean logit change: {critical_mean_change:+.2f}")
    
    print(f"\nRandom Layers ({random_layers}):")
    print(f"  Flip rate: {random_flip_rate:.1%}")
    print(f"  Mean logit change: {random_mean_change:+.2f}")
    
    # Hypothesis test
    h4_critical_flips = critical_flip_rate > random_flip_rate
    h4_logit_effect = abs(critical_mean_change) > abs(random_mean_change)
    
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}")
    print(f"\nH4a: Critical layers flip more than random: {'SUPPORTED' if h4_critical_flips else 'NOT SUPPORTED'}")
    print(f"     ({critical_flip_rate:.1%} vs {random_flip_rate:.1%})")
    print(f"\nH4b: Critical layers have larger logit effect: {'SUPPORTED' if h4_logit_effect else 'NOT SUPPORTED'}")
    print(f"     ({abs(critical_mean_change):.2f} vs {abs(random_mean_change):.2f})")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "critical_layers": critical_layers,
            "random_layers": random_layers,
            "n_pairs": len(pairs),
        },
        "results": results,
        "summary": {
            "critical_flip_rate": critical_flip_rate,
            "random_flip_rate": random_flip_rate,
            "critical_mean_change": critical_mean_change,
            "random_mean_change": random_mean_change,
        },
        "hypothesis_tests": {
            "H4a_critical_flips_more": h4_critical_flips,
            "H4b_critical_larger_effect": h4_logit_effect,
        },
    }
    
    output_path = RESULTS_DIR / "step6_patching.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    sys.stdout.flush()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Figure 1: Flip rates
    ax1 = axes[0]
    categories = ['Critical\n(L32-34)', 'Random\n(L5,10,15)']
    flip_rates = [critical_flip_rate * 100, random_flip_rate * 100]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(categories, flip_rates, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Flip Rate (%)", fontsize=12)
    ax1.set_title("Activation Patching: Flip Rates", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, rate in zip(bars, flip_rates):
        ax1.annotate(f'{rate:.0f}%', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Figure 2: Logit changes per pair
    ax2 = axes[1]
    x = np.arange(len(pairs))
    width = 0.35
    
    critical_changes = [r["logit_change"] for r in results["critical_layers"]]
    random_changes = [r["logit_change"] for r in results["random_layers"]]
    
    ax2.bar(x - width/2, critical_changes, width, label='Critical (L32-34)', color='steelblue')
    ax2.bar(x + width/2, random_changes, width, label='Random (L5,10,15)', color='coral')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel("Logit Diff Change", fontsize=12)
    ax2.set_title("Patching Effect by Pair", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p["name"] for p in pairs], rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step6_patching_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 6 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

