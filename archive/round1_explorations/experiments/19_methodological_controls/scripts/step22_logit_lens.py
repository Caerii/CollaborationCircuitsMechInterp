"""
Step 22: Logit Lens Analysis - Where Does the Model Decide?

The ablation doesn't fix "told" - so where is the decision made?

This script tracks how the model's prediction evolves through layers
to find WHERE the "told" → "update belief" shortcut happens.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test cases
TEST_CASES = {
    "told": {
        "prompt": "Alice puts the ball in the drawer. Alice leaves.\nBob told Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the",
        "correct": " drawer",
        "wrong": " basket",
        "expected": "should fail (0%)"
    },
    "announced": {
        "prompt": "Alice puts the ball in the drawer. Alice leaves.\nBob announced to Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the",
        "correct": " drawer", 
        "wrong": " basket",
        "expected": "should pass (100%)"
    },
    "will_tell": {
        "prompt": "Alice puts the ball in the drawer. Alice leaves.\nBob will tell Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the",
        "correct": " drawer",
        "wrong": " basket", 
        "expected": "should pass (100%)"
    }
}

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_model():
    """Load model."""
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_layer_predictions(model, tokenizer, prompt: str, correct: str, wrong: str) -> dict:
    """Get predictions at each layer using the logit lens technique.
    
    This applies the final unembedding (lm_head) to intermediate hidden states.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get token IDs for answers
    correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
    
    # Store hidden states
    hidden_states = []
    
    def capture_hook(module, input, output):
        # Capture the hidden state after each layer
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())
    
    # Register hooks on each layer's output
    hooks = []
    
    # Embedding layer
    hooks.append(model.model.embed_tokens.register_forward_hook(capture_hook))
    
    # Each transformer layer
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(capture_hook))
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        final_logits = outputs.logits[0, -1]
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Now apply lm_head to each layer's hidden state
    results = {
        "layers": [],
        "correct_logits": [],
        "wrong_logits": [],
        "diffs": [],
        "predictions": [],
        "final_prediction": "correct" if final_logits[correct_id] > final_logits[wrong_id] else "wrong"
    }
    
    # Apply layer norm and lm_head to each hidden state
    for i, hidden in enumerate(hidden_states):
        # Apply final layer norm
        normed = model.model.norm(hidden)
        
        # Apply lm_head (unembedding)
        logits = model.lm_head(normed)[0, -1]  # Last token logits
        
        correct_logit = float(logits[correct_id])
        wrong_logit = float(logits[wrong_id])
        diff = correct_logit - wrong_logit
        
        layer_name = "embed" if i == 0 else f"L{i-1}"
        
        results["layers"].append(layer_name)
        results["correct_logits"].append(correct_logit)
        results["wrong_logits"].append(wrong_logit)
        results["diffs"].append(diff)
        results["predictions"].append("correct" if diff > 0 else "wrong")
    
    return results


def plot_logit_evolution(all_results: dict):
    """Plot how predictions evolve through layers."""
    
    fig, axes = plt.subplots(len(all_results), 2, figsize=(14, 4 * len(all_results)))
    fig.suptitle("Logit Lens: How Predictions Evolve Through Layers", fontsize=14, fontweight='bold')
    
    for idx, (name, results) in enumerate(all_results.items()):
        # Plot 1: Logit values
        ax1 = axes[idx, 0] if len(all_results) > 1 else axes[0]
        
        x = range(len(results["layers"]))
        ax1.plot(x, results["correct_logits"], 'g-', label='Correct (drawer)', linewidth=2)
        ax1.plot(x, results["wrong_logits"], 'r-', label='Wrong (basket)', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark where prediction flips
        prev_pred = None
        for i, pred in enumerate(results["predictions"]):
            if prev_pred is not None and pred != prev_pred:
                ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.7, label='Flip')
            prev_pred = pred
        
        ax1.set_xticks(x[::5])
        ax1.set_xticklabels([results["layers"][i] for i in x[::5]], rotation=45)
        ax1.set_ylabel("Logit Value")
        ax1.set_title(f"'{name}' - Logit Evolution\n(Final: {results['final_prediction']})")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference
        ax2 = axes[idx, 1] if len(all_results) > 1 else axes[1]
        
        colors = ['green' if d > 0 else 'red' for d in results["diffs"]]
        ax2.bar(x, results["diffs"], color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=1)
        
        ax2.set_xticks(x[::5])
        ax2.set_xticklabels([results["layers"][i] for i in x[::5]], rotation=45)
        ax2.set_ylabel("Logit Diff (correct - wrong)")
        ax2.set_title(f"'{name}' - Prediction Margin")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_lens_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'logit_lens_evolution.png'}")


def find_decision_point(results: dict) -> dict:
    """Find where the model's prediction crystallizes."""
    
    diffs = results["diffs"]
    
    # Find first layer where prediction is stable until end
    final_pred = results["predictions"][-1]
    
    decision_layer = None
    for i in range(len(diffs) - 1, -1, -1):
        current_pred = results["predictions"][i]
        if current_pred != final_pred:
            decision_layer = i + 1
            break
    
    if decision_layer is None:
        decision_layer = 0  # Always had this prediction
    
    # Find the layer with maximum absolute difference
    max_diff_layer = np.argmax(np.abs(diffs))
    
    # Find where the sign first becomes consistent with final
    first_consistent = 0
    final_sign = 1 if diffs[-1] > 0 else -1
    for i, d in enumerate(diffs):
        if (d > 0 and final_sign > 0) or (d < 0 and final_sign < 0):
            first_consistent = i
            break
    
    return {
        "decision_layer": results["layers"][decision_layer],
        "decision_layer_idx": decision_layer,
        "max_diff_layer": results["layers"][max_diff_layer],
        "max_diff_value": diffs[max_diff_layer],
        "first_consistent_layer": results["layers"][first_consistent],
        "final_diff": diffs[-1]
    }


def main():
    print("=" * 70)
    print("LOGIT LENS ANALYSIS: Where Does the Model Decide?")
    print("=" * 70)
    print()
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    print("\nAnalyzing each test case...")
    print("-" * 70)
    
    for name, case in TEST_CASES.items():
        print(f"\n>> {name} ({case['expected']})")
        
        results = get_layer_predictions(
            model, tokenizer, 
            case["prompt"], case["correct"], case["wrong"]
        )
        
        all_results[name] = results
        
        # Analyze decision point
        decision = find_decision_point(results)
        
        print(f"   Final prediction: {results['final_prediction']}")
        print(f"   Decision crystallizes at: {decision['decision_layer']}")
        print(f"   Max difference at: {decision['max_diff_layer']} ({decision['max_diff_value']:.2f})")
        print(f"   First correct sign at: {decision['first_consistent_layer']}")
    
    # Create visualization
    print("\n" + "=" * 50)
    print("Creating visualizations...")
    plot_logit_evolution(all_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    for name, results in all_results.items():
        decision = find_decision_point(results)
        print(f"\n{name}:")
        print(f"  Final: {results['final_prediction']}")
        print(f"  Decision layer: {decision['decision_layer']}")
        
        # Layer-by-layer breakdown for first 10 and last 10
        print(f"  Early layers (embed-L9):")
        for i in range(min(11, len(results['diffs']))):
            d = results['diffs'][i]
            sign = "+" if d > 0 else "-"
            print(f"    {results['layers'][i]}: {sign}{abs(d):.2f}")
        
        print(f"  Late layers (L26-final):")
        for i in range(max(0, len(results['diffs']) - 10), len(results['diffs'])):
            d = results['diffs'][i]
            sign = "+" if d > 0 else "-"
            print(f"    {results['layers'][i]}: {sign}{abs(d):.2f}")
    
    # Save results
    save_results = {
        name: {
            "layers": r["layers"],
            "diffs": r["diffs"],
            "final_prediction": r["final_prediction"],
            "decision_point": find_decision_point(r)
        }
        for name, r in all_results.items()
    }
    
    with open(OUTPUT_DIR / "logit_lens_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": save_results
        }, f, indent=2)
    
    print(f"\nSaved to: {OUTPUT_DIR / 'logit_lens_results.json'}")


if __name__ == "__main__":
    main()


