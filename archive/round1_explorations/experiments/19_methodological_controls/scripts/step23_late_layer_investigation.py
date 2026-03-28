"""
Step 23: Late Layer Investigation - Where Does the Override Happen?

BREAKTHROUGH from logit lens:
- Model has CORRECT answer at L31 (+6.50 for "told")
- But flips to WRONG in L32-L35 (-0.98 final)

This script investigates:
1. Which heads in L32-L35 cause the flip?
2. Can we ablate THOSE heads to fix the problem?
3. What do these heads attend to?
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

# Late layers to investigate
LATE_LAYERS = [32, 33, 34, 35]

# Test prompt
TEST_PROMPT = """Alice puts the ball in the drawer. Alice leaves.
Bob told Carol that he moved the ball to the basket.
Alice returns. Alice will look for the ball in the"""

CORRECT_ANSWER = " drawer"
WRONG_ANSWER = " basket"

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_model():
    """Load model with eager attention."""
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # For attention extraction
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_baseline(model, tokenizer) -> dict:
    """Get baseline prediction and logits."""
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    
    correct_id = tokenizer.encode(CORRECT_ANSWER, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(WRONG_ANSWER, add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]
    
    return {
        "correct_logit": float(logits[correct_id]),
        "wrong_logit": float(logits[wrong_id]),
        "diff": float(logits[correct_id] - logits[wrong_id]),
        "prediction": "correct" if logits[correct_id] > logits[wrong_id] else "wrong"
    }


def ablate_single_head(model, tokenizer, layer_idx: int, head_idx: int) -> dict:
    """Ablate a single head and measure effect."""
    
    def ablation_hook(module, input, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        original_shape = hidden.shape
        
        if hidden.dim() == 2:
            seq, hidden_size = hidden.shape
            hidden = hidden.unsqueeze(0)
        else:
            hidden_size = hidden.shape[-1]
        
        n_heads = model.config.num_attention_heads
        head_dim = hidden_size // n_heads
        
        hidden = hidden.view(-1, hidden.shape[1], n_heads, head_dim)
        hidden[:, :, head_idx, :] = 0
        hidden = hidden.view(-1, hidden.shape[1], hidden_size)
        
        if len(original_shape) == 2:
            hidden = hidden.squeeze(0)
        
        if isinstance(output, torch.Tensor):
            return hidden
        return (hidden,) + output[1:]
    
    # Register hook
    layer = model.model.layers[layer_idx].self_attn.o_proj
    hook = layer.register_forward_hook(ablation_hook)
    
    try:
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
        correct_id = tokenizer.encode(CORRECT_ANSWER, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(WRONG_ANSWER, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        return {
            "correct_logit": float(logits[correct_id]),
            "wrong_logit": float(logits[wrong_id]),
            "diff": float(logits[correct_id] - logits[wrong_id]),
            "prediction": "correct" if logits[correct_id] > logits[wrong_id] else "wrong"
        }
    finally:
        hook.remove()


def get_head_attention(model, tokenizer, layer_idx: int, head_idx: int) -> np.ndarray:
    """Get attention pattern for a specific head."""
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention for this head
    layer_attn = outputs.attentions[layer_idx]  # (1, n_heads, seq, seq)
    head_attn = layer_attn[0, head_idx].cpu().float().numpy()  # (seq, seq)
    
    return head_attn


def main():
    print("=" * 70)
    print("LATE LAYER INVESTIGATION: Finding the Override Mechanism")
    print("=" * 70)
    print()
    
    model, tokenizer = load_model()
    
    # Get number of heads
    n_heads = model.config.num_attention_heads
    print(f"Model has {n_heads} attention heads per layer")
    print(f"Investigating layers: {LATE_LAYERS}")
    print()
    
    # Baseline
    baseline = get_baseline(model, tokenizer)
    print(f"BASELINE: diff={baseline['diff']:.2f}, prediction={baseline['prediction']}")
    print("-" * 70)
    
    # Test each head in late layers
    results = {}
    significant_heads = []
    
    for layer_idx in LATE_LAYERS:
        print(f"\nLayer {layer_idx}:")
        layer_results = []
        
        for head_idx in range(n_heads):
            result = ablate_single_head(model, tokenizer, layer_idx, head_idx)
            
            # Calculate change from baseline
            change = result["diff"] - baseline["diff"]
            
            layer_results.append({
                "head_idx": head_idx,
                "diff": result["diff"],
                "change": change,
                "prediction": result["prediction"],
                "flipped": result["prediction"] != baseline["prediction"]
            })
            
            # Report significant changes
            if abs(change) > 0.5 or result["prediction"] != baseline["prediction"]:
                flip_str = " [FLIPPED!]" if result["prediction"] != baseline["prediction"] else ""
                sign = "+" if change > 0 else ""
                print(f"  H{head_idx}: diff={result['diff']:.2f} ({sign}{change:.2f}){flip_str}")
            
            if result["prediction"] != baseline["prediction"]:
                significant_heads.append((layer_idx, head_idx, change))
        
        results[f"L{layer_idx}"] = layer_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Heads That Can Fix the Prediction")
    print("=" * 70)
    
    if significant_heads:
        print("\nAblating these heads FLIPS the prediction from wrong to correct:")
        for layer, head, change in significant_heads:
            print(f"  L{layer}H{head}: change={change:+.2f}")
    else:
        print("\nNo single head ablation flips the prediction.")
        print("The override mechanism may be distributed across multiple heads.")
    
    # Find top positive-change heads
    print("\nTop heads that IMPROVE the diff (towards correct):")
    all_results = []
    for layer_name, layer_results in results.items():
        for r in layer_results:
            all_results.append({
                "layer": layer_name,
                "head": r["head_idx"],
                "change": r["change"],
                "diff": r["diff"]
            })
    
    sorted_results = sorted(all_results, key=lambda x: x["change"], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1}. {r['layer']}H{r['head']}: change={r['change']:+.2f}")
    
    # Test combined ablation of top heads
    print("\n" + "=" * 70)
    print("TESTING COMBINED ABLATION")
    print("=" * 70)
    
    top_heads = [(int(r["layer"][1:]), r["head"]) for r in sorted_results[:5]]
    print(f"\nAblating top 5 heads: {top_heads}")
    
    # Combined ablation
    def combined_hook_factory(heads_to_ablate):
        def hook_creator(target_layer, target_head):
            def hook(module, input, output):
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                original_shape = hidden.shape
                
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
                
                hidden_size = hidden.shape[-1]
                n_heads = model.config.num_attention_heads
                head_dim = hidden_size // n_heads
                
                hidden = hidden.view(-1, hidden.shape[1], n_heads, head_dim)
                hidden[:, :, target_head, :] = 0
                hidden = hidden.view(-1, hidden.shape[1], hidden_size)
                
                if len(original_shape) == 2:
                    hidden = hidden.squeeze(0)
                
                if isinstance(output, torch.Tensor):
                    return hidden
                return (hidden,) + output[1:]
            return hook
        return hook_creator
    
    hooks = []
    for layer_idx, head_idx in top_heads:
        layer = model.model.layers[layer_idx].self_attn.o_proj
        hook_creator = combined_hook_factory(top_heads)
        hook = layer.register_forward_hook(hook_creator(layer_idx, head_idx))
        hooks.append(hook)
    
    try:
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
        correct_id = tokenizer.encode(CORRECT_ANSWER, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(WRONG_ANSWER, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        combined_result = {
            "diff": float(logits[correct_id] - logits[wrong_id]),
            "prediction": "correct" if logits[correct_id] > logits[wrong_id] else "wrong"
        }
        
        print(f"\nCombined ablation result:")
        print(f"  Baseline diff: {baseline['diff']:.2f}")
        print(f"  After ablation: {combined_result['diff']:.2f}")
        print(f"  Change: {combined_result['diff'] - baseline['diff']:+.2f}")
        print(f"  Prediction: {combined_result['prediction']}")
        
        if combined_result["prediction"] == "correct":
            print("\n*** SUCCESS! Combined ablation fixes the prediction! ***")
        
    finally:
        for hook in hooks:
            hook.remove()
    
    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "layer_results": {k: [
            {"head": r["head_idx"], "diff": r["diff"], "change": r["change"], "flipped": r["flipped"]}
            for r in v
        ] for k, v in results.items()},
        "significant_heads": [(l, h, float(c)) for l, h, c in significant_heads],
        "top_positive_heads": [(r["layer"], r["head"], r["change"]) for r in sorted_results[:10]]
    }
    
    with open(OUTPUT_DIR / "late_layer_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nSaved to: {OUTPUT_DIR / 'late_layer_results.json'}")


if __name__ == "__main__":
    main()


