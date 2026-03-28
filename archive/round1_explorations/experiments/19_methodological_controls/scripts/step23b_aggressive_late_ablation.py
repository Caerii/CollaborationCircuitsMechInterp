"""
Step 23b: Aggressive Late Layer Ablation

Previous finding: 5-head ablation improved diff from -0.98 to -0.08
Let's try more aggressive ablation to flip the prediction.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"

# Test cases
TEST_CASES = [
    ("told", "Alice puts the ball in the drawer. Alice leaves.\nBob told Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the"),
    ("said", "Alice puts the ball in the drawer. Alice leaves.\nBob said to Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the"),
    ("announced", "Alice puts the ball in the drawer. Alice leaves.\nBob announced to Carol that he moved the ball to the basket.\nAlice returns. Alice will look for the ball in the"),
]

CORRECT_ANSWER = " drawer"
WRONG_ANSWER = " basket"

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"


def load_model():
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


def test_with_ablation(model, tokenizer, prompt: str, heads_to_ablate: list) -> dict:
    """Test prediction with specified heads ablated."""
    
    hooks = []
    
    for layer_idx, head_idx in heads_to_ablate:
        def make_hook(target_head):
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
        
        layer = model.model.layers[layer_idx].self_attn.o_proj
        hook = layer.register_forward_hook(make_hook(head_idx))
        hooks.append(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        for hook in hooks:
            hook.remove()


def main():
    print("=" * 70)
    print("AGGRESSIVE LATE LAYER ABLATION")
    print("=" * 70)
    
    model, tokenizer = load_model()
    n_heads = model.config.num_attention_heads
    
    # Top heads from previous analysis
    top_heads_by_layer = {
        32: [6, 31],
        33: [6, 13, 17, 31],
        34: [17],
        35: [0, 1, 17]
    }
    
    # Build cumulative head lists
    ablation_configs = [
        ("Top 5 (previous)", [(35, 0), (35, 1), (32, 6), (32, 31), (33, 13)]),
        ("Top 10", [(35, 0), (35, 1), (32, 6), (32, 31), (33, 13), (33, 17), (35, 17), (33, 6), (33, 31), (34, 17)]),
        ("All L35", [(35, h) for h in range(n_heads)]),
        ("All L34+L35", [(34, h) for h in range(n_heads)] + [(35, h) for h in range(n_heads)]),
        ("All L33-L35", [(33, h) for h in range(n_heads)] + [(34, h) for h in range(n_heads)] + [(35, h) for h in range(n_heads)]),
        ("All L32-L35", [(32, h) for h in range(n_heads)] + [(33, h) for h in range(n_heads)] + [(34, h) for h in range(n_heads)] + [(35, h) for h in range(n_heads)]),
    ]
    
    results = {}
    
    for verb, prompt in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"Testing: '{verb}'")
        print("=" * 70)
        
        # Baseline
        baseline = test_with_ablation(model, tokenizer, prompt, [])
        print(f"\nBaseline: diff={baseline['diff']:.2f}, pred={baseline['prediction']}")
        
        verb_results = {"baseline": baseline, "ablations": {}}
        
        for config_name, heads in ablation_configs:
            result = test_with_ablation(model, tokenizer, prompt, heads)
            change = result["diff"] - baseline["diff"]
            flipped = result["prediction"] != baseline["prediction"]
            
            flip_str = " *** FLIPPED! ***" if flipped else ""
            print(f"{config_name:20s}: diff={result['diff']:+.2f} (change={change:+.2f}){flip_str}")
            
            verb_results["ablations"][config_name] = {
                "heads": len(heads),
                "diff": result["diff"],
                "change": change,
                "prediction": result["prediction"],
                "flipped": flipped
            }
        
        results[verb] = verb_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for verb, verb_results in results.items():
        print(f"\n{verb}:")
        for config_name, r in verb_results["ablations"].items():
            if r["flipped"]:
                print(f"  [FIXED] {config_name}: {r['heads']} heads")
    
    # Save
    with open(OUTPUT_DIR / "aggressive_late_ablation_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, default=str)
    
    print(f"\nSaved to: {OUTPUT_DIR / 'aggressive_late_ablation_results.json'}")


if __name__ == "__main__":
    main()


