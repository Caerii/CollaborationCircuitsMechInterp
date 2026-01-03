"""
Step 31: Cross-Model Validation

Test if the late-layer override pattern exists in other models.
We'll test:
1. Qwen3-4B (our main model) - baseline reference
2. Other available small models

For each model, we check:
- Does "told" cause ToM failure?
- Is there a late-layer override pattern?
- Which layers/heads matter?
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Models to test (smaller ones that fit in VRAM)
MODELS_TO_TEST = [
    {
        "name": "Qwen3-4B",
        "model_id": "Qwen/Qwen3-4B",
        "n_layers": 36,
        "n_heads": 32,
        "late_layers": [32, 33, 34, 35]
    },
    {
        "name": "Qwen2.5-1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "n_layers": 28,
        "n_heads": 12,
        "late_layers": [24, 25, 26, 27]
    },
]

BAD_VERBS = ["told", "said", "mentioned"]
GOOD_VERBS = ["announced", "asked", "hinted"]


def load_model(model_id):
    """Load a model."""
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def create_prompt(verb):
    return f"""Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice looked for the ball. Alice searched in the"""


def test_verb(model, tokenizer, verb):
    """Test a single verb and return logit difference."""
    prompt = create_prompt(verb)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get drawer and basket token IDs
    drawer_tokens = tokenizer.encode(" drawer", add_special_tokens=False)
    basket_tokens = tokenizer.encode(" basket", add_special_tokens=False)
    
    if not drawer_tokens or not basket_tokens:
        return {"error": "Could not tokenize drawer/basket"}
    
    drawer_id = drawer_tokens[0]
    basket_id = basket_tokens[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    drawer_logit = logits[drawer_id].item()
    basket_logit = logits[basket_id].item()
    
    return {
        "drawer": drawer_logit,
        "basket": basket_logit,
        "diff": drawer_logit - basket_logit,
        "correct": drawer_logit > basket_logit
    }


def test_model_baseline(model, tokenizer, model_name):
    """Test baseline ToM performance for a model."""
    print(f"\n{'='*60}")
    print(f"BASELINE: {model_name}")
    print(f"{'='*60}")
    
    results = {"bad_verbs": {}, "good_verbs": {}}
    
    print("\nBAD VERBS:")
    for verb in BAD_VERBS:
        result = test_verb(model, tokenizer, verb)
        results["bad_verbs"][verb] = result
        status = "[OK]" if result.get("correct", False) else "[FAIL]"
        print(f"  {status} {verb}: diff={result.get('diff', 'N/A'):+.2f}")
    
    print("\nGOOD VERBS:")
    for verb in GOOD_VERBS:
        result = test_verb(model, tokenizer, verb)
        results["good_verbs"][verb] = result
        status = "[OK]" if result.get("correct", False) else "[FAIL]"
        print(f"  {status} {verb}: diff={result.get('diff', 'N/A'):+.2f}")
    
    bad_correct = sum(1 for r in results["bad_verbs"].values() if r.get("correct", False))
    good_correct = sum(1 for r in results["good_verbs"].values() if r.get("correct", False))
    
    print(f"\nSummary: Bad verbs {bad_correct}/{len(BAD_VERBS)}, Good verbs {good_correct}/{len(GOOD_VERBS)}")
    
    return results


def search_override_circuit(model, tokenizer, model_config, verb="told"):
    """Search for override circuit in late layers."""
    print(f"\n{'='*60}")
    print(f"SEARCHING FOR OVERRIDE CIRCUIT: {model_config['name']}")
    print(f"{'='*60}")
    
    # Test baseline
    baseline = test_verb(model, tokenizer, verb)
    print(f"\nBaseline '{verb}': diff={baseline['diff']:+.2f} ({'OK' if baseline['correct'] else 'FAIL'})")
    
    if baseline["correct"]:
        print("Baseline is correct - no override circuit needed")
        return {"baseline_correct": True, "improving_heads": []}
    
    # Search late layers for improving heads
    improving_heads = []
    
    for layer_idx in model_config["late_layers"]:
        print(f"\nSearching Layer {layer_idx}...")
        
        for head_idx in range(model_config["n_heads"]):
            # Register ablation hook
            hooks = []
            layer = model.model.layers[layer_idx]
            
            def make_hook(h_idx, n_heads):
                def hook(module, input, output):
                    hidden = output
                    batch, seq_len, hidden_size = hidden.shape
                    head_dim = hidden_size // n_heads
                    hidden = hidden.view(batch, seq_len, n_heads, head_dim)
                    hidden[:, :, h_idx, :] = 0
                    hidden = hidden.view(batch, seq_len, hidden_size)
                    return hidden
                return hook
            
            hook = layer.self_attn.o_proj.register_forward_hook(
                make_hook(head_idx, model_config["n_heads"])
            )
            hooks.append(hook)
            
            # Test with ablation
            result = test_verb(model, tokenizer, verb)
            
            # Clean up
            for h in hooks:
                h.remove()
            
            # Check if it improved
            if result["diff"] > baseline["diff"] + 0.5:  # Meaningful improvement
                improving_heads.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "diff": result["diff"],
                    "improvement": result["diff"] - baseline["diff"]
                })
                print(f"  L{layer_idx}H{head_idx}: diff={result['diff']:+.2f} (+{result['diff']-baseline['diff']:.2f})")
    
    # Sort by improvement
    improving_heads.sort(key=lambda x: x["improvement"], reverse=True)
    
    print(f"\nFound {len(improving_heads)} improving heads")
    
    return {
        "baseline_correct": False,
        "baseline_diff": baseline["diff"],
        "improving_heads": improving_heads[:10]  # Top 10
    }


def run_cross_model_validation():
    """Run validation across multiple models."""
    all_results = {}
    
    for model_config in MODELS_TO_TEST:
        model_name = model_config["name"]
        print(f"\n{'#'*70}")
        print(f"# TESTING: {model_name}")
        print(f"{'#'*70}")
        
        try:
            model, tokenizer = load_model(model_config["model_id"])
            
            # Test baseline
            baseline_results = test_model_baseline(model, tokenizer, model_name)
            
            # Search for override circuit
            circuit_results = search_override_circuit(model, tokenizer, model_config)
            
            all_results[model_name] = {
                "baseline": baseline_results,
                "circuit_search": circuit_results
            }
            
            # Clean up to free VRAM
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"\n{model_name}: ERROR - {results['error']}")
            continue
        
        baseline = results["baseline"]
        circuit = results["circuit_search"]
        
        bad_correct = sum(1 for r in baseline["bad_verbs"].values() if r.get("correct", False))
        good_correct = sum(1 for r in baseline["good_verbs"].values() if r.get("correct", False))
        
        print(f"\n{model_name}:")
        print(f"  Baseline: Bad {bad_correct}/3, Good {good_correct}/3")
        
        if circuit.get("baseline_correct"):
            print(f"  Circuit: Not needed (baseline correct)")
        else:
            n_improving = len(circuit.get("improving_heads", []))
            print(f"  Circuit: Found {n_improving} improving heads")
            if n_improving > 0:
                top_head = circuit["improving_heads"][0]
                print(f"  Top head: L{top_head['layer']}H{top_head['head']} (+{top_head['improvement']:.2f})")
    
    # Save results
    save_path = RESULTS_DIR / "cross_model_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


def main():
    print("="*70)
    print("STEP 31: Cross-Model Validation")
    print("="*70)
    print("\nTesting whether the late-layer override pattern exists in other models")
    
    results = run_cross_model_validation()


if __name__ == "__main__":
    main()


