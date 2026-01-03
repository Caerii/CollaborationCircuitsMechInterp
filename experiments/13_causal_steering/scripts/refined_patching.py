"""
Refined Activation Patching
============================

More careful approach:
1. Only patch at specific positions (not all)
2. Use additive patching (blend rather than replace)
3. Focus on the key "agent" tokens
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Simpler, more controlled test cases
TEST_CASES = [
    {
        "agree_prompt": "Q: Is 2+2=4? A: Yes, that is",
        "disagree_prompt": "Q: Is 2+2=5? A: No, that is",
        "test_prompt": "Q: Is 2+2=4? A:",
    },
    {
        "agree_prompt": "Agent A is correct. Agent B agrees with",
        "disagree_prompt": "Agent A is wrong. Agent B disagrees with",
        "test_prompt": "Agent A makes a claim. Agent B",
    },
]


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("REFINED ACTIVATION PATCHING")
    print("=" * 60)
    
    print("\n[1/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK]", flush=True)
    
    results = {"tests": []}
    
    print("\n[2/4] Running refined patching tests...", flush=True)
    
    for i, tc in enumerate(TEST_CASES):
        print(f"\n  Test {i+1}:", flush=True)
        print(f"    Agree: '{tc['agree_prompt'][:40]}'", flush=True)
        print(f"    Disagree: '{tc['disagree_prompt'][:40]}'", flush=True)
        
        # Get baseline generation
        inputs = tokenizer(tc["test_prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=15, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        baseline = tokenizer.decode(output[0], skip_special_tokens=True)[len(tc["test_prompt"]):]
        print(f"    Baseline: '{baseline[:40]}'", flush=True)
        
        # Extract steering direction from agree vs disagree
        agree_inputs = tokenizer(tc["agree_prompt"], return_tensors="pt").to("cuda")
        disagree_inputs = tokenizer(tc["disagree_prompt"], return_tensors="pt").to("cuda")
        
        # Get last layer activations (residual stream at end)
        captured = {}
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()
        
        # Test layers
        test_layers = [12, 24, 35]
        
        for layer in test_layers:
            handle = model.model.layers[layer].register_forward_hook(hook)
            
            with torch.no_grad():
                _ = model(**agree_inputs)
                agree_act = captured["hidden"][0, -1, :].clone()
                
                _ = model(**disagree_inputs)
                disagree_act = captured["hidden"][0, -1, :].clone()
            
            handle.remove()
            
            # Steering vector
            steer = agree_act - disagree_act
            steer_norm = steer / (steer.norm() + 1e-8)
            
            # Apply steering during generation
            def steer_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Add steering to last position
                hidden[:, -1, :] = hidden[:, -1, :] + 2.0 * steer_norm * hidden[:, -1, :].norm()
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            
            steer_handle = model.model.layers[layer].register_forward_hook(steer_hook)
            
            with torch.no_grad():
                steered_output = model.generate(**inputs, max_new_tokens=15, do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id)
            steered = tokenizer.decode(steered_output[0], skip_special_tokens=True)[len(tc["test_prompt"]):]
            
            steer_handle.remove()
            
            changed = baseline.strip()[:10] != steered.strip()[:10]
            print(f"    L{layer} steered: '{steered[:30]}' {'[CHANGED]' if changed else ''}", flush=True)
            
            results["tests"].append({
                "test_idx": i,
                "layer": layer,
                "baseline": baseline[:50],
                "steered": steered[:50],
                "changed": changed,
            })
    
    print("\n[3/4] Analyzing results...", flush=True)
    
    # Count changes per layer
    by_layer = {}
    for r in results["tests"]:
        l = r["layer"]
        if l not in by_layer:
            by_layer[l] = {"total": 0, "changed": 0}
        by_layer[l]["total"] += 1
        if r["changed"]:
            by_layer[l]["changed"] += 1
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nChange rate by layer:")
    for l in sorted(by_layer.keys()):
        rate = by_layer[l]["changed"] / by_layer[l]["total"] if by_layer[l]["total"] > 0 else 0
        print(f"  Layer {l}: {by_layer[l]['changed']}/{by_layer[l]['total']} = {rate:.0%}")
    
    # Save
    with open(RESULTS_DIR / "refined_patching.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()























