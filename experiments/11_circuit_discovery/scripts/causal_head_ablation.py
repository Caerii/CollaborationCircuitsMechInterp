"""
Causal Head Ablation
=====================

TEST: Are the identified heads CAUSALLY NECESSARY for agent modeling?

Method:
1. Zero out specific head outputs
2. Check if model's agreement prediction changes
3. Heads that matter when ablated = causal ToM heads

EFFICIENT: Vectorized ablation, batched inference
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("CAUSAL HEAD ABLATION")
print("=" * 60)

# Test cases where we know the expected answer
TEST_CASES = [
    {
        "prompt": "Agent A: '2+2=4' Agent B knows math. Does B agree with A? Answer yes or no: ",
        "expected": "yes",
        "b_should_agree": True,
    },
    {
        "prompt": "Agent A: '2+2=5' Agent B knows math. Does B agree with A? Answer yes or no: ",
        "expected": "no",
        "b_should_agree": False,
    },
    {
        "prompt": "Agent A: 'Paris is in France.' Agent B studied geography. Does B agree? Answer yes or no: ",
        "expected": "yes",
        "b_should_agree": True,
    },
    {
        "prompt": "Agent A: 'Tokyo is in Europe.' Agent B knows Asia. Does B agree? Answer yes or no: ",
        "expected": "no",
        "b_should_agree": False,
    },
]


class HeadAblator:
    """Efficiently ablate specific attention heads."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.ablation_hooks = []
        self.ablate_targets = set()
    
    def set_ablation(self, layer: int, head: int):
        """Mark a head for ablation."""
        self.ablate_targets.add((layer, head))
    
    def clear_ablation(self):
        """Clear all ablation targets."""
        self.ablate_targets.clear()
    
    def _make_ablation_hook(self, layer_idx: int):
        """Create hook that zeros out specific heads."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch, seq, hidden_size)
            batch, seq_len, hidden_size = hidden.shape
            
            # Reshape to (batch, seq, n_heads, head_dim)
            hidden_reshaped = hidden.view(batch, seq_len, self.n_heads, self.head_dim)
            
            # Zero out ablated heads
            for (target_layer, target_head) in self.ablate_targets:
                if target_layer == layer_idx:
                    hidden_reshaped[:, :, target_head, :] = 0
            
            # Reshape back
            new_hidden = hidden_reshaped.view(batch, seq_len, hidden_size)
            
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        
        return hook
    
    def install_hooks(self):
        """Install ablation hooks on all layers."""
        self.remove_hooks()
        for layer_idx in range(self.n_layers):
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                self._make_ablation_hook(layer_idx)
            )
            self.ablation_hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.ablation_hooks:
            hook.remove()
        self.ablation_hooks = []
    
    def generate(self, prompt: str, max_new_tokens: int = 5) -> str:
        """Generate with current ablation settings."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip().lower()


def check_response(response: str, expected: str) -> bool:
    """Check if response matches expected."""
    response = response.lower()
    if expected == "yes":
        return "yes" in response and "no" not in response
    else:
        return "no" in response and "yes" not in response


def main():
    start_time = time.perf_counter()
    
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
    
    load_time = time.perf_counter() - start_time
    print(f"  [OK] Loaded in {load_time:.1f}s")
    
    ablator = HeadAblator(model, tokenizer)
    n_layers = ablator.n_layers
    n_heads = ablator.n_heads
    
    print(f"  {n_layers} layers x {n_heads} heads = {n_layers * n_heads} total heads")
    
    # Get baseline responses
    print("\n[2/4] Getting baseline responses...", flush=True)
    baseline_start = time.perf_counter()
    
    baselines = []
    for tc in TEST_CASES:
        response = ablator.generate(tc["prompt"])
        correct = check_response(response, tc["expected"])
        baselines.append({
            "prompt": tc["prompt"][:50],
            "expected": tc["expected"],
            "response": response[:20],
            "correct": correct,
        })
        print(f"    Expected '{tc['expected']}', got '{response[:15]}' -> {'OK' if correct else 'WRONG'}")
    
    baseline_time = time.perf_counter() - baseline_start
    print(f"  Baseline time: {baseline_time:.1f}s")
    
    # Test ablating heads (sample subset for speed)
    print("\n[3/4] Testing head ablations...", flush=True)
    ablation_start = time.perf_counter()
    
    ablator.install_hooks()
    
    # Test layers that showed high ToM signal (from previous analysis)
    test_layers = [12, 18, 24, 30, 35]  # Sample of layers
    test_heads = list(range(0, n_heads, 4))  # Every 4th head for speed
    
    ablation_effects = []
    
    total_tests = len(test_layers) * len(test_heads)
    test_idx = 0
    
    for layer in test_layers:
        for head in test_heads:
            test_idx += 1
            if test_idx % 10 == 0:
                print(f"    Progress: {test_idx}/{total_tests}", flush=True)
            
            ablator.clear_ablation()
            ablator.set_ablation(layer, head)
            
            # Test on all cases
            flips = 0
            for i, tc in enumerate(TEST_CASES):
                response = ablator.generate(tc["prompt"])
                correct = check_response(response, tc["expected"])
                
                # Did ablation flip the response?
                if correct != baselines[i]["correct"]:
                    flips += 1
            
            if flips > 0:
                ablation_effects.append({
                    "layer": layer,
                    "head": head,
                    "flips": flips,
                    "flip_rate": flips / len(TEST_CASES),
                })
    
    ablator.remove_hooks()
    ablation_time = time.perf_counter() - ablation_start
    
    # Sort by effect
    ablation_effects.sort(key=lambda x: x["flips"], reverse=True)
    
    print(f"\n  Ablation testing time: {ablation_time:.1f}s")
    print(f"  Tested {total_tests} head ablations")
    
    # Results
    print("\n[4/4] Results...", flush=True)
    
    print("\n" + "=" * 60)
    print("CAUSAL HEAD ABLATION RESULTS")
    print("=" * 60)
    
    if ablation_effects:
        print("\nHeads that CAUSALLY AFFECT agent modeling (when ablated):")
        print(f"{'Layer':<8} {'Head':<8} {'Flips':<8} {'Flip Rate':<12}")
        print("-" * 36)
        
        for effect in ablation_effects[:20]:
            print(f"{effect['layer']:<8} {effect['head']:<8} {effect['flips']:<8} {effect['flip_rate']:.1%}")
        
        # Find most causal layer
        layer_effects = {}
        for e in ablation_effects:
            l = e["layer"]
            layer_effects[l] = layer_effects.get(l, 0) + e["flips"]
        
        if layer_effects:
            most_causal_layer = max(layer_effects.keys(), key=lambda l: layer_effects[l])
            print(f"\nMost causally important layer: {most_causal_layer}")
    else:
        print("\nNo heads showed causal effect (responses unchanged by ablation)")
        print("This could mean:")
        print("  1. ToM is distributed across many heads")
        print("  2. Model is robust to single-head ablation")
        print("  3. Need to test different heads/layers")
    
    # Save results
    results = {
        "baselines": baselines,
        "ablation_effects": ablation_effects,
        "tested_layers": test_layers,
        "tested_heads": test_heads,
        "timing": {
            "model_load": load_time,
            "baseline": baseline_time,
            "ablation": ablation_time,
            "total": time.perf_counter() - start_time,
        },
    }
    
    with open(RESULTS_DIR / "causal_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - start_time
    print(f"\n" + "=" * 60)
    print(f"TOTAL TIME: {total_time:.1f}s")
    print("=" * 60)
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'causal_ablation.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()






















