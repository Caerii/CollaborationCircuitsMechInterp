"""
COMBINED INHIBITOR ABLATION
===========================

Now that we know there's a distributed inhibitory network,
let's test ablating MULTIPLE inhibitors together.

Questions:
1. Does ablating top 3 inhibitors give better than single ablation?
2. Is there additive or ceiling effect?
3. Can we reach near-100% without prompting?
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random
from itertools import combinations

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MultiAblator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        
    def _create_ablation_hook(self, head_indices: list):
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            for head_idx in head_indices:
                if head_idx < n_heads:
                    reshaped[:, :, head_idx, :] = 0
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def install_multi_ablation(self, layer_head_pairs: list):
        """Ablate multiple heads across multiple layers."""
        self.clear_hooks()
        
        # Group by layer
        layer_to_heads = {}
        for layer, head in layer_head_pairs:
            if layer not in layer_to_heads:
                layer_to_heads[layer] = []
            layer_to_heads[layer].append(head)
        
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_indices))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_batch(self, scenarios: list) -> float:
        correct = 0
        for s in scenarios:
            inputs = self.tokenizer(s["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
            
            correct_id = self.tokenizer.encode(s["correct"], add_special_tokens=False)[0]
            wrong_id = self.tokenizer.encode(s["wrong"], add_special_tokens=False)[0]
            
            if logits[correct_id].item() > logits[wrong_id].item():
                correct += 1
        
        return correct / len(scenarios)


def generate_scenarios(n: int = 40) -> list:
    random.seed(42)
    scenarios = []
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    INFORMERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice([x for x in INFORMERS if x != agent])
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        prompt = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        scenarios.append({
            "prompt": prompt,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
        })
    
    return scenarios


def main():
    print("=" * 70)
    print("COMBINED INHIBITOR ABLATION TEST")
    print("=" * 70)
    print("Can ablating multiple inhibitors unlock near-perfect ToM?")
    print()
    
    # Load model
    print("[1/4] Loading model...", flush=True)
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
    
    ablator = MultiAblator(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/4] Generating scenarios...", flush=True)
    scenarios = generate_scenarios(40)
    print(f"  {len(scenarios)} scenarios", flush=True)
    
    # Define top inhibitors from our search
    TOP_INHIBITORS = [
        (17, 4),   # +48% - STRONGEST
        (15, 12),  # +36%
        (24, 29),  # +32% - Our original find
        (22, 11),  # +28%
        (23, 0),   # +24%
        (20, 28),  # +24%
    ]
    
    results = {}
    
    # Baseline
    print("\n[3/4] Testing baselines...", flush=True)
    ablator.clear_hooks()
    baseline = ablator.test_batch(scenarios)
    results["baseline"] = baseline
    print(f"  Baseline (no ablation): {baseline:.1%}")
    
    # Single ablations
    print("\n[4/4] Testing combinations...", flush=True)
    print()
    print("  SINGLE ABLATIONS:")
    print("  " + "-" * 50)
    
    for layer, head in TOP_INHIBITORS[:6]:
        ablator.install_multi_ablation([(layer, head)])
        acc = ablator.test_batch(scenarios)
        boost = acc - baseline
        results[f"L{layer}H{head}"] = acc
        print(f"    L{layer}H{head}: {acc:.1%} (boost: {boost:+.1%})")
    ablator.clear_hooks()
    
    # Combinations of 2
    print("\n  TOP 2 COMBINATIONS:")
    print("  " + "-" * 50)
    
    best_2_combo = None
    best_2_acc = 0
    
    for (l1, h1), (l2, h2) in combinations(TOP_INHIBITORS[:4], 2):
        ablator.install_multi_ablation([(l1, h1), (l2, h2)])
        acc = ablator.test_batch(scenarios)
        boost = acc - baseline
        key = f"L{l1}H{h1}+L{l2}H{h2}"
        results[key] = acc
        print(f"    {key}: {acc:.1%} (boost: {boost:+.1%})")
        if acc > best_2_acc:
            best_2_acc = acc
            best_2_combo = [(l1, h1), (l2, h2)]
    ablator.clear_hooks()
    
    # Combinations of 3
    print("\n  TOP 3 COMBINATIONS:")
    print("  " + "-" * 50)
    
    best_3_combo = None
    best_3_acc = 0
    
    for combo in combinations(TOP_INHIBITORS[:4], 3):
        ablator.install_multi_ablation(list(combo))
        acc = ablator.test_batch(scenarios)
        boost = acc - baseline
        key = "+".join([f"L{l}H{h}" for l, h in combo])
        results[key] = acc
        print(f"    {key}: {acc:.1%} (boost: {boost:+.1%})")
        if acc > best_3_acc:
            best_3_acc = acc
            best_3_combo = list(combo)
    ablator.clear_hooks()
    
    # All top 4
    print("\n  ALL TOP 4:")
    print("  " + "-" * 50)
    
    ablator.install_multi_ablation(TOP_INHIBITORS[:4])
    acc = ablator.test_batch(scenarios)
    boost = acc - baseline
    key = "ALL_TOP_4"
    results[key] = acc
    print(f"    All top 4 inhibitors: {acc:.1%} (boost: {boost:+.1%})")
    ablator.clear_hooks()
    
    # All top 6
    ablator.install_multi_ablation(TOP_INHIBITORS[:6])
    acc = ablator.test_batch(scenarios)
    boost = acc - baseline
    results["ALL_TOP_6"] = acc
    print(f"    All top 6 inhibitors: {acc:.1%} (boost: {boost:+.1%})")
    ablator.clear_hooks()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n  Baseline:              {baseline:.1%}")
    print(f"  Best single ablation:  {max([results[f'L{l}H{h}'] for l, h in TOP_INHIBITORS[:6]]):.1%}")
    print(f"  Best 2-combo:          {best_2_acc:.1%} ({best_2_combo})")
    print(f"  Best 3-combo:          {best_3_acc:.1%} ({best_3_combo})")
    print(f"  All top 4:             {results['ALL_TOP_4']:.1%}")
    print(f"  All top 6:             {results['ALL_TOP_6']:.1%}")
    
    # Analysis
    print("\n  INTERPRETATION:")
    if results["ALL_TOP_6"] > 0.90:
        print("  >>> NEAR-PERFECT ToM achieved by ablating inhibitory network!")
    elif results["ALL_TOP_6"] > results["ALL_TOP_4"]:
        print("  >>> More inhibitors = better. Network is additive.")
    elif results["ALL_TOP_6"] < max([results[f'L{l}H{h}'] for l, h in TOP_INHIBITORS[:6]]):
        print("  >>> Diminishing returns. Some heads may be redundant.")
    else:
        print("  >>> Complex interaction pattern. Not simply additive.")
    
    # Save
    with open(RESULTS_DIR / "combined_inhibitor_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'combined_inhibitor_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

