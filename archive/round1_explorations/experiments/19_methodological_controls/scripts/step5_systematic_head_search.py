"""
Systematic Head Search: Which Heads Matter Most for Belief Update?
===================================================================

The initial ablation showed ~10% drop from L23H4+L28H0.
Let's do a systematic search to find the MOST important heads.

Test each head from our attention analysis individually to find:
1. Which single head has the biggest impact when ablated?
2. Is the effect distributed or concentrated?
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SingleHeadAblator:
    """Ablate individual attention heads."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        
    def _create_single_head_ablation_hook(self, head_idx: int):
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            reshaped[:, :, head_idx, :] = 0
            modified = reshaped.view(batch, seq_len, hidden)
            return (modified,) + args[1:] if len(args) > 1 else (modified,)
        return hook
    
    def install_single_ablation(self, layer_idx: int, head_idx: int):
        self.clear_hooks()
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(self._create_single_head_ablation_hook(head_idx))
        self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_prompt(self, prompt: str, correct: str, wrong: str) -> bool:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        return logits[correct_id].item() > logits[wrong_id].item()


def generate_test_scenarios(n: int = 30) -> list:
    """Generate scenarios for testing."""
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    INFORMERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    scenarios = []
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice(INFORMERS)
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        bridged = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"so {agent} updated their belief. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        scenarios.append({
            "bridged": bridged,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("SYSTEMATIC HEAD SEARCH: Find Most Important Heads")
    print("=" * 70)
    print("""
    Testing individual heads from the attention analysis.
    Looking for heads that cause the BIGGEST drop when ablated.
    """)
    
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
    print("  [OK] Model loaded", flush=True)
    
    ablator = SingleHeadAblator(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/4] Generating test scenarios...", flush=True)
    scenarios = generate_test_scenarios(30)  # Smaller N for speed
    print(f"  Generated {len(scenarios)} scenarios")
    
    # Heads to test (from attention analysis + some controls)
    # Top positive changes from our analysis
    HEADS_TO_TEST = [
        (23, 4),   # +0.54 - biggest
        (28, 0),   # +0.50 - second
        (24, 29),  # +0.45
        (26, 26),  # +0.38
        (23, 30),  # +0.33
        (28, 23),  # +0.32
        (28, 3),   # +0.29
        (23, 31),  # +0.29
        (29, 11),  # +0.26
        (24, 23),  # +0.25
        # Our original ToM heads
        (12, 0),   # explicit parser
        (23, 0),   # explicit parser
        # Heads with negative change (attend LESS with bridge)
        (17, 15),  # -0.33
        (17, 23),  # -0.29
        (26, 25),  # -0.24
        # Random controls
        (5, 5),
        (10, 10),
        (15, 15),
    ]
    
    # First: get baseline (no ablation)
    print("\n[3/4] Testing baseline (no ablation)...", flush=True)
    ablator.clear_hooks()
    baseline_correct = sum(
        1 for s in scenarios 
        if ablator.test_prompt(s["bridged"], s["correct"], s["wrong"])
    )
    baseline_acc = baseline_correct / len(scenarios)
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    
    # Test each head
    print("\n[4/4] Testing individual heads...", flush=True)
    head_results = []
    
    for layer, head in HEADS_TO_TEST:
        ablator.install_single_ablation(layer, head)
        
        correct = sum(
            1 for s in scenarios 
            if ablator.test_prompt(s["bridged"], s["correct"], s["wrong"])
        )
        acc = correct / len(scenarios)
        drop = baseline_acc - acc
        
        head_results.append({
            "layer": layer,
            "head": head,
            "accuracy": acc,
            "drop": drop,
        })
        
        ablator.clear_hooks()
        print(f"  L{layer:2d}H{head:2d}: {acc:5.1%} (drop: {drop:+.1%})")
    
    # Sort by impact
    head_results.sort(key=lambda x: x["drop"], reverse=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Heads Ranked by Impact (largest drop = most important)")
    print("=" * 70)
    
    print(f"\n  Baseline (no ablation): {baseline_acc:.1%}")
    print("\n  HEAD     ACCURACY    DROP FROM BASELINE")
    print("  " + "-" * 45)
    
    for r in head_results:
        bar = "X" * int(r["drop"] * 50) if r["drop"] > 0 else ""
        marker = ""
        if r["drop"] >= 0.15:
            marker = " <-- HIGH IMPACT!"
        elif r["drop"] <= -0.05:
            marker = " (helps when ablated?)"
        print(f"  L{r['layer']:2d}H{r['head']:2d}:  {r['accuracy']:5.1%}     {r['drop']:+6.1%}  {bar}{marker}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    high_impact = [r for r in head_results if r["drop"] >= 0.10]
    medium_impact = [r for r in head_results if 0.05 <= r["drop"] < 0.10]
    no_impact = [r for r in head_results if -0.05 < r["drop"] < 0.05]
    helps = [r for r in head_results if r["drop"] <= -0.05]
    
    print(f"\n  HIGH IMPACT (>10% drop): {len(high_impact)} heads")
    for r in high_impact:
        print(f"    L{r['layer']}H{r['head']}: {r['drop']:+.1%}")
    
    print(f"\n  MEDIUM IMPACT (5-10% drop): {len(medium_impact)} heads")
    for r in medium_impact:
        print(f"    L{r['layer']}H{r['head']}: {r['drop']:+.1%}")
    
    print(f"\n  NO IMPACT (<5% drop): {len(no_impact)} heads")
    
    if helps:
        print(f"\n  HELPS WHEN ABLATED (negative drop): {len(helps)} heads")
        for r in helps:
            print(f"    L{r['layer']}H{r['head']}: {r['drop']:+.1%}")
    
    # Save results
    output = {
        "baseline_accuracy": baseline_acc,
        "head_results": head_results,
        "high_impact_heads": [(r["layer"], r["head"]) for r in high_impact],
        "n_scenarios": len(scenarios),
    }
    
    with open(RESULTS_DIR / "systematic_head_search_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


