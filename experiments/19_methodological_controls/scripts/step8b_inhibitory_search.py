"""
INHIBITORY HEAD SEARCH: Find ALL heads that suppress ToM
=========================================================

Thorough search through layers 15-35 for heads like L24H29 that
suppress belief update inference when active.

Search strategy:
- Test ALL 32 heads in key layers (22-28)
- Test every 4th head in surrounding layers (15-21, 29-35)
- Print every result for full visibility
- Time estimates for planning
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


class InhibitorySearcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        
    def _create_ablation_hook(self, head_idx: int):
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            reshaped[:, :, head_idx, :] = 0
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def install_ablation(self, layer_idx: int, head_idx: int):
        self.clear_hooks()
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_idx))
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


def generate_scenarios(n: int = 25) -> list:
    """Generate belief update scenarios. Smaller N for faster search."""
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
    start_time = time.time()
    
    print("=" * 70)
    print("INHIBITORY HEAD SEARCH")
    print("=" * 70)
    print("Finding ALL heads that suppress ToM (help when ablated)")
    print()
    
    # Load model
    print("[1/3] Loading model...", flush=True)
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
    
    searcher = InhibitorySearcher(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/3] Generating test scenarios...", flush=True)
    scenarios = generate_scenarios(25)  # 25 for speed
    print(f"  {len(scenarios)} scenarios", flush=True)
    
    # Get baseline
    print("\n[3/3] Starting head search...", flush=True)
    searcher.clear_hooks()
    baseline_acc = searcher.test_batch(scenarios)
    print(f"  Baseline (no ablation): {baseline_acc:.1%}")
    
    # Build search plan:
    # - Layers 22-28: ALL 32 heads (core ToM area around L24)
    # - Layers 15-21: every 4th head
    # - Layers 29-35: every 4th head
    search_plan = []
    
    # Core layers: all heads
    for layer in range(22, 29):  # 22-28 inclusive
        for head in range(32):
            search_plan.append((layer, head))
    
    # Early layers: sparse
    for layer in range(15, 22):
        for head in range(0, 32, 4):
            search_plan.append((layer, head))
    
    # Late layers: sparse  
    for layer in range(29, 36):
        for head in range(0, 32, 4):
            search_plan.append((layer, head))
    
    total_tests = len(search_plan)
    print(f"\n  Search plan: {total_tests} heads to test")
    print(f"  Estimated time: ~{total_tests * 1.5 / 60:.1f} minutes")
    print()
    
    results = []
    inhibitors_found = []
    
    print("  " + "-" * 55)
    print("  LAYER  HEAD   ACC     BOOST    STATUS")
    print("  " + "-" * 55)
    
    for i, (layer, head) in enumerate(search_plan):
        test_start = time.time()
        
        searcher.install_ablation(layer, head)
        acc = searcher.test_batch(scenarios)
        searcher.clear_hooks()
        
        boost = acc - baseline_acc
        
        results.append({
            "layer": layer,
            "head": head,
            "accuracy": float(acc),
            "boost": float(boost),
        })
        
        # Determine status
        status = ""
        if boost > 0.20:
            status = "*** STRONG INHIBITOR ***"
            inhibitors_found.append(f"L{layer}H{head}")
        elif boost > 0.10:
            status = "** inhibitor **"
            inhibitors_found.append(f"L{layer}H{head}")
        elif boost < -0.15:
            status = "(critical - hurts)"
        elif boost < -0.08:
            status = "(important)"
        
        # Progress indicator
        pct_done = (i + 1) / total_tests * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_tests - i - 1)
        
        # Print every result in core layers, or interesting ones elsewhere
        is_core_layer = 22 <= layer <= 28
        is_interesting = abs(boost) > 0.06
        
        if is_core_layer or is_interesting or (i + 1) % 20 == 0:
            print(f"  L{layer:2d}    H{head:2d}    {acc:5.1%}   {boost:+5.1%}   {status}", flush=True)
        
        # Progress update every 50 tests
        if (i + 1) % 50 == 0:
            print(f"  --- Progress: {pct_done:.0f}% ({i+1}/{total_tests}), ETA: {eta/60:.1f}m ---", flush=True)
    
    # Sort by boost (highest first)
    results.sort(key=lambda x: x["boost"], reverse=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Heads tested: {len(results)}")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    
    print("\n" + "-" * 50)
    print("TOP 15 INHIBITORY HEADS (ablating HELPS)")
    print("-" * 50)
    
    for i, r in enumerate(results[:15]):
        marker = ""
        if r["boost"] > 0.15:
            marker = " <-- STRONG"
        elif r["boost"] > 0.08:
            marker = " <-- moderate"
        print(f"  {i+1:2d}. L{r['layer']:2d}H{r['head']:2d}: {r['accuracy']:.1%} (boost: {r['boost']:+.1%}){marker}")
    
    print("\n" + "-" * 50)
    print("TOP 10 CRITICAL HEADS (ablating HURTS)")
    print("-" * 50)
    
    for i, r in enumerate(results[-10:]):
        print(f"  L{r['layer']:2d}H{r['head']:2d}: {r['accuracy']:.1%} (boost: {r['boost']:+.1%})")
    
    # Layer 24 specifically (where we found the inhibitor)
    l24_results = [r for r in results if r["layer"] == 24]
    l24_results.sort(key=lambda x: x["head"])
    
    print("\n" + "-" * 50)
    print("LAYER 24 DETAILED (where L24H29 was found)")
    print("-" * 50)
    
    for r in l24_results:
        marker = " <-- KNOWN INHIBITOR" if r["head"] == 29 else ""
        if r["boost"] > 0.08:
            marker = " <-- INHIBITOR"
        print(f"  H{r['head']:2d}: {r['accuracy']:.1%} ({r['boost']:+.1%}){marker}")
    
    # Save results
    output = {
        "baseline": float(baseline_acc),
        "n_scenarios": len(scenarios),
        "n_heads_tested": len(results),
        "search_time_minutes": (time.time() - start_time) / 60,
        "top_inhibitors": results[:15],
        "top_critical": results[-10:],
        "layer_24": l24_results,
        "all_results": results,
    }
    
    with open(RESULTS_DIR / "inhibitory_search_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'inhibitory_search_results.json'}")
    
    # Key finding
    if inhibitors_found:
        print("\n" + "=" * 70)
        print(f"INHIBITORY HEADS FOUND: {', '.join(inhibitors_found)}")
        print("=" * 70)
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
