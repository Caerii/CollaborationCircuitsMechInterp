"""
COMPLETE SYSTEMATIC HEAD SEARCH
===============================

Full coverage of ALL 40 layers × 32 heads = 1280 heads.

Strategy for speed:
- Phase 1: Test all 1280 heads with N=15 scenarios (fast scan)
- Phase 2: Re-test top candidates with N=40 for confirmation

This gives us COMPLETE, systematic coverage for MATS.
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


class HeadSearcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        self.n_layers = model.config.num_hidden_layers
        
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


def generate_scenarios(n: int, seed: int = 42) -> list:
    random.seed(seed)
    scenarios = []
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    INFORMERS = ["Grace", "Henry", "Iris", "Jack", "Kate", "Leo"]
    OBJECTS = ["ball", "key", "book", "toy", "pen", "hat"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "box", "bag"]
    
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice(INFORMERS)
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
    print("COMPLETE SYSTEMATIC HEAD SEARCH")
    print("=" * 70)
    print("Testing ALL 40 layers x 32 heads = 1280 heads")
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
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  Model: {n_layers} layers, {n_heads} heads/layer", flush=True)
    print(f"  Total heads: {n_layers * n_heads}", flush=True)
    
    searcher = HeadSearcher(model, tokenizer)
    
    # Phase 1: Fast scan with N=15
    print("\n[2/4] PHASE 1: Fast scan (N=15 scenarios)...", flush=True)
    scenarios_fast = generate_scenarios(15)
    
    searcher.clear_hooks()
    baseline = searcher.test_batch(scenarios_fast)
    print(f"  Baseline: {baseline:.1%}")
    
    total_heads = n_layers * n_heads
    print(f"  Estimated time: ~{total_heads * 0.8 / 60:.0f} minutes")
    print()
    
    all_results = []
    layer_summaries = {}
    
    print("  LAYER | BEST HEAD | BOOST | WORST HEAD | BOOST")
    print("  " + "-" * 55)
    
    for layer in range(n_layers):
        layer_results = []
        
        for head in range(n_heads):
            searcher.install_ablation(layer, head)
            acc = searcher.test_batch(scenarios_fast)
            searcher.clear_hooks()
            
            boost = acc - baseline
            
            result = {
                "layer": layer,
                "head": head,
                "accuracy": float(acc),
                "boost": float(boost),
            }
            all_results.append(result)
            layer_results.append(result)
        
        # Summarize this layer
        layer_results.sort(key=lambda x: x["boost"], reverse=True)
        best = layer_results[0]
        worst = layer_results[-1]
        
        layer_summaries[layer] = {
            "best": best,
            "worst": worst,
            "avg_boost": np.mean([r["boost"] for r in layer_results]),
        }
        
        # Print progress
        best_marker = "***" if best["boost"] > 0.20 else "**" if best["boost"] > 0.10 else ""
        print(f"  L{layer:2d}   |   H{best['head']:2d}    | {best['boost']:+5.1%} {best_marker:3s} |   H{worst['head']:2d}    | {worst['boost']:+5.1%}", flush=True)
    
    # Sort all results
    all_results.sort(key=lambda x: x["boost"], reverse=True)
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE - TOP 30 INHIBITORY HEADS")
    print("=" * 70)
    
    for i, r in enumerate(all_results[:30]):
        marker = "STRONG" if r["boost"] > 0.25 else "moderate" if r["boost"] > 0.15 else ""
        print(f"  {i+1:2d}. L{r['layer']:2d}H{r['head']:2d}: {r['accuracy']:.1%} ({r['boost']:+.1%}) {marker}")
    
    print("\n" + "-" * 50)
    print("TOP 10 CRITICAL HEADS (ablating hurts)")
    print("-" * 50)
    
    for i, r in enumerate(all_results[-10:]):
        print(f"  L{r['layer']:2d}H{r['head']:2d}: {r['accuracy']:.1%} ({r['boost']:+.1%})")
    
    # Phase 2: Validate top 20 with larger N
    print("\n[3/4] PHASE 2: Validating top 20 with N=30...", flush=True)
    scenarios_valid = generate_scenarios(30, seed=123)  # Different seed
    
    searcher.clear_hooks()
    baseline_valid = searcher.test_batch(scenarios_valid)
    print(f"  Validation baseline: {baseline_valid:.1%}")
    
    validated_results = []
    
    for r in all_results[:20]:
        searcher.install_ablation(r["layer"], r["head"])
        acc = searcher.test_batch(scenarios_valid)
        searcher.clear_hooks()
        
        validated_results.append({
            "layer": r["layer"],
            "head": r["head"],
            "phase1_acc": r["accuracy"],
            "phase1_boost": r["boost"],
            "phase2_acc": float(acc),
            "phase2_boost": float(acc - baseline_valid),
        })
        print(f"  L{r['layer']:2d}H{r['head']:2d}: Phase1={r['accuracy']:.1%}, Phase2={acc:.1%}", flush=True)
    
    # Also validate bottom 10
    print("\n  Validating bottom 10 (critical heads)...", flush=True)
    critical_validated = []
    
    for r in all_results[-10:]:
        searcher.install_ablation(r["layer"], r["head"])
        acc = searcher.test_batch(scenarios_valid)
        searcher.clear_hooks()
        
        critical_validated.append({
            "layer": r["layer"],
            "head": r["head"],
            "phase1_acc": r["accuracy"],
            "phase1_boost": r["boost"],
            "phase2_acc": float(acc),
            "phase2_boost": float(acc - baseline_valid),
        })
        print(f"  L{r['layer']:2d}H{r['head']:2d}: Phase1={r['accuracy']:.1%}, Phase2={acc:.1%}", flush=True)
    
    # Final summary
    print("\n" + "=" * 70)
    print("[4/4] FINAL SUMMARY")
    print("=" * 70)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n  Total time: {total_time:.1f} minutes")
    print(f"  Heads tested: {len(all_results)}")
    print(f"  Phase 1 baseline: {baseline:.1%}")
    print(f"  Phase 2 baseline: {baseline_valid:.1%}")
    
    # Layer distribution of inhibitors
    print("\n  LAYER DISTRIBUTION OF INHIBITORS (boost > 10%):")
    inhibitor_counts = {}
    for r in all_results:
        if r["boost"] > 0.10:
            l = r["layer"]
            inhibitor_counts[l] = inhibitor_counts.get(l, 0) + 1
    
    for layer in sorted(inhibitor_counts.keys()):
        bar = "#" * inhibitor_counts[layer]
        print(f"    L{layer:2d}: {bar} ({inhibitor_counts[layer]})")
    
    # Save comprehensive results
    output = {
        "model": MODEL_CFG.model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads_tested": len(all_results),
        "phase1_baseline": float(baseline),
        "phase1_n_scenarios": 15,
        "phase2_baseline": float(baseline_valid),
        "phase2_n_scenarios": 30,
        "search_time_minutes": total_time,
        "top_30_inhibitors": all_results[:30],
        "bottom_10_critical": all_results[-10:],
        "validated_inhibitors": validated_results,
        "validated_critical": critical_validated,
        "layer_summaries": {str(k): v for k, v in layer_summaries.items()},
        "inhibitor_layer_distribution": inhibitor_counts,
        "all_results": all_results,
    }
    
    with open(RESULTS_DIR / "complete_head_search.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'complete_head_search.json'}")
    
    # Key finding
    print("\n" + "=" * 70)
    print("KEY VALIDATED INHIBITORS")
    print("=" * 70)
    
    # Sort validated by phase2 boost
    validated_results.sort(key=lambda x: x["phase2_boost"], reverse=True)
    
    for i, r in enumerate(validated_results[:10]):
        consistent = "CONSISTENT" if abs(r["phase1_boost"] - r["phase2_boost"]) < 0.15 else "varies"
        print(f"  {i+1}. L{r['layer']:2d}H{r['head']:2d}: {r['phase2_acc']:.1%} ({r['phase2_boost']:+.1%}) [{consistent}]")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

