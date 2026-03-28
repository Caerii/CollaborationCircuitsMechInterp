"""
PROPER Attention Head Ablation
===============================

This script correctly ablates attention heads by:
1. Hooking INSIDE the attention computation
2. Modifying the attention output BEFORE o_proj combines heads
3. Using ToM-specific metrics (belief flip, not just output change)

Qwen3 Attention Architecture:
- q_proj, k_proj, v_proj: Project to (batch, seq, num_heads, head_dim)
- Attention computed: (batch, num_heads, seq, seq)
- Output: (batch, seq, num_heads, head_dim)
- o_proj: Combines heads back to (batch, seq, hidden_size)

We need to hook BEFORE o_proj to properly ablate individual heads.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = PROJECT_ROOT / "experiments" / "14_rigorous_reboot" / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ToM test prompts with expected belief vs reality completions (EXPANDED for statistical power)
OBJECTS = ["ball", "book", "keys", "phone", "wallet", "cake", "toy", "letter", "ring", "bag"]
LOCATIONS1 = ["basket", "drawer", "box", "shelf", "cupboard", "table", "desk", "closet", "bag", "pocket"]
LOCATIONS2 = ["box", "shelf", "basket", "drawer", "fridge", "couch", "chair", "bed", "counter", "bench"]
NAMES = [("Alice", "Bob"), ("Carol", "David"), ("Emma", "Frank"), ("Grace", "Henry"), 
         ("Ivan", "Julia"), ("Kate", "Leo"), ("Mia", "Noah"), ("Olivia", "Paul"),
         ("Quinn", "Ryan"), ("Sara", "Tom")]

TOM_TEST_PROMPTS = []
for i in range(50):  # Generate 50 prompts for statistical power
    obj = OBJECTS[i % len(OBJECTS)]
    loc1 = LOCATIONS1[i % len(LOCATIONS1)]
    loc2 = LOCATIONS2[i % len(LOCATIONS2)]
    if loc1 == loc2:  # Avoid same location
        loc2 = LOCATIONS2[(i + 1) % len(LOCATIONS2)]
    name1, name2 = NAMES[i % len(NAMES)]
    
    TOM_TEST_PROMPTS.append({
        "id": f"tom{i+1}",
        "prompt": f"{name1} puts the {obj} in the {loc1}. {name1} leaves. {name2} moves the {obj} to the {loc2}. {name1} returns. {name1} will look for the {obj} in the",
        "belief_completion": f" {loc1}",
        "reality_completion": f" {loc2}",
    })


class ProperAttentionAblator:
    """
    Correctly ablates attention heads by modifying attention output before o_proj.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.hooks = []
        self.ablation_config = None
    
    def _find_attention_hook_point(self, layer_idx: int):
        """
        Find the correct module to hook for attention output before o_proj.
        
        In Qwen3, we need to hook the attention output before it goes to o_proj.
        The attention module structure varies by implementation.
        """
        attn = self.model.model.layers[layer_idx].self_attn
        
        # Check what submodules exist
        submodules = dict(attn.named_modules())
        
        # We want to hook right after attention computation, before o_proj
        # In most implementations, this is accessing the intermediate result
        return attn
    
    def _create_attention_ablation_hook(self, head_idx: int):
        """
        Create a hook that modifies attention BEFORE o_proj.
        
        For Qwen3 with GQA:
        - Input to o_proj has shape (batch, seq, num_attention_heads * head_dim)
        - For Qwen3-4B: (batch, seq, 32 * 128) = (batch, seq, 4096)
        - head_dim at this stage is 128, not 80
        """
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            
            # The o_proj input dimension tells us the actual head_dim at this stage
            actual_head_dim = hidden // self.n_heads
            
            # Reshape to (batch, seq, num_heads, actual_head_dim)
            reshaped = hidden_states.view(batch, seq_len, self.n_heads, actual_head_dim)
            
            # Zero out the specific head
            reshaped[:, :, head_idx, :] = 0
            
            # Reshape back
            modified = reshaped.view(batch, seq_len, hidden)
            
            return (modified,) + args[1:] if len(args) > 1 else (modified,)
        
        return hook
    
    def install_ablation(self, layer_idx: int, head_idx: int):
        """Install ablation hook on o_proj's INPUT (the attention output)."""
        self.clear_hooks()
        
        # Hook o_proj's input (this is the attention output before combination)
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        
        # Use register_forward_pre_hook to modify input before o_proj processes it
        hook = o_proj.register_forward_pre_hook(self._create_attention_ablation_hook(head_idx))
        self.hooks.append(hook)
        self.ablation_config = (layer_idx, head_idx)
    
    def install_multi_ablation(self, configs: list):
        """Install ablation hooks on multiple heads at once."""
        self.clear_hooks()
        
        for layer_idx, head_idx in configs:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_attention_ablation_hook(head_idx))
            self.hooks.append(hook)
        
        self.ablation_config = configs
    
    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.ablation_config = None
    
    def get_belief_preference(self, prompt: str, belief_completion: str, reality_completion: str) -> Tuple[str, float, float]:
        """
        Measure whether model prefers belief-based or reality-based completion.
        
        Returns: (preference, belief_logprob, reality_logprob)
        """
        def get_completion_logprob(prompt, completion):
            full_text = prompt + completion
            inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            log_probs = torch.log_softmax(logits[0], dim=-1)
            
            completion_tokens = inputs.input_ids[0, prompt_tokens:]
            total_logprob = 0
            for i, token_id in enumerate(completion_tokens):
                pos = prompt_tokens + i - 1
                if pos >= 0:
                    total_logprob += log_probs[pos, token_id].item()
            
            return total_logprob
        
        belief_logprob = get_completion_logprob(prompt, belief_completion)
        reality_logprob = get_completion_logprob(prompt, reality_completion)
        
        preference = "belief" if belief_logprob > reality_logprob else "reality"
        
        return preference, belief_logprob, reality_logprob


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("PROPER ATTENTION HEAD ABLATION")
    print("=" * 60)
    print("\nHooking o_proj INPUT (attention output before head combination)")
    
    print("\n[1/5] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  [OK] {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    
    ablator = ProperAttentionAblator(model, tokenizer)
    
    print("\n[2/5] Getting baseline belief preferences...", flush=True)
    baselines = {}
    for test in TOM_TEST_PROMPTS:
        pref, belief_lp, reality_lp = ablator.get_belief_preference(
            test["prompt"], test["belief_completion"], test["reality_completion"]
        )
        baselines[test["id"]] = {
            "preference": pref,
            "belief_logprob": belief_lp,
            "reality_logprob": reality_lp,
            "margin": belief_lp - reality_lp,
        }
        print(f"  {test['id']}: {pref} (margin: {belief_lp - reality_lp:.2f})")
    
    print("\n[3/5] Ablation sweep (ToM-specific metric)...", flush=True)
    
    # Test layers and heads - focused set for N=50 prompts
    test_configs = [
        # Previously "identified" heads
        (12, 0), (24, 0), (30, 0), (23, 0),
        # Control heads (matched layers, different head)
        (12, 16), (24, 16), (30, 16), (23, 16),
        # Early layer control
        (3, 0), (3, 16),
        # Late layer control  
        (33, 0), (33, 16),
    ]
    
    results = {
        "baselines": baselines,
        "ablations": [],
    }
    
    for layer_idx, head_idx in test_configs:
        print(f"\n  Testing L{layer_idx}H{head_idx}...", flush=True)
        
        ablator.install_ablation(layer_idx, head_idx)
        
        ablation_result = {
            "layer": layer_idx,
            "head": head_idx,
            "tests": [],
            "belief_flips": 0,  # KEY METRIC: Did belief preference flip to reality?
        }
        
        for test in TOM_TEST_PROMPTS:
            pref, belief_lp, reality_lp = ablator.get_belief_preference(
                test["prompt"], test["belief_completion"], test["reality_completion"]
            )
            
            baseline_pref = baselines[test["id"]]["preference"]
            flipped = (baseline_pref == "belief" and pref == "reality")
            
            if flipped:
                ablation_result["belief_flips"] += 1
            
            ablation_result["tests"].append({
                "id": test["id"],
                "preference": pref,
                "margin": belief_lp - reality_lp,
                "baseline_margin": baselines[test["id"]]["margin"],
                "flipped_to_reality": flipped,
            })
        
        ablation_result["flip_rate"] = ablation_result["belief_flips"] / len(TOM_TEST_PROMPTS)
        results["ablations"].append(ablation_result)
        
        print(f"    Belief->Reality flips: {ablation_result['belief_flips']}/{len(TOM_TEST_PROMPTS)} ({ablation_result['flip_rate']:.0%})")
        
        ablator.clear_hooks()
    
    print("\n[4/5] Multi-head ablation test...", flush=True)
    
    # Test ablating BOTH L12H0 and L23H0 together
    multi_configs = [
        ([(12, 0), (23, 0)], "L12H0+L23H0"),
        ([(12, 0), (23, 0), (24, 0), (30, 0)], "All_ToM_Heads"),
    ]
    
    for heads, label in multi_configs:
        print(f"\n  Testing {label}...", flush=True)
        ablator.install_multi_ablation(heads)
        
        flips = 0
        for test in TOM_TEST_PROMPTS:
            pref, belief_lp, reality_lp = ablator.get_belief_preference(
                test["prompt"], test["belief_completion"], test["reality_completion"]
            )
            baseline_pref = baselines[test["id"]]["preference"]
            if baseline_pref == "belief" and pref == "reality":
                flips += 1
        
        flip_rate = flips / len(TOM_TEST_PROMPTS)
        results["ablations"].append({
            "layer": label,
            "head": "multi",
            "belief_flips": flips,
            "flip_rate": flip_rate,
            "tests": [],
        })
        print(f"    Belief->Reality flips: {flips}/{len(TOM_TEST_PROMPTS)} ({flip_rate:.0%})")
        ablator.clear_hooks()
    
    print("\n[5/5] Statistical analysis...", flush=True)
    
    from scipy import stats
    
    # Compare "ToM heads" vs others
    tom_heads = [(12, 0), (24, 0), (30, 0), (23, 0)]
    tom_flips = [a["flip_rate"] for a in results["ablations"] if (a["layer"], a["head"]) in tom_heads]
    other_flips = [a["flip_rate"] for a in results["ablations"] if (a["layer"], a["head"]) not in tom_heads]
    
    if tom_flips and other_flips:
        _, p_value = stats.mannwhitneyu(tom_flips, other_flips, alternative='greater')
        results["statistics"] = {
            "tom_mean_flip_rate": float(np.mean(tom_flips)),
            "other_mean_flip_rate": float(np.mean(other_flips)),
            "tom_vs_other_pvalue": float(p_value),
        }
        print(f"  ToM heads flip rate: {np.mean(tom_flips):.0%}")
        print(f"  Other heads flip rate: {np.mean(other_flips):.0%}")
        print(f"  p-value: {p_value:.4f}")
    
    print("\n[6/6] Saving results...", flush=True)
    
    with open(RESULTS_DIR / "proper_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. BASELINE BELIEF PREFERENCES")
    print("-" * 40)
    for test_id, baseline in baselines.items():
        print(f"  {test_id}: {baseline['preference']} (margin: {baseline['margin']:.2f})")
    
    print("\n2. ABLATION EFFECTS (Belief->Reality Flip Rate)")
    print("-" * 40)
    for a in sorted(results["ablations"], key=lambda x: -x["flip_rate"]):
        tom_marker = "*" if (a["layer"], a["head"]) in tom_heads else " "
        print(f"  {tom_marker} L{a['layer']}H{a['head']}: {a['flip_rate']:.0%} ({a['belief_flips']}/{len(TOM_TEST_PROMPTS)})")
    
    print("\n3. KEY FINDING")
    print("-" * 40)
    max_flip = max(results["ablations"], key=lambda x: x["flip_rate"])
    if max_flip["flip_rate"] > 0:
        print(f"  Head L{max_flip['layer']}H{max_flip['head']} causes {max_flip['flip_rate']:.0%} belief->reality flips")
        print("  -> This head may be involved in ToM processing!")
    else:
        print("  NO heads caused belief->reality flips")
        print("  -> ToM computation may be distributed or resilient to single-head ablation")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'proper_ablation_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

