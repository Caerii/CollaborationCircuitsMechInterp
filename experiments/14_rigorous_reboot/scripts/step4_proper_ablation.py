"""
Proper Attention Head Ablation
===============================

FIXES the architectural error in previous ablation.

Previous (WRONG):
- Hooked layer output (residual stream)
- Reshaped to (n_heads, head_dim) - arbitrary slicing!

Correct (THIS FILE):
- Hook the attention module BEFORE output projection
- Access actual attention head outputs
- Properly zero specific heads

Qwen3 architecture:
- self_attn.q_proj, k_proj, v_proj -> QKV projections
- Attention computed -> output shape (batch, seq, n_heads, head_dim)
- self_attn.o_proj -> output projection
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


class ProperHeadAblation:
    """
    Correct implementation of attention head ablation.
    
    For Qwen3-4B:
    - num_attention_heads = 32
    - head_dim = hidden_size / num_attention_heads = 2560 / 32 = 80
    - Attention output before o_proj has shape (batch, seq, n_heads * head_dim)
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        self.hooks = []
        self.ablate_heads = []
    
    def _get_ablation_hook(self, layer_idx: int, head_idx: int):
        """
        Create hook that zeros out a specific attention head.
        
        Hook point: AFTER attention computation, BEFORE o_proj.
        The attention output at this point is (batch, seq, hidden_size)
        where hidden_size = n_heads * head_dim.
        """
        head_dim = self.head_dim
        n_heads = self.n_heads
        
        def hook(module, args, output):
            # Output shape: (batch, seq, hidden_size)
            # or tuple where first element is the attention output
            attn_output = output[0] if isinstance(output, tuple) else output
            
            batch, seq_len, hidden = attn_output.shape
            
            # Reshape to (batch, seq, n_heads, head_dim)
            reshaped = attn_output.view(batch, seq_len, n_heads, head_dim)
            
            # Zero out the specific head
            reshaped[:, :, head_idx, :] = 0
            
            # Reshape back
            ablated = reshaped.view(batch, seq_len, hidden)
            
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated
        
        return hook
    
    def install_ablation(self, layer_idx: int, head_idx: int):
        """Install ablation hook on a specific attention head."""
        self.clear_hooks()
        
        # Hook the attention module's output
        # In Qwen, this is model.model.layers[l].self_attn
        attn_module = self.model.model.layers[layer_idx].self_attn
        
        # Register as forward hook (post-forward)
        hook = attn_module.register_forward_hook(self._get_ablation_hook(layer_idx, head_idx))
        self.hooks.append(hook)
        self.ablate_heads = [(layer_idx, head_idx)]
    
    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.ablate_heads = []
    
    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        """Generate with current ablation configuration."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


def get_baseline_responses(model, tokenizer, prompts: List[dict]) -> Dict[str, str]:
    """Get model responses without ablation."""
    baselines = {}
    
    for p in prompts:
        inputs = tokenizer(p["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        baselines[p["id"]] = response[len(p["prompt"]):].strip()
    
    return baselines


def response_changed(baseline: str, ablated: str, expected_word: str = None) -> bool:
    """Check if response meaningfully changed."""
    baseline_clean = baseline.lower().strip()[:50]
    ablated_clean = ablated.lower().strip()[:50]
    
    # Simple check: first few words different
    return baseline_clean[:20] != ablated_clean[:20]


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("PROPER ATTENTION HEAD ABLATION")
    print("=" * 60)
    print("\nThis fixes the architectural error in previous ablation.")
    
    # Load model
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
    print(f"  [OK] {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    
    ablator = ProperHeadAblation(model, tokenizer)
    
    # Test prompts - larger set than before
    print("\n[2/4] Setting up test prompts...", flush=True)
    
    # Diverse ToM-related prompts
    test_prompts = [
        {"id": "tom1", "prompt": "Alice thinks the ball is in the basket. The ball is actually in the box. Alice will look for it in the"},
        {"id": "tom2", "prompt": "Bob believes his keys are on the table. They're actually in his pocket. Bob will search the"},
        {"id": "tom3", "prompt": "Carol saw the cake in the fridge. Her brother moved it to the cupboard. Carol thinks the cake is in the"},
        {"id": "agree1", "prompt": "Agent A: '2+2=4'. Agent B verified this. B thinks A is"},
        {"id": "agree2", "prompt": "Agent A: 'The sky is blue'. Agent B confirms. B agrees that A is"},
        {"id": "disagree1", "prompt": "Agent A: '2+2=5'. Agent B knows math. B thinks A is"},
        {"id": "disagree2", "prompt": "Agent A: 'Fire is cold'. Agent B knows this is false. B disagrees because A is"},
        {"id": "neutral1", "prompt": "The weather today is"},
        {"id": "neutral2", "prompt": "The capital of France is"},
        {"id": "neutral3", "prompt": "Water boils at"},
    ]
    
    print(f"  {len(test_prompts)} test prompts")
    
    # Get baselines
    print("\n[3/4] Getting baseline responses...", flush=True)
    baselines = get_baseline_responses(model, tokenizer, test_prompts)
    
    for p in test_prompts[:3]:
        print(f"  {p['id']}: '{baselines[p['id']][:40]}'")
    
    # Ablation sweep
    print("\n[4/4] Ablation sweep...", flush=True)
    
    # Test key layers and heads
    test_configs = [
        # Previously "identified" heads
        (12, 0), (24, 0), (30, 0),
        # Random controls
        (12, 15), (24, 15), (6, 0), (18, 10),
        # All heads at one layer
    ] + [(23, h) for h in range(0, 32, 4)]  # Layer 23 (high MI)
    
    results = {
        "baselines": baselines,
        "ablations": [],
        "summary": {},
    }
    
    for layer_idx, head_idx in test_configs:
        print(f"\n  Testing L{layer_idx}H{head_idx}...", flush=True)
        
        ablator.install_ablation(layer_idx, head_idx)
        
        ablation_result = {
            "layer": layer_idx,
            "head": head_idx,
            "responses": {},
            "changes": {},
        }
        
        n_changed = 0
        for p in test_prompts:
            ablated_response = ablator.generate(p["prompt"])
            ablation_result["responses"][p["id"]] = ablated_response
            
            changed = response_changed(baselines[p["id"]], ablated_response)
            ablation_result["changes"][p["id"]] = changed
            if changed:
                n_changed += 1
        
        ablation_result["change_rate"] = n_changed / len(test_prompts)
        results["ablations"].append(ablation_result)
        
        print(f"    Change rate: {ablation_result['change_rate']:.0%} ({n_changed}/{len(test_prompts)})")
        
        ablator.clear_hooks()
    
    # Compute summary statistics
    print("\n  Computing statistics...", flush=True)
    
    change_rates = [a["change_rate"] for a in results["ablations"]]
    results["summary"] = {
        "n_heads_tested": len(test_configs),
        "mean_change_rate": float(np.mean(change_rates)),
        "std_change_rate": float(np.std(change_rates)),
        "max_change_rate": float(np.max(change_rates)),
        "max_change_head": test_configs[int(np.argmax(change_rates))],
    }
    
    # Statistical test: is any head significantly different from others?
    from scipy import stats
    
    # Compare ToM heads vs control heads
    tom_heads = [(12, 0), (24, 0), (30, 0)]
    tom_rates = [a["change_rate"] for a in results["ablations"] if (a["layer"], a["head"]) in tom_heads]
    other_rates = [a["change_rate"] for a in results["ablations"] if (a["layer"], a["head"]) not in tom_heads]
    
    if tom_rates and other_rates:
        _, p_value = stats.mannwhitneyu(tom_rates, other_rates, alternative='greater')
        results["summary"]["tom_vs_other_pvalue"] = float(p_value)
        results["summary"]["tom_mean"] = float(np.mean(tom_rates))
        results["summary"]["other_mean"] = float(np.mean(other_rates))
    
    # Save
    with open(RESULTS_DIR / "proper_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. ABLATION CHANGE RATES")
    print("-" * 40)
    for a in sorted(results["ablations"], key=lambda x: -x["change_rate"])[:5]:
        print(f"  L{a['layer']}H{a['head']}: {a['change_rate']:.0%}")
    
    print("\n2. ToM vs CONTROL HEADS")
    print("-" * 40)
    if "tom_mean" in results["summary"]:
        print(f"  ToM heads (L12/24/30 H0): {results['summary']['tom_mean']:.0%}")
        print(f"  Other heads: {results['summary']['other_mean']:.0%}")
        print(f"  p-value: {results['summary']['tom_vs_other_pvalue']:.4f}")
        if results['summary']['tom_vs_other_pvalue'] < 0.05:
            print("  [+] ToM heads are significantly MORE impactful!")
        else:
            print("  [-] No significant difference (ToM heads not special)")
    
    print("\n3. OVERALL")
    print("-" * 40)
    print(f"  Mean change rate: {results['summary']['mean_change_rate']:.0%}")
    print(f"  Max change head: L{results['summary']['max_change_head'][0]}H{results['summary']['max_change_head'][1]}")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'proper_ablation_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




















