"""
Activation Patching for Causal Tracing
========================================

More rigorous causal test:
1. Run model on "agree" context, save activations
2. Run model on "disagree" context
3. Patch in "agree" activations at specific layers/heads
4. Check if response flips from disagree to agree

This proves the activations CAUSE the behavior.
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

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Source context (agree)
AGREE_CONTEXT = "Agent A: '2+2=4' Agent B verified this is correct. Does B agree with A? B says:"

# Target context (disagree) - we'll patch agree activations into this
DISAGREE_CONTEXT = "Agent A: '2+2=5' Agent B knows this is wrong. Does B agree with A? B says:"


class ActivationPatcher:
    """Perform activation patching experiments."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.saved_activations = {}
        self.patch_config = None
        self.hooks = []
    
    def save_activations(self, prompt: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Run forward pass and save activations."""
        saved = {}
        
        def make_save_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                saved[layer_idx] = hidden.detach().clone()
            return hook
        
        handles = []
        for layer_idx in layers:
            handle = self.model.model.layers[layer_idx].register_forward_hook(make_save_hook(layer_idx))
            handles.append(handle)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for handle in handles:
            handle.remove()
        
        return saved
    
    def patch_and_generate(self, prompt: str, source_activations: Dict[int, torch.Tensor], 
                           patch_layers: List[int], patch_positions: str = "all") -> str:
        """Generate with patched activations."""
        
        def make_patch_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                source = source_activations[layer_idx]
                
                # Match sequence length
                min_len = min(hidden.shape[1], source.shape[1])
                
                if patch_positions == "all":
                    hidden[:, :min_len, :] = source[:, :min_len, :]
                elif patch_positions == "last":
                    hidden[:, -1, :] = source[:, -1, :]
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook
        
        handles = []
        for layer_idx in patch_layers:
            if layer_idx in source_activations:
                handle = self.model.model.layers[layer_idx].register_forward_hook(make_patch_hook(layer_idx))
                handles.append(handle)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        for handle in handles:
            handle.remove()
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def generate_baseline(self, prompt: str) -> str:
        """Generate without patching."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


def check_agreement(response: str) -> str:
    """Check if response indicates agreement or disagreement."""
    response = response.lower()
    if any(w in response for w in ["yes", "agree", "correct", "right"]):
        if not any(w in response for w in ["no", "disagree", "incorrect", "wrong"]):
            return "agree"
    if any(w in response for w in ["no", "disagree", "incorrect", "wrong"]):
        return "disagree"
    return "unclear"


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("ACTIVATION PATCHING EXPERIMENT")
    print("=" * 60)
    
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
    print(f"  [OK] {model.config.num_hidden_layers} layers")
    
    patcher = ActivationPatcher(model, tokenizer)
    
    print("\n[2/5] Getting baseline responses...", flush=True)
    agree_baseline = patcher.generate_baseline(AGREE_CONTEXT)
    disagree_baseline = patcher.generate_baseline(DISAGREE_CONTEXT)
    
    print(f"  Agree context response: '{agree_baseline[:50]}' -> {check_agreement(agree_baseline)}")
    print(f"  Disagree context response: '{disagree_baseline[:50]}' -> {check_agreement(disagree_baseline)}")
    
    print("\n[3/5] Saving agree context activations...", flush=True)
    all_layers = list(range(patcher.n_layers))
    agree_activations = patcher.save_activations(AGREE_CONTEXT, all_layers)
    print(f"  Saved activations from {len(agree_activations)} layers")
    
    print("\n[4/5] Patching experiments...", flush=True)
    
    # Test patching different layer combinations
    layer_groups = {
        "early": [0, 1, 2, 3, 4, 5],
        "mid": [12, 13, 14, 15, 16, 17],
        "tom_identified": [12, 24, 30],
        "late": [30, 31, 32, 33, 34, 35],
        "all": all_layers,
    }
    
    results = {
        "baselines": {
            "agree_context": {"response": agree_baseline, "judgment": check_agreement(agree_baseline)},
            "disagree_context": {"response": disagree_baseline, "judgment": check_agreement(disagree_baseline)},
        },
        "patching_results": [],
    }
    
    for group_name, layers in layer_groups.items():
        print(f"  Testing {group_name} layers ({layers[:3]}...)...", flush=True)
        
        # Patch agree activations into disagree context
        patched_response = patcher.patch_and_generate(
            DISAGREE_CONTEXT, 
            agree_activations,
            layers,
            patch_positions="all"
        )
        
        judgment = check_agreement(patched_response)
        flipped = (judgment == "agree")  # Successfully flipped to agree?
        
        result = {
            "group": group_name,
            "layers": layers,
            "response": patched_response[:50],
            "judgment": judgment,
            "flipped_to_agree": flipped,
        }
        results["patching_results"].append(result)
        
        status = "FLIPPED!" if flipped else "no change"
        print(f"    Response: '{patched_response[:30]}' -> {judgment} ({status})")
    
    print("\n[5/5] Single-layer patching sweep...", flush=True)
    
    single_layer_results = []
    test_layers = [0, 6, 12, 18, 24, 30, 35]
    
    for layer in test_layers:
        patched = patcher.patch_and_generate(
            DISAGREE_CONTEXT,
            agree_activations,
            [layer],
            patch_positions="all"
        )
        judgment = check_agreement(patched)
        single_layer_results.append({
            "layer": layer,
            "response": patched[:30],
            "judgment": judgment,
            "flipped": judgment == "agree",
        })
        print(f"  Layer {layer}: '{patched[:25]}' -> {judgment}")
    
    results["single_layer_patching"] = single_layer_results
    
    # Save results
    with open(RESULTS_DIR / "activation_patching.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. BASELINE BEHAVIOR")
    print("-" * 40)
    print(f"  Agree context: '{agree_baseline[:40]}' -> {check_agreement(agree_baseline)}")
    print(f"  Disagree context: '{disagree_baseline[:40]}' -> {check_agreement(disagree_baseline)}")
    
    print("\n2. PATCHING RESULTS (Agree -> Disagree context)")
    print("-" * 40)
    for r in results["patching_results"]:
        status = "FLIPPED" if r["flipped_to_agree"] else "no flip"
        print(f"  {r['group']:<15}: {r['judgment']:<10} [{status}]")
    
    print("\n3. SINGLE-LAYER PATCHING")
    print("-" * 40)
    flipped_layers = [r["layer"] for r in single_layer_results if r["flipped"]]
    if flipped_layers:
        print(f"  Layers that flip behavior when patched: {flipped_layers}")
    else:
        print("  No single layer flipped behavior")
    
    print("\n4. CAUSAL INTERPRETATION")
    print("-" * 40)
    tom_result = next((r for r in results["patching_results"] if r["group"] == "tom_identified"), None)
    if tom_result and tom_result["flipped_to_agree"]:
        print("  ToM layers (12, 24, 30) CAUSALLY CONTROL agreement behavior!")
    else:
        print("  Mixed results - behavior may be distributed")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n" + "=" * 60)
    print(f"TOTAL TIME: {total_time:.1f}s")
    print("=" * 60)
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'activation_patching.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()






















