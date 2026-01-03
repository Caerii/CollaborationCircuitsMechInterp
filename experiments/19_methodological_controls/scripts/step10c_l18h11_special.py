"""
L18H11 IS SPECIAL: The Robust Correct-Pusher
=============================================

Finding: L18H11 is the ONLY head that pushes toward correct
even when the model predicts incorrectly.

Hypothesis: L18H11 might be HELPING, not inhibiting.
We should test what happens when we:
1. Ablate ONLY L17H4 + L18H14 (not L18H11)
2. Ablate L18H11 alone
3. AMPLIFY L18H11

If L18H11 is actually helping, ablating it should HURT performance.
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"


class SpecialHeadTester:
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
                reshaped[:, :, head_idx, :] = 0
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def _create_amplification_hook(self, head_idx: int, scale: float):
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            reshaped[:, :, head_idx, :] = reshaped[:, :, head_idx, :] * scale
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def install_ablation(self, layer_head_pairs: list):
        self.clear_hooks()
        layer_to_heads = {}
        for layer, head in layer_head_pairs:
            if layer not in layer_to_heads:
                layer_to_heads[layer] = []
            layer_to_heads[layer].append(head)
        
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_indices))
            self.hooks.append(hook)
    
    def install_amplification(self, layer: int, head: int, scale: float):
        self.clear_hooks()
        o_proj = self.model.model.layers[layer].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(self._create_amplification_hook(head, scale))
        self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test(self, scenarios: list) -> float:
        correct = 0
        for s in scenarios:
            inputs = self.tokenizer(s["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
            correct_id = self.tokenizer.encode(s["correct"], add_special_tokens=False)[0]
            wrong_id = self.tokenizer.encode(s["wrong"], add_special_tokens=False)[0]
            if logits[correct_id] > logits[wrong_id]:
                correct += 1
        return correct / len(scenarios)


def main():
    print("=" * 70)
    print("L18H11 IS SPECIAL: Testing The Hypothesis")
    print("=" * 70)
    
    # Load scenarios
    with open(RESULTS_DIR / "fixed_scenarios.json") as f:
        scenarios = json.load(f)
    
    # Load model
    print("\nLoading model...", flush=True)
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
    
    tester = SpecialHeadTester(model, tokenizer)
    
    results = {}
    
    # Baseline
    print("\n[1/7] Baseline...", flush=True)
    tester.clear_hooks()
    baseline = tester.test(scenarios)
    results["baseline"] = baseline
    print(f"  Baseline: {baseline:.1%}")
    
    # Test 1: Ablate L17H4 + L18H14 (NOT L18H11)
    print("\n[2/7] Ablate L17H4 + L18H14 only (keep L18H11)...", flush=True)
    tester.install_ablation([(17, 4), (18, 14)])
    acc = tester.test(scenarios)
    results["ablate_17_4_and_18_14"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    # Test 2: Ablate L18H11 alone
    print("\n[3/7] Ablate L18H11 alone...", flush=True)
    tester.install_ablation([(18, 11)])
    acc = tester.test(scenarios)
    results["ablate_18_11_alone"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    # Test 3: Ablate L17H4 alone
    print("\n[4/7] Ablate L17H4 alone...", flush=True)
    tester.install_ablation([(17, 4)])
    acc = tester.test(scenarios)
    results["ablate_17_4_alone"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    # Test 4: Ablate all three together
    print("\n[5/7] Ablate L17H4 + L18H11 + L18H14 (all three)...", flush=True)
    tester.install_ablation([(17, 4), (18, 11), (18, 14)])
    acc = tester.test(scenarios)
    results["ablate_all_three"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    # Test 5: AMPLIFY L18H11
    print("\n[6/7] AMPLIFY L18H11 by 2x...", flush=True)
    tester.install_amplification(18, 11, 2.0)
    acc = tester.test(scenarios)
    results["amplify_18_11_2x"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    # Test 6: AMPLIFY L18H11 by 3x
    print("\n[7/7] AMPLIFY L18H11 by 3x...", flush=True)
    tester.install_amplification(18, 11, 3.0)
    acc = tester.test(scenarios)
    results["amplify_18_11_3x"] = acc
    boost = acc - baseline
    print(f"  Accuracy: {acc:.1%} (boost: {boost:+.1%})")
    
    tester.clear_hooks()
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("\n  CONDITION                      | ACCURACY | BOOST")
    print("  " + "-" * 50)
    print(f"  Baseline                       | {results['baseline']:6.1%}   |   ---")
    print(f"  Ablate L18H11 alone            | {results['ablate_18_11_alone']:6.1%}   | {results['ablate_18_11_alone'] - baseline:+5.1%}")
    print(f"  Ablate L17H4 alone             | {results['ablate_17_4_alone']:6.1%}   | {results['ablate_17_4_alone'] - baseline:+5.1%}")
    print(f"  Ablate L17H4 + L18H14          | {results['ablate_17_4_and_18_14']:6.1%}   | {results['ablate_17_4_and_18_14'] - baseline:+5.1%}")
    print(f"  Ablate all three               | {results['ablate_all_three']:6.1%}   | {results['ablate_all_three'] - baseline:+5.1%}")
    print(f"  Amplify L18H11 by 2x           | {results['amplify_18_11_2x']:6.1%}   | {results['amplify_18_11_2x'] - baseline:+5.1%}")
    print(f"  Amplify L18H11 by 3x           | {results['amplify_18_11_3x']:6.1%}   | {results['amplify_18_11_3x'] - baseline:+5.1%}")
    
    # Hypothesis test
    print("\n" + "-" * 50)
    print("HYPOTHESIS TEST:")
    print("-" * 50)
    
    if results["ablate_18_11_alone"] < results["ablate_17_4_alone"]:
        print("\n  L18H11 ablation HURTS more than L17H4 ablation")
        print("  --> L18H11 is actually HELPING, not inhibiting!")
    elif results["ablate_18_11_alone"] > results["ablate_17_4_alone"]:
        print("\n  L18H11 ablation HELPS more than L17H4 ablation")
        print("  --> L18H11 is the primary inhibitor")
    else:
        print("\n  Similar effect from both ablations")
    
    if results["amplify_18_11_2x"] > baseline:
        print("\n  Amplifying L18H11 HELPS!")
        print("  --> L18H11 is definitely a positive contributor")
    elif results["amplify_18_11_2x"] < baseline:
        print("\n  Amplifying L18H11 HURTS")
        print("  --> L18H11 needs to be balanced, not amplified")
    
    if results["ablate_17_4_and_18_14"] > results["ablate_all_three"]:
        print("\n  Keeping L18H11 (not ablating) gives BETTER results!")
        print("  --> CONFIRMED: L18H11 should NOT be ablated")
    
    # Save
    with open(RESULTS_DIR / "l18h11_special_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'l18h11_special_test.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

