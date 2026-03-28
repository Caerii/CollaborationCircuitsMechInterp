"""
INHIBITORY HEAD DISCOVERY: L24H29 Suppresses Belief Update!
============================================================

MAJOR FINDING: Ablating L24H29 improves BOTH:
- Bridged prompts: 94% → 100%
- BASELINE prompts: 26% → 54%

This head is ACTIVELY SUPPRESSING the belief update inference!

Questions:
1. Is this reproducible across more samples?
2. What happens to baseline (no bridge) with L24H29 ablated?
3. Can we achieve good ToM without ANY bridge phrase by just ablating L24H29?
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


class InhibitoryHeadTester:
    """Test the inhibitory head hypothesis."""
    
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
            modified = reshaped.view(batch, seq_len, hidden)
            return (modified,) + args[1:] if len(args) > 1 else (modified,)
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
    
    def test_prompt(self, prompt: str, correct: str, wrong: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        correct_logit = logits[correct_id].item()
        wrong_logit = logits[wrong_id].item()
        
        # Compute probability
        max_logit = max(correct_logit, wrong_logit)
        correct_prob = np.exp(correct_logit - max_logit) / (np.exp(correct_logit - max_logit) + np.exp(wrong_logit - max_logit))
        
        return {
            "chose_correct": correct_logit > wrong_logit,
            "margin": correct_logit - wrong_logit,
            "prob": float(correct_prob),
        }


def generate_test_scenarios(n: int = 100) -> list:
    """Generate more scenarios for robust testing."""
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    INFORMERS = ["Eve", "Grace", "Henry", "Iris", "Jack", "Kate"]
    OBJECTS = ["ball", "key", "book", "toy", "cup", "pen"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "box", "bag"]
    
    scenarios = []
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice([x for x in INFORMERS if x != agent])
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # Pure baseline - no bridge at all
        baseline = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # Minimal bridge
        bridged = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"so {agent} updated their belief. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        scenarios.append({
            "baseline": baseline,
            "bridged": bridged,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("INHIBITORY HEAD DISCOVERY: Can We Activate ToM by Ablation?")
    print("=" * 70)
    print("""
    HYPOTHESIS: L24H29 is SUPPRESSING belief update inference.
    
    Test: With L24H29 ablated, can we get good accuracy on 
    BASELINE prompts (no bridge phrase needed)?
    
    If YES: We've found a "ToM unlock" intervention!
    If NO: The previous finding was noise.
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
    
    tester = InhibitoryHeadTester(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/4] Generating test scenarios...", flush=True)
    scenarios = generate_test_scenarios(100)  # Larger N for robustness
    print(f"  Generated {len(scenarios)} scenarios")
    
    # Test conditions
    conditions = [
        ("no_ablation", None),
        ("ablate_L24H29", (24, 29)),
        # Also test nearby heads to check specificity
        ("ablate_L24H28", (24, 28)),
        ("ablate_L24H30", (24, 30)),
        ("ablate_L25H29", (25, 29)),
        ("ablate_L23H29", (23, 29)),
    ]
    
    results = {}
    
    print("\n[3/4] Testing conditions...", flush=True)
    
    for cond_name, ablation_target in conditions:
        if ablation_target:
            tester.install_ablation(*ablation_target)
        else:
            tester.clear_hooks()
        
        baseline_results = []
        bridged_results = []
        
        for i, scenario in enumerate(scenarios):
            if i % 25 == 0:
                print(f"  {cond_name}: [{i}/{len(scenarios)}]", flush=True)
            
            base_result = tester.test_prompt(scenario["baseline"], scenario["correct"], scenario["wrong"])
            bridge_result = tester.test_prompt(scenario["bridged"], scenario["correct"], scenario["wrong"])
            
            baseline_results.append(base_result)
            bridged_results.append(bridge_result)
        
        tester.clear_hooks()
        
        baseline_acc = sum(1 for r in baseline_results if r["chose_correct"]) / len(baseline_results)
        bridged_acc = sum(1 for r in bridged_results if r["chose_correct"]) / len(bridged_results)
        baseline_prob = np.mean([r["prob"] for r in baseline_results])
        bridged_prob = np.mean([r["prob"] for r in bridged_results])
        
        results[cond_name] = {
            "baseline_acc": baseline_acc,
            "bridged_acc": bridged_acc,
            "baseline_prob": float(baseline_prob),
            "bridged_prob": float(bridged_prob),
        }
        
        print(f"  {cond_name}: baseline={baseline_acc:.1%}, bridged={bridged_acc:.1%}")
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Inhibitory Head Test (N=100)")
    print("=" * 70)
    
    ref_baseline = results["no_ablation"]["baseline_acc"]
    ref_bridged = results["no_ablation"]["bridged_acc"]
    
    print(f"\n  {'CONDITION':<20} {'BASELINE':<12} {'BRIDGED':<12} {'BASE CHG':<10} {'BRIDGE CHG':<10}")
    print("  " + "-" * 70)
    
    for cond_name, data in results.items():
        base_delta = data["baseline_acc"] - ref_baseline
        bridge_delta = data["bridged_acc"] - ref_bridged
        
        marker = ""
        if cond_name == "ablate_L24H29" and base_delta > 0.15:
            marker = " *** CONFIRMED ***"
        
        print(f"  {cond_name:<20} {data['baseline_acc']:>6.1%}       {data['bridged_acc']:>6.1%}       {base_delta:>+6.1%}     {bridge_delta:>+6.1%}{marker}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Is L24H29 the Inhibitory Head?")
    print("=" * 70)
    
    L24H29_baseline_boost = results["ablate_L24H29"]["baseline_acc"] - ref_baseline
    L24H29_bridged_boost = results["ablate_L24H29"]["bridged_acc"] - ref_bridged
    
    # Check if other nearby heads have similar effect
    other_boosts = []
    for cond in ["ablate_L24H28", "ablate_L24H30", "ablate_L25H29", "ablate_L23H29"]:
        boost = results[cond]["baseline_acc"] - ref_baseline
        other_boosts.append((cond, boost))
    
    max_other_boost = max(b for _, b in other_boosts)
    
    print(f"\n  Reference (no ablation):")
    print(f"    Baseline: {ref_baseline:.1%}")
    print(f"    Bridged:  {ref_bridged:.1%}")
    
    print(f"\n  L24H29 ablation effect:")
    print(f"    Baseline boost: {L24H29_baseline_boost:+.1%}")
    print(f"    Bridged boost:  {L24H29_bridged_boost:+.1%}")
    
    print(f"\n  Nearby heads (specificity check):")
    for cond, boost in other_boosts:
        print(f"    {cond}: {boost:+.1%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if L24H29_baseline_boost > 0.20 and L24H29_baseline_boost > max_other_boost + 0.10:
        print(f"""
    [INHIBITORY HEAD CONFIRMED!]
    
    L24H29 is SPECIFICALLY suppressing belief update inference!
    
    - Ablating L24H29 boosts baseline by {L24H29_baseline_boost:+.1%}
    - Best nearby head only provides {max_other_boost:+.1%}
    - This is a SPECIFIC, TARGETED effect
    
    IMPLICATION: We can "unlock" belief update inference by ablating L24H29!
    
    This is a mechanistic intervention that doesn't require prompt engineering.
    The model HAS the capability, but this head is suppressing it.
        """)
    elif L24H29_baseline_boost > 0.10:
        print(f"""
    [PARTIAL EFFECT]
    
    L24H29 ablation helps baseline by {L24H29_baseline_boost:+.1%}.
    But effect is moderate or not highly specific.
    
    May be part of a larger inhibitory circuit, not a single "off switch".
        """)
    else:
        print(f"""
    [NOT CONFIRMED]
    
    L24H29 ablation effect: {L24H29_baseline_boost:+.1%}
    
    The earlier finding may have been noise or sample-size dependent.
    The inhibitory head hypothesis is not supported at N=100.
        """)
    
    # Save results
    with open(RESULTS_DIR / "inhibitory_head_discovery_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

