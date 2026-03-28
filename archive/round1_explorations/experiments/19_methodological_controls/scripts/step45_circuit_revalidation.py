"""
Step 45: Re-validate Inhibitory Circuit with Rigorous Methodology

Test if the L17H4, L15H12, L24H29 circuit findings hold with:
- Syntax-controlled prompts
- n=30 scenarios (minimum for statistical power)
- Both implicit and explicit ToM
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The heads we claimed were inhibitory
INHIBITORY_HEADS = [
    (17, 4),   # Main inhibitor: +45% claimed
    (15, 12),  # +30% claimed
    (24, 29),  # +30% claimed
]

def load_model():
    """Load model."""
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate_syntax_controlled_scenarios(n=30):
    """Generate scenarios with controlled syntax."""
    scenarios = []
    
    # Template: controlled syntax (pronoun + infinitive - the WORKING format)
    names_a = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry"]
    names_b = ["Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry", "Ivan"]
    objects = ["ball", "book", "key", "toy", "phone", "cup", "hat", "bag"]
    loc_a = ["drawer", "basket", "box", "shelf", "cabinet", "desk", "table", "chair"]
    loc_b = ["basket", "box", "shelf", "cabinet", "desk", "table", "chair", "drawer"]
    
    for i in range(n):
        idx = i % len(names_a)
        idx2 = (i + 1) % len(names_b)
        a, b = names_a[idx], names_b[idx2]
        obj = objects[i % len(objects)]
        la, lb = loc_a[i % len(loc_a)], loc_b[i % len(loc_b)]
        
        # Ensure locations are different
        if la == lb:
            lb = loc_b[(i + 1) % len(loc_b)]
        
        # IMPLICIT ToM (narrative inference required)
        implicit = {
            "prompt": f"{a} put the {obj} in the {la}. {a} left. {b} moved the {obj} to the {lb}. {a} returned. {a} looks for the {obj} in the",
            "correct": la,
            "incorrect": lb,
            "type": "implicit"
        }
        
        # EXPLICIT ToM (belief stated)
        explicit = {
            "prompt": f"{a} believes the {obj} is in the {la}. The {obj} is actually in the {lb}. {a} will look in the",
            "correct": la,
            "incorrect": lb,
            "type": "explicit"
        }
        
        scenarios.append(implicit)
        scenarios.append(explicit)
    
    return scenarios


def ablation_hook(head_list):
    """Create hook to ablate specific heads."""
    def hook(module, input, output):
        # output shape: (batch, seq, hidden)
        # Need to zero out specific head contributions
        hidden = output[0] if isinstance(output, tuple) else output
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook


def test_with_ablation(model, tokenizer, scenarios, heads_to_ablate=None):
    """Test scenarios with optional head ablation."""
    results = {"correct": 0, "total": 0, "details": []}
    
    # Set up ablation hooks if needed
    hooks = []
    if heads_to_ablate:
        for layer_idx, head_idx in heads_to_ablate:
            layer = model.model.layers[layer_idx]
            
            def make_hook(h_idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    batch, seq, dim = hidden.shape
                    n_heads = 32  # Qwen3-4B
                    head_dim = dim // n_heads
                    
                    # Reshape, zero head, reshape back
                    hidden = hidden.view(batch, seq, n_heads, head_dim)
                    hidden[:, :, h_idx, :] = 0
                    hidden = hidden.view(batch, seq, dim)
                    
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return hook
            
            h = layer.self_attn.o_proj.register_forward_hook(make_hook(head_idx))
            hooks.append(h)
    
    try:
        for scenario in scenarios:
            prompt = scenario["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
            
            # Get token IDs
            correct_ids = tokenizer.encode(" " + scenario["correct"], add_special_tokens=False)
            incorrect_ids = tokenizer.encode(" " + scenario["incorrect"], add_special_tokens=False)
            
            if correct_ids and incorrect_ids:
                correct_logit = logits[correct_ids[0]].item()
                incorrect_logit = logits[incorrect_ids[0]].item()
                
                is_correct = correct_logit > incorrect_logit
                results["correct"] += int(is_correct)
                results["total"] += 1
                
                results["details"].append({
                    "type": scenario["type"],
                    "correct_answer": scenario["correct"],
                    "is_correct": is_correct,
                    "logit_diff": correct_logit - incorrect_logit
                })
    finally:
        for h in hooks:
            h.remove()
    
    return results


def main():
    print("="*70)
    print("STEP 45: Circuit Re-validation with Rigorous Methodology")
    print("="*70)
    
    model, tokenizer = load_model()
    
    # Generate scenarios
    print("\nGenerating 30 syntax-controlled scenarios (60 total with explicit pairs)...")
    scenarios = generate_syntax_controlled_scenarios(n=30)
    print(f"  Total scenarios: {len(scenarios)}")
    
    # Split by type
    implicit = [s for s in scenarios if s["type"] == "implicit"]
    explicit = [s for s in scenarios if s["type"] == "explicit"]
    
    results = {}
    
    # Test 1: Baseline (no ablation)
    print("\n[1/4] Testing BASELINE (no ablation)...")
    baseline_implicit = test_with_ablation(model, tokenizer, implicit, heads_to_ablate=None)
    baseline_explicit = test_with_ablation(model, tokenizer, explicit, heads_to_ablate=None)
    
    results["baseline"] = {
        "implicit": {"accuracy": baseline_implicit["correct"] / baseline_implicit["total"] * 100,
                     "n": baseline_implicit["total"]},
        "explicit": {"accuracy": baseline_explicit["correct"] / baseline_explicit["total"] * 100,
                     "n": baseline_explicit["total"]}
    }
    print(f"  Implicit ToM: {results['baseline']['implicit']['accuracy']:.1f}%")
    print(f"  Explicit ToM: {results['baseline']['explicit']['accuracy']:.1f}%")
    
    # Test 2: Single head ablation (L17H4)
    print("\n[2/4] Testing L17H4 ablation (claimed +45%)...")
    l17h4_implicit = test_with_ablation(model, tokenizer, implicit, heads_to_ablate=[(17, 4)])
    l17h4_explicit = test_with_ablation(model, tokenizer, explicit, heads_to_ablate=[(17, 4)])
    
    results["L17H4_ablation"] = {
        "implicit": {"accuracy": l17h4_implicit["correct"] / l17h4_implicit["total"] * 100,
                     "n": l17h4_implicit["total"]},
        "explicit": {"accuracy": l17h4_explicit["correct"] / l17h4_explicit["total"] * 100,
                     "n": l17h4_explicit["total"]}
    }
    print(f"  Implicit ToM: {results['L17H4_ablation']['implicit']['accuracy']:.1f}%")
    print(f"  Explicit ToM: {results['L17H4_ablation']['explicit']['accuracy']:.1f}%")
    
    # Test 3: Combined 3-head ablation
    print("\n[3/4] Testing 3-head ablation (claimed 90%)...")
    three_head_implicit = test_with_ablation(model, tokenizer, implicit, heads_to_ablate=INHIBITORY_HEADS)
    three_head_explicit = test_with_ablation(model, tokenizer, explicit, heads_to_ablate=INHIBITORY_HEADS)
    
    results["three_head_ablation"] = {
        "implicit": {"accuracy": three_head_implicit["correct"] / three_head_implicit["total"] * 100,
                     "n": three_head_implicit["total"]},
        "explicit": {"accuracy": three_head_explicit["correct"] / three_head_explicit["total"] * 100,
                     "n": three_head_explicit["total"]}
    }
    print(f"  Implicit ToM: {results['three_head_ablation']['implicit']['accuracy']:.1f}%")
    print(f"  Explicit ToM: {results['three_head_ablation']['explicit']['accuracy']:.1f}%")
    
    # Test 4: Random heads (control)
    print("\n[4/4] Testing RANDOM heads ablation (control)...")
    random_heads = [(10, 5), (20, 10), (25, 15)]  # Random control heads
    random_implicit = test_with_ablation(model, tokenizer, implicit, heads_to_ablate=random_heads)
    random_explicit = test_with_ablation(model, tokenizer, explicit, heads_to_ablate=random_heads)
    
    results["random_ablation"] = {
        "implicit": {"accuracy": random_implicit["correct"] / random_implicit["total"] * 100,
                     "n": random_implicit["total"]},
        "explicit": {"accuracy": random_explicit["correct"] / random_explicit["total"] * 100,
                     "n": random_explicit["total"]}
    }
    print(f"  Implicit ToM: {results['random_ablation']['implicit']['accuracy']:.1f}%")
    print(f"  Explicit ToM: {results['random_ablation']['explicit']['accuracy']:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Circuit Re-validation Results")
    print("="*70)
    
    print("\n+----------------------+--------------+--------------+")
    print("| Condition            | Implicit ToM | Explicit ToM |")
    print("+----------------------+--------------+--------------+")
    for cond, data in results.items():
        print(f"| {cond:20} | {data['implicit']['accuracy']:10.1f}% | {data['explicit']['accuracy']:10.1f}% |")
    print("+----------------------+--------------+--------------+")
    
    # Calculate effects
    implicit_effect = results["three_head_ablation"]["implicit"]["accuracy"] - results["baseline"]["implicit"]["accuracy"]
    explicit_effect = results["three_head_ablation"]["explicit"]["accuracy"] - results["baseline"]["explicit"]["accuracy"]
    random_effect = results["random_ablation"]["implicit"]["accuracy"] - results["baseline"]["implicit"]["accuracy"]
    
    print(f"\n3-Head Ablation Effect:")
    print(f"  Implicit: {implicit_effect:+.1f}%")
    print(f"  Explicit: {explicit_effect:+.1f}%")
    print(f"  Random control: {random_effect:+.1f}%")
    
    # Verdict
    print("\n" + "="*70)
    if implicit_effect > 20 and implicit_effect > random_effect + 10:
        print("VERDICT: Circuit finding VALIDATED - substantial improvement over baseline and control")
    elif implicit_effect > 10:
        print("VERDICT: Circuit finding PARTIALLY VALIDATED - modest improvement")
    else:
        print("VERDICT: Circuit finding NOT VALIDATED - no significant improvement")
    print("="*70)
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(scenarios),
        "results": results,
        "effects": {
            "three_head_implicit": implicit_effect,
            "three_head_explicit": explicit_effect,
            "random_control": random_effect
        }
    }
    
    output_path = RESULTS_DIR / "step45_circuit_revalidation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()


