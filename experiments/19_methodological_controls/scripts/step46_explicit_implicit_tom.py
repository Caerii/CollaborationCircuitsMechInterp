"""
Step 46: Explicit vs Implicit ToM - Direct Comparison

Test the hypothesis that models are good at EXPLICIT belief processing
but weak at IMPLICIT belief computation.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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


def generate_matched_pairs(n=25):
    """Generate matched pairs of explicit vs implicit ToM scenarios."""
    pairs = []
    
    names = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry",
             "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar"]
    objects = ["ball", "book", "key", "toy", "phone", "cup", "hat", "bag",
               "pen", "coin", "ring", "watch", "card", "note", "box"]
    locations_a = ["drawer", "basket", "box", "shelf", "cabinet", "desk", "table", 
                   "closet", "bag", "pocket", "case", "folder", "jar", "bin", "rack"]
    locations_b = ["basket", "box", "shelf", "cabinet", "desk", "table", "closet",
                   "bag", "pocket", "case", "folder", "jar", "bin", "rack", "drawer"]
    
    for i in range(n):
        name = names[i % len(names)]
        obj = objects[i % len(objects)]
        loc_a = locations_a[i % len(locations_a)]
        loc_b = locations_b[(i + 3) % len(locations_b)]
        
        if loc_a == loc_b:
            loc_b = locations_b[(i + 5) % len(locations_b)]
        
        # IMPLICIT: Must infer belief from narrative
        implicit = f"{name} put the {obj} in the {loc_a}. {name} left the room. Someone moved the {obj} to the {loc_b}. {name} returned. {name} will look in the"
        
        # EXPLICIT: Belief stated directly  
        explicit = f"{name} believes the {obj} is in the {loc_a}. The {obj} is actually in the {loc_b}. {name} will look in the"
        
        # SEMI-EXPLICIT: Partial information
        semi = f"{name} last saw the {obj} in the {loc_a}. {name} doesn't know it was moved to the {loc_b}. {name} will look in the"
        
        # STRUCTURED: Role-based format
        structured = f"[{name.upper()}'S BELIEF]: The {obj} is in the {loc_a}\n[REALITY]: The {obj} is in the {loc_b}\n[QUESTION]: Where will {name} look? Answer: the"
        
        pairs.append({
            "implicit": implicit,
            "explicit": explicit,
            "semi_explicit": semi,
            "structured": structured,
            "correct": loc_a,
            "incorrect": loc_b,
            "agent": name,
            "object": obj
        })
    
    return pairs


def test_scenario(model, tokenizer, prompt, correct, incorrect):
    """Test a single scenario and return logit difference."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_ids = tokenizer.encode(" " + correct, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(" " + incorrect, add_special_tokens=False)
    
    if correct_ids and incorrect_ids:
        correct_logit = logits[correct_ids[0]].item()
        incorrect_logit = logits[incorrect_ids[0]].item()
        return correct_logit - incorrect_logit, correct_logit > incorrect_logit
    return 0.0, False


def main():
    print("="*70)
    print("STEP 46: Explicit vs Implicit ToM Comparison")
    print("="*70)
    
    model, tokenizer = load_model()
    
    print("\nGenerating 25 matched pairs...")
    pairs = generate_matched_pairs(n=25)
    
    results = {
        "implicit": {"correct": 0, "logit_diffs": []},
        "explicit": {"correct": 0, "logit_diffs": []},
        "semi_explicit": {"correct": 0, "logit_diffs": []},
        "structured": {"correct": 0, "logit_diffs": []}
    }
    
    print("\nTesting all conditions...")
    for i, pair in enumerate(pairs):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(pairs)}")
        
        for cond in ["implicit", "explicit", "semi_explicit", "structured"]:
            diff, is_correct = test_scenario(
                model, tokenizer, pair[cond], pair["correct"], pair["incorrect"]
            )
            results[cond]["logit_diffs"].append(diff)
            results[cond]["correct"] += int(is_correct)
    
    # Compute statistics
    n = len(pairs)
    print("\n" + "="*70)
    print("RESULTS: Explicit vs Implicit ToM")
    print("="*70)
    
    print("\n+----------------+----------+------------+----------+")
    print("| Condition      | Accuracy | Mean LogDf | Std Dev  |")
    print("+----------------+----------+------------+----------+")
    
    for cond in ["implicit", "explicit", "semi_explicit", "structured"]:
        acc = results[cond]["correct"] / n * 100
        mean_diff = np.mean(results[cond]["logit_diffs"])
        std_diff = np.std(results[cond]["logit_diffs"])
        results[cond]["accuracy"] = acc
        results[cond]["mean_logit_diff"] = mean_diff
        results[cond]["std_logit_diff"] = std_diff
        print(f"| {cond:14} | {acc:6.1f}% | {mean_diff:+10.2f} | {std_diff:8.2f} |")
    
    print("+----------------+----------+------------+----------+")
    
    # Key comparisons
    print("\n" + "-"*70)
    print("KEY COMPARISONS:")
    print("-"*70)
    
    explicit_acc = results["explicit"]["accuracy"]
    implicit_acc = results["implicit"]["accuracy"]
    diff = explicit_acc - implicit_acc
    
    print(f"\nExplicit - Implicit = {diff:+.1f}%")
    
    if diff > 20:
        print("\n[CONFIRMED] Explicit ToM >> Implicit ToM")
        print("The model is MUCH better when beliefs are stated explicitly!")
    elif diff > 5:
        print("\n[PARTIAL] Explicit ToM > Implicit ToM")
        print("Some advantage for explicit framing.")
    else:
        print("\n[NOT CONFIRMED] No significant difference")
        print("Model handles both similarly.")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_pairs": n,
        "results": {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else 
                           [float(x) for x in vv] if isinstance(vv, list) else vv) 
                       for kk, vv in v.items()} 
                   for k, v in results.items()},
        "explicit_implicit_diff": diff
    }
    
    output_path = RESULTS_DIR / "step46_explicit_implicit_tom.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


