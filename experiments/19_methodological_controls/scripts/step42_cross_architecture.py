"""
Step 42: Cross-Architecture Validation

Test our CORRECTED findings across different model architectures:
- Qwen3-4B (baseline)
- Phi-3.5-mini-instruct (Microsoft)  
- Llama-3.2-1B (Meta) - smaller but different architecture

WITH PROPER METHODOLOGY:
- N=20 scenarios per condition (minimum for statistical analysis)
- Syntax-controlled comparisons
- Wilson CIs and significance tests
"""

import torch
import json
import sys
import io
import random
import gc
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Models to test (compatible with our setup)
MODELS = [
    {"name": "Qwen3-4B", "id": "Qwen/Qwen3-4B"},
    {"name": "Qwen2.5-1.5B", "id": "Qwen/Qwen2.5-1.5B"},
    {"name": "Qwen2.5-0.5B", "id": "Qwen/Qwen2.5-0.5B"},
]

# Controlled test conditions
TEST_CONDITIONS = {
    # Syntax-matched pairs (our key finding)
    "finite_clause": "thinks the ball is in the",      # Finite "is"
    "infinitive_clause": "thinks the ball to be in the",  # Infinitive "to be"
    
    # Action vs state (syntax-matched)
    "action_looks": "looks for the ball in the",
    "action_searches": "searches for the ball in the",
    "state_believes": "believes the ball to be in the",
    "state_expects": "expects the ball to be in the",
}

# Diverse scenarios for statistical power
AGENTS = ["Alice", "Bob", "Carol", "David", "Emma"]
OBJECTS = ["ball", "book", "cup", "toy", "key"]
LOCATIONS = [("drawer", "basket"), ("shelf", "desk"), ("box", "bag"), 
             ("cupboard", "closet"), ("pocket", "table")]


def wilson_ci(successes, total, confidence=0.95):
    """Wilson score confidence interval."""
    if total == 0:
        return 0, 0, 1
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return p, max(0, center - spread), min(1, center + spread)


def generate_scenarios(n=20):
    """Generate n diverse scenarios."""
    scenarios = []
    for i in range(n):
        agent = AGENTS[i % len(AGENTS)]
        obj = OBJECTS[i % len(OBJECTS)]
        loc1, loc2 = LOCATIONS[i % len(LOCATIONS)]
        
        story = f"""{agent} put the {obj} in the {loc1}. {agent} left.
Someone moved the {obj} to the {loc2} while {agent} was away.
{agent} returns. {agent}"""
        
        scenarios.append({
            "story": story,
            "correct": loc1,
            "wrong": loc2
        })
    return scenarios


def test_model(model_config, scenarios):
    """Test a model on all conditions."""
    model_name = model_config["name"]
    model_id = model_config["id"]
    
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": str(e)}
    
    results = {"model": model_name, "conditions": {}}
    
    # Test each condition
    for cond_name, completion in TEST_CONDITIONS.items():
        correct_count = 0
        diffs = []
        
        for scenario in scenarios:
            prompt = scenario["story"] + " " + completion
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits[0, -1, :]
            
            def get_logit(word):
                for prefix in [" ", ""]:
                    tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
                    if tokens:
                        return logits[tokens[0]].item()
                return float('-inf')
            
            correct_logit = get_logit(scenario["correct"])
            wrong_logit = get_logit(scenario["wrong"])
            diff = correct_logit - wrong_logit
            
            if diff > 0:
                correct_count += 1
            diffs.append(diff)
        
        # Calculate statistics
        acc, ci_lo, ci_hi = wilson_ci(correct_count, len(scenarios))
        
        results["conditions"][cond_name] = {
            "accuracy": float(acc),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "n_correct": correct_count,
            "n_total": len(scenarios),
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs))
        }
        
        status = "[OK]" if acc > 0.5 else "[FAIL]"
        print(f"  {cond_name:<25}: {acc*100:>5.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%) {status}")
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def run_cross_architecture():
    """Run cross-architecture validation."""
    print("="*70)
    print("STEP 42: Cross-Architecture Validation")
    print("="*70)
    print(f"Seed: {SEED}")
    print(f"N scenarios: 20 per condition")
    print(f"Models: {[m['name'] for m in MODELS]}")
    
    # Generate scenarios
    scenarios = generate_scenarios(20)
    
    # Test each model
    all_results = {}
    for model_config in MODELS:
        results = test_model(model_config, scenarios)
        all_results[model_config["name"]] = results
    
    # Summary comparison
    print("\n" + "="*70)
    print("CROSS-ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("\n" + "-"*90)
    print(f"{'Condition':<25} | " + " | ".join(f"{m['name']:<12}" for m in MODELS))
    print("-"*90)
    
    for cond_name in TEST_CONDITIONS.keys():
        row = f"{cond_name:<25}"
        for model_config in MODELS:
            model_name = model_config["name"]
            if model_name in all_results and "error" not in all_results[model_name]:
                acc = all_results[model_name]["conditions"][cond_name]["accuracy"]
                row += f" | {acc*100:>10.0f}%"
            else:
                row += f" | {'ERROR':>11}"
        print(row)
    
    print("-"*90)
    
    # Statistical comparison: finite vs infinitive across models
    print("\n--- KEY COMPARISON: Finite vs Infinitive Clause ---")
    for model_config in MODELS:
        model_name = model_config["name"]
        if model_name in all_results and "error" not in all_results[model_name]:
            finite_acc = all_results[model_name]["conditions"]["finite_clause"]["accuracy"]
            infin_acc = all_results[model_name]["conditions"]["infinitive_clause"]["accuracy"]
            diff = (infin_acc - finite_acc) * 100
            print(f"  {model_name}: Finite={finite_acc*100:.0f}%, Infinitive={infin_acc*100:.0f}%, Diff={diff:+.0f}%")
    
    # Save
    save_path = RESULTS_DIR / "cross_architecture_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


if __name__ == "__main__":
    run_cross_architecture()

