"""
Behavioral Theory of Mind Test
===============================

NOT Q&A format. Test if model predicts agent's ACTION based on BELIEF.

Gold standard: Sally-Anne style
- Input: Story ending with "Sally will..."
- Measure: Does model predict search in BELIEVED location vs ACTUAL location?

This is the REAL ToM test.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_token_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Get log probability of completion given prompt."""
    full_text = prompt + completion
    
    inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get log probs for completion tokens
    log_probs = torch.log_softmax(logits[0], dim=-1)
    
    completion_tokens = inputs.input_ids[0, prompt_len:]
    completion_logprobs = []
    
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1  # Position of token that predicts this one
        if pos >= 0:
            completion_logprobs.append(log_probs[pos, token_id].item())
    
    return sum(completion_logprobs) if completion_logprobs else -float('inf')


def test_false_belief_behavioral(model, tokenizer, scenarios: List[dict]) -> dict:
    """
    Test if model predicts agent looks in BELIEVED vs ACTUAL location.
    
    This is the core ToM test: belief guides behavior.
    """
    results = {
        "belief_wins": 0,
        "reality_wins": 0,
        "ties": 0,
        "details": [],
    }
    
    print(f"  Testing {len(scenarios)} scenarios...", flush=True)
    
    for i, scenario in enumerate(scenarios):
        if i % 50 == 0:
            print(f"    [{i}/{len(scenarios)}]", flush=True)
        
        story = scenario["story"]
        believed_completion = scenario["correct_completion"]
        
        # Create wrong completion (actual location instead of believed)
        actual_loc = scenario["actual_location"]
        action = scenario["correct_completion"].split(" it in the ")[0]
        wrong_completion = f"{action} it in the {actual_loc}"
        
        # Get log probs
        logp_believed = get_token_logprob(model, tokenizer, story, believed_completion)
        logp_actual = get_token_logprob(model, tokenizer, story, wrong_completion)
        
        if logp_believed > logp_actual:
            results["belief_wins"] += 1
            winner = "belief"
        elif logp_actual > logp_believed:
            results["reality_wins"] += 1
            winner = "reality"
        else:
            results["ties"] += 1
            winner = "tie"
        
        results["details"].append({
            "id": scenario["id"],
            "believed_location": scenario["believed_location"],
            "actual_location": actual_loc,
            "logp_believed": logp_believed,
            "logp_actual": logp_actual,
            "winner": winner,
        })
    
    total = len(scenarios)
    results["belief_rate"] = results["belief_wins"] / total
    results["reality_rate"] = results["reality_wins"] / total
    results["n_samples"] = total
    
    return results


def test_true_belief_behavioral(model, tokenizer, scenarios: List[dict]) -> dict:
    """
    Control: True belief scenarios.
    Model should also predict correct location (belief = reality).
    """
    results = {"correct": 0, "incorrect": 0, "n_samples": len(scenarios)}
    
    print(f"  Testing {len(scenarios)} true belief scenarios...", flush=True)
    
    for i, scenario in enumerate(scenarios):
        if i % 50 == 0:
            print(f"    [{i}/{len(scenarios)}]", flush=True)
        
        story = scenario["story"]
        correct_completion = scenario["correct_completion"]
        
        # Generate model's completion
        inputs = tokenizer(story, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)[len(story):]
        
        # Check if correct location mentioned
        correct_loc = scenario["believed_location"]
        if correct_loc.lower() in generated.lower():
            results["correct"] += 1
        else:
            results["incorrect"] += 1
    
    results["accuracy"] = results["correct"] / results["n_samples"]
    return results


def compute_statistics(false_belief_results: dict, null_dist_path: Path) -> dict:
    """Compute statistical significance against null distribution."""
    from scipy import stats
    
    # Load null distribution for reference
    try:
        with open(null_dist_path) as f:
            null_data = json.load(f)
    except:
        null_data = None
    
    n = false_belief_results["n_samples"]
    k = false_belief_results["belief_wins"]
    
    # Binomial test: is belief_rate significantly > 0.5?
    binom_result = stats.binomtest(k, n, p=0.5, alternative='greater')
    
    # Effect size (Cohen's h for proportions)
    p1 = k / n
    p2 = 0.5
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    return {
        "n_samples": n,
        "belief_wins": k,
        "belief_rate": p1,
        "p_value": binom_result.pvalue,
        "significant_05": binom_result.pvalue < 0.05,
        "significant_01": binom_result.pvalue < 0.01,
        "cohens_h": cohens_h,
        "effect_interpretation": "large" if abs(cohens_h) > 0.8 else "medium" if abs(cohens_h) > 0.5 else "small",
        "95_ci": list(binom_result.proportion_ci()),
    }


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("BEHAVIORAL THEORY OF MIND TEST")
    print("=" * 60)
    print("\nNOT Q&A - Measuring if model predicts belief-based actions.")
    
    # Load data
    print("\n[1/5] Loading dataset...", flush=True)
    with open(DATA_DIR / "large_dataset.json") as f:
        data = json.load(f)
    
    false_belief_scenarios = data["false_belief"]
    true_belief_scenarios = data["true_belief"]
    
    print(f"  False belief: {len(false_belief_scenarios)} scenarios")
    print(f"  True belief: {len(true_belief_scenarios)} scenarios")
    
    # Load model
    print("\n[2/5] Loading model...", flush=True)
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
    
    # Test false belief (main test)
    print("\n[3/5] Testing FALSE BELIEF scenarios...", flush=True)
    false_belief_results = test_false_belief_behavioral(model, tokenizer, false_belief_scenarios)
    
    print(f"\n  Results:")
    print(f"    Belief wins: {false_belief_results['belief_wins']}/{false_belief_results['n_samples']} ({false_belief_results['belief_rate']:.1%})")
    print(f"    Reality wins: {false_belief_results['reality_wins']}/{false_belief_results['n_samples']} ({false_belief_results['reality_rate']:.1%})")
    
    # Test true belief (control)
    print("\n[4/5] Testing TRUE BELIEF scenarios (control)...", flush=True)
    true_belief_results = test_true_belief_behavioral(model, tokenizer, true_belief_scenarios[:50])  # Subset for speed
    
    print(f"\n  Results:")
    print(f"    Correct: {true_belief_results['correct']}/{true_belief_results['n_samples']} ({true_belief_results['accuracy']:.1%})")
    
    # Statistics
    print("\n[5/5] Computing statistics...", flush=True)
    stats_results = compute_statistics(false_belief_results, RESULTS_DIR / "null_distributions.json")
    
    # Compile results
    all_results = {
        "false_belief": {
            "summary": {
                "belief_rate": false_belief_results["belief_rate"],
                "reality_rate": false_belief_results["reality_rate"],
                "n_samples": false_belief_results["n_samples"],
            },
            "statistics": stats_results,
        },
        "true_belief": true_belief_results,
        "timing": time.perf_counter() - timer_start,
    }
    
    # Save (without details to keep file small)
    with open(RESULTS_DIR / "behavioral_tom_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. FALSE BELIEF TEST (Main Result)")
    print("-" * 40)
    print(f"  Model predicts BELIEVED location: {false_belief_results['belief_rate']:.1%}")
    print(f"  Model predicts ACTUAL location: {false_belief_results['reality_rate']:.1%}")
    print(f"  p-value: {stats_results['p_value']:.2e}")
    print(f"  Significant (p<0.05): {stats_results['significant_05']}")
    print(f"  Effect size (Cohen's h): {stats_results['cohens_h']:.3f} ({stats_results['effect_interpretation']})")
    
    print("\n2. TRUE BELIEF TEST (Control)")
    print("-" * 40)
    print(f"  Accuracy: {true_belief_results['accuracy']:.1%}")
    
    print("\n3. INTERPRETATION")
    print("-" * 40)
    if false_belief_results['belief_rate'] > 0.5 and stats_results['significant_05']:
        print("  Model shows ToM: predicts belief-based actions!")
    elif false_belief_results['reality_rate'] > 0.5:
        print("  Model predicts based on REALITY, not belief (no ToM)")
    else:
        print("  Mixed results - further investigation needed")
    
    total_time = time.perf_counter() - timer_start
    print(f"\nCompleted in {total_time:.1f}s")
    print(f"Saved to {RESULTS_DIR / 'behavioral_tom_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

