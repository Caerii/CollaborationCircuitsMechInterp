"""
Multi-Agent Behavioral ToM Test
================================

Test if model can track multiple agents' beliefs and predict
their actions correctly.

Key tests:
1. Second-order belief: What does A think B will do?
2. Belief divergence: A vs B have different beliefs - predict each correctly
3. Dialogue tracking: Track belief updates across turns
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
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
    
    log_probs = torch.log_softmax(logits[0], dim=-1)
    
    completion_tokens = inputs.input_ids[0, prompt_len:]
    completion_logprobs = []
    
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1
        if pos >= 0:
            completion_logprobs.append(log_probs[pos, token_id].item())
    
    return sum(completion_logprobs) if completion_logprobs else -float('inf')


def test_second_order_beliefs(model, tokenizer, scenarios: List[dict]) -> dict:
    """
    Test second-order belief understanding.
    
    Question: What does A think B will do?
    Correct: A knows B believes X, so A predicts B searches X
    Wrong: A uses own belief Y to predict B's action
    """
    results = {
        "correct": 0,
        "incorrect": 0,
        "details": [],
    }
    
    print(f"  Testing {len(scenarios)} second-order scenarios...", flush=True)
    
    for i, s in enumerate(scenarios):
        if i % 25 == 0:
            print(f"    [{i}/{len(scenarios)}]", flush=True)
        
        logp_correct = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_wrong = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_correct > logp_wrong
        
        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
        
        results["details"].append({
            "id": s["id"],
            "logp_correct": logp_correct,
            "logp_wrong": logp_wrong,
            "correct": is_correct,
            "a_belief": s["a_belief"],
            "b_belief": s["b_belief"],
            "a_model_of_b": s["a_model_of_b"],
        })
    
    n = len(scenarios)
    results["accuracy"] = results["correct"] / n
    results["n_samples"] = n
    
    return results


def test_belief_divergence(model, tokenizer, scenarios: List[dict]) -> dict:
    """
    Test if model can track divergent beliefs.
    
    Both A and B have different beliefs - model should predict correct action for each.
    """
    results = {
        "correct": 0,
        "incorrect": 0,
        "by_agent": {},
        "details": [],
    }
    
    print(f"  Testing {len(scenarios)} divergent belief scenarios...", flush=True)
    
    for i, s in enumerate(scenarios):
        if i % 50 == 0:
            print(f"    [{i}/{len(scenarios)}]", flush=True)
        
        logp_correct = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_wrong = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_correct > logp_wrong
        
        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
        
        # Track by agent position (first vs second mentioned)
        agent = s["target_agent"]
        if agent not in results["by_agent"]:
            results["by_agent"][agent] = {"correct": 0, "total": 0}
        results["by_agent"][agent]["total"] += 1
        if is_correct:
            results["by_agent"][agent]["correct"] += 1
        
        results["details"].append({
            "id": s["id"],
            "target_agent": agent,
            "correct": is_correct,
        })
    
    n = len(scenarios)
    results["accuracy"] = results["correct"] / n
    results["n_samples"] = n
    
    return results


def test_dialogue_tracking(model, tokenizer, scenarios: List[dict]) -> dict:
    """
    Test belief tracking across dialogue turns.
    
    Key: One agent's belief updates, another's doesn't (due to partial information).
    """
    results = {
        "correct": 0,
        "incorrect": 0,
        "updated_agent_correct": 0,
        "unchanged_agent_correct": 0,
        "updated_agent_total": 0,
        "unchanged_agent_total": 0,
        "details": [],
    }
    
    print(f"  Testing {len(scenarios)} dialogue scenarios...", flush=True)
    
    for i, s in enumerate(scenarios):
        if i % 25 == 0:
            print(f"    [{i}/{len(scenarios)}]", flush=True)
        
        # Determine if this is the "updated" or "unchanged" agent
        a_updated = s["a_belief"] != s["b_belief"]  # A got update, B didn't
        is_updated_agent = (s["target_agent"] == s["story"].split()[3])  # First agent in dialogue
        
        # Create wrong completion (the OTHER agent's belief)
        wrong_belief = s["b_belief"] if s["target_belief"] == s["a_belief"] else s["a_belief"]
        wrong_completion = f" {wrong_belief}"
        
        logp_correct = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_wrong = get_token_logprob(model, tokenizer, s["story"], wrong_completion)
        
        is_correct = logp_correct > logp_wrong
        
        if is_correct:
            results["correct"] += 1
            if is_updated_agent:
                results["updated_agent_correct"] += 1
            else:
                results["unchanged_agent_correct"] += 1
        else:
            results["incorrect"] += 1
        
        if is_updated_agent:
            results["updated_agent_total"] += 1
        else:
            results["unchanged_agent_total"] += 1
        
        results["details"].append({
            "id": s["id"],
            "target_agent": s["target_agent"],
            "correct": is_correct,
            "is_updated_agent": is_updated_agent,
        })
    
    n = len(scenarios)
    results["accuracy"] = results["correct"] / n
    results["n_samples"] = n
    
    if results["updated_agent_total"] > 0:
        results["updated_agent_accuracy"] = (
            results["updated_agent_correct"] / results["updated_agent_total"]
        )
    if results["unchanged_agent_total"] > 0:
        results["unchanged_agent_accuracy"] = (
            results["unchanged_agent_correct"] / results["unchanged_agent_total"]
        )
    
    return results


def compute_statistics(results: dict, test_name: str) -> dict:
    """Compute statistical significance."""
    from scipy import stats
    
    n = results["n_samples"]
    k = results["correct"]
    
    # Binomial test against chance (50%)
    binom_result = stats.binomtest(k, n, p=0.5, alternative='greater')
    
    # Effect size
    p1 = k / n
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(0.5)))
    
    # Convert all values to native Python types for JSON serialization
    return {
        "test": test_name,
        "accuracy": float(p1),
        "n_samples": int(n),
        "correct": int(k),
        "p_value": float(binom_result.pvalue),
        "significant": bool(binom_result.pvalue < 0.05),
        "cohens_h": float(cohens_h),
        "95_ci": [float(x) for x in binom_result.proportion_ci()],
    }


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("MULTI-AGENT BEHAVIORAL TOM TEST")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading scenarios...", flush=True)
    with open(DATA_DIR / "multi_agent_scenarios.json") as f:
        data = json.load(f)
    
    print(f"  Second-order: {len(data['second_order'])}")
    print(f"  Divergent: {len(data['divergent'])}")
    print(f"  Dialogue: {len(data['dialogue'])}")
    print(f"  Comparison: {len(data['comparison'])}")
    
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
    
    all_results = {}
    
    # Test 1: Second-order beliefs
    print("\n[3/5] Testing SECOND-ORDER beliefs...", flush=True)
    second_order_results = test_second_order_beliefs(model, tokenizer, data["second_order"])
    second_order_stats = compute_statistics(second_order_results, "second_order")
    all_results["second_order"] = {
        "results": {k: v for k, v in second_order_results.items() if k != "details"},
        "statistics": second_order_stats,
    }
    print(f"  Accuracy: {second_order_results['accuracy']:.1%}")
    print(f"  p-value: {second_order_stats['p_value']:.2e}")
    
    # Test 2: Belief divergence
    print("\n[4/5] Testing BELIEF DIVERGENCE...", flush=True)
    divergent_results = test_belief_divergence(model, tokenizer, data["divergent"])
    divergent_stats = compute_statistics(divergent_results, "divergent")
    all_results["divergent"] = {
        "results": {k: v for k, v in divergent_results.items() if k != "details"},
        "statistics": divergent_stats,
    }
    print(f"  Accuracy: {divergent_results['accuracy']:.1%}")
    print(f"  p-value: {divergent_stats['p_value']:.2e}")
    
    # Test 3: Dialogue tracking
    print("\n[5/5] Testing DIALOGUE TRACKING...", flush=True)
    dialogue_results = test_dialogue_tracking(model, tokenizer, data["dialogue"])
    dialogue_stats = compute_statistics(dialogue_results, "dialogue")
    all_results["dialogue"] = {
        "results": {k: v for k, v in dialogue_results.items() if k != "details"},
        "statistics": dialogue_stats,
    }
    print(f"  Overall accuracy: {dialogue_results['accuracy']:.1%}")
    if "updated_agent_accuracy" in dialogue_results:
        print(f"  Updated agent: {dialogue_results['updated_agent_accuracy']:.1%}")
    if "unchanged_agent_accuracy" in dialogue_results:
        print(f"  Unchanged agent: {dialogue_results['unchanged_agent_accuracy']:.1%}")
    print(f"  p-value: {dialogue_stats['p_value']:.2e}")
    
    # Save
    with open(RESULTS_DIR / "behavioral_multi_agent.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Test':<25} {'Accuracy':<12} {'p-value':<15} {'Significant':<12}")
    print("-" * 64)
    
    for test_name in ["second_order", "divergent", "dialogue"]:
        stats = all_results[test_name]["statistics"]
        sig = "YES" if stats["significant"] else "no"
        print(f"{test_name:<25} {stats['accuracy']:.1%}        {stats['p_value']:.2e}       {sig}")
    
    print("\n" + "-" * 64)
    
    # Overall assessment
    n_significant = sum(
        1 for t in all_results.values() 
        if t["statistics"]["significant"]
    )
    
    if n_significant == 3:
        print("\n[+++] STRONG MULTI-AGENT TOM EVIDENCE")
        print("      Model tracks multiple agents' beliefs correctly!")
    elif n_significant >= 2:
        print("\n[++] PARTIAL MULTI-AGENT TOM EVIDENCE")
        print("     Some tests pass, further investigation needed")
    elif n_significant == 1:
        print("\n[+] WEAK EVIDENCE")
        print("    Only one test significant")
    else:
        print("\n[-] NO MULTI-AGENT TOM EVIDENCE")
        print("    Model may not track per-agent beliefs")
    
    total_time = time.perf_counter() - timer_start
    print(f"\nCompleted in {total_time:.1f}s")
    print(f"Saved to {RESULTS_DIR / 'behavioral_multi_agent.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


