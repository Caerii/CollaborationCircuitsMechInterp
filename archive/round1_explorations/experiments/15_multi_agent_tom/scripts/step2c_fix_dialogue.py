"""
Fix Dialogue Tracking Test
===========================

Problems in original:
1. is_updated_agent logic was broken (s["story"].split()[3] doesn't get agent)
2. updated_agent_total was 0 (never identified any updated agents)

Fix: Properly identify which agent got the update vs which didn't.
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from scipy import stats
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

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
    
    total = 0
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1
        if pos >= 0:
            total += log_probs[pos, token_id].item()
    
    return total


def generate_fixed_dialogue_scenarios(n: int = 50) -> list:
    """
    Generate dialogue scenarios with CLEAR tracking of who knows what.
    
    Structure:
    - Agent A tells Agent B object is in location X
    - Agent B leaves
    - Agent C tells Agent A (only) that object moved to Y
    - Agent B returns
    
    Test: 
    - Ask where A will look -> should be Y (updated)
    - Ask where B will look -> should be X (outdated)
    """
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 3)
        a, b, c = agents
        obj = random.choice(OBJECTS)
        original_loc, new_loc = random.sample(LOCATIONS, 2)
        
        # Base story
        story_base = (
            f"{a} tells {b} that the {obj} is in the {original_loc}. "
            f"{b} leaves the room. "
            f"While {b} is away, {c} tells {a} that the {obj} has been moved to the {new_loc}. "
            f"{b} returns, unaware of the change. "
        )
        
        # Test A (updated agent) - should search new location
        scenarios.append({
            "id": f"dialogue_{i}_updated",
            "story": story_base + f"Now {a} needs the {obj}. {a} will search in the",
            "target_agent": a,
            "is_updated_agent": True,
            "correct_location": new_loc,
            "wrong_location": original_loc,
            "correct_completion": f" {new_loc}",
            "wrong_completion": f" {original_loc}",
        })
        
        # Test B (unchanged agent) - should search original location
        scenarios.append({
            "id": f"dialogue_{i}_unchanged",
            "story": story_base + f"Now {b} needs the {obj}. {b} will search in the",
            "target_agent": b,
            "is_updated_agent": False,
            "correct_location": original_loc,
            "wrong_location": new_loc,
            "correct_completion": f" {original_loc}",
            "wrong_completion": f" {new_loc}",
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("FIXED DIALOGUE TRACKING TEST")
    print("=" * 60)
    print("\nProperly tracking updated vs unchanged agents...")
    
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
    print("  [OK]", flush=True)
    
    # Generate fixed scenarios
    print("\n[2/4] Generating fixed scenarios...", flush=True)
    scenarios = generate_fixed_dialogue_scenarios(50)
    print(f"  Generated {len(scenarios)} scenarios (50 updated + 50 unchanged)")
    
    # Run test
    print("\n[3/4] Testing...", flush=True)
    results = {
        "overall": {"correct": 0, "incorrect": 0},
        "updated_agent": {"correct": 0, "total": 0},
        "unchanged_agent": {"correct": 0, "total": 0},
        "details": [],
    }
    
    for i, s in enumerate(scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        logp_correct = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_wrong = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_correct > logp_wrong
        
        # Track overall
        if is_correct:
            results["overall"]["correct"] += 1
        else:
            results["overall"]["incorrect"] += 1
        
        # Track by agent type
        if s["is_updated_agent"]:
            results["updated_agent"]["total"] += 1
            if is_correct:
                results["updated_agent"]["correct"] += 1
        else:
            results["unchanged_agent"]["total"] += 1
            if is_correct:
                results["unchanged_agent"]["correct"] += 1
        
        results["details"].append({
            "id": s["id"],
            "is_updated_agent": s["is_updated_agent"],
            "correct": is_correct,
            "margin": logp_correct - logp_wrong,
        })
    
    # Statistics
    print("\n[4/4] Statistical analysis...", flush=True)
    
    n_total = len(scenarios)
    k_total = results["overall"]["correct"]
    overall_acc = k_total / n_total
    
    updated_acc = (
        results["updated_agent"]["correct"] / results["updated_agent"]["total"]
        if results["updated_agent"]["total"] > 0 else 0
    )
    unchanged_acc = (
        results["unchanged_agent"]["correct"] / results["unchanged_agent"]["total"]
        if results["unchanged_agent"]["total"] > 0 else 0
    )
    
    # Binomial test
    binom_result = stats.binomtest(k_total, n_total, p=0.5, alternative='greater')
    
    results["statistics"] = {
        "overall": {
            "n_samples": n_total,
            "correct": k_total,
            "accuracy": overall_acc,
            "p_value": float(binom_result.pvalue),
            "significant": binom_result.pvalue < 0.05,
        },
        "updated_agent": {
            "accuracy": updated_acc,
            "n_samples": results["updated_agent"]["total"],
            "correct": results["updated_agent"]["correct"],
        },
        "unchanged_agent": {
            "accuracy": unchanged_acc,
            "n_samples": results["unchanged_agent"]["total"],
            "correct": results["unchanged_agent"]["correct"],
        },
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n  OVERALL: {overall_acc:.1%} ({k_total}/{n_total})")
    print(f"  p-value: {binom_result.pvalue:.2e}")
    
    print(f"\n  BREAKDOWN:")
    print(f"    Updated agent (should use NEW location):   {updated_acc:.1%}")
    print(f"    Unchanged agent (should use OLD location): {unchanged_acc:.1%}")
    
    # Interpretation
    print("\n  INTERPRETATION:")
    if overall_acc >= 0.70:
        print("  [+] Model tracks belief updates in dialogue!")
    elif overall_acc >= 0.55:
        print("  [~] Weak evidence of dialogue tracking")
    else:
        print("  [-] Model fails at dialogue tracking")
    
    # Check for asymmetry
    if abs(updated_acc - unchanged_acc) > 0.2:
        print(f"\n  [!] ASYMMETRY DETECTED:")
        if updated_acc > unchanged_acc:
            print("      Model better at updated agents (uses most recent info)")
        else:
            print("      Model better at unchanged agents (uses original info)")
            print("      -> Model uses 'first-mentioned location' HEURISTIC, not ToM!")
    
    # Save - convert numpy types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return obj
    
    output = {k: v for k, v in results.items() if k != "details"}
    output = convert_to_native(output)
    
    with open(RESULTS_DIR / "fixed_dialogue_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'fixed_dialogue_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

