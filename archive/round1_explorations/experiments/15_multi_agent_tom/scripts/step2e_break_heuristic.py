"""
Break the First-Mentioned Heuristic Test
==========================================

Hypothesis: Model uses "first-mentioned location" heuristic, not ToM.

This test SWAPS the order so that:
- First-mentioned = REALITY (where it actually is)
- Second-mentioned = BELIEF (where agent thinks it is)

If model uses heuristic: will predict first-mentioned (reality) = WRONG for ToM
If model uses ToM: will predict belief location = CORRECT
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


def generate_swapped_order_scenarios(n: int = 100) -> list:
    """
    Generate scenarios where first-mentioned = reality, second-mentioned = belief.
    
    Standard Sally-Anne:
      "Sally puts ball in BASKET" (first = belief)
      "Anne moves to BOX" (second = reality)
      
    SWAPPED:
      "Ball is in BOX" (first = reality)
      "Sally thinks it's in BASKET" (second = belief, mentioned after)
    """
    random.seed(42)
    
    AGENTS = ["Sally", "Anne", "Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agent = random.choice(AGENTS)
        obj = random.choice(OBJECTS)
        belief_loc, reality_loc = random.sample(LOCATIONS, 2)
        
        # SWAPPED ORDER: Reality first, then belief
        story = (
            f"The {obj} is currently in the {reality_loc}. "
            f"However, {agent} last saw the {obj} in the {belief_loc} "
            f"and does not know it has been moved. "
            f"{agent} will look for the {obj} in the"
        )
        
        scenarios.append({
            "id": f"swapped_{i}",
            "story": story,
            "agent": agent,
            "belief_loc": belief_loc,     # Second mentioned (where agent thinks)
            "reality_loc": reality_loc,   # First mentioned (where it actually is)
            "correct_completion": f" {belief_loc}",   # ToM = belief
            "wrong_completion": f" {reality_loc}",    # Heuristic = reality (first)
        })
    
    return scenarios


def generate_standard_order_scenarios(n: int = 100) -> list:
    """
    Standard order: belief first, reality second.
    This is the control condition where heuristic matches ToM.
    """
    random.seed(43)
    
    AGENTS = ["Sally", "Anne", "Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agent = random.choice(AGENTS)
        obj = random.choice(OBJECTS)
        belief_loc, reality_loc = random.sample(LOCATIONS, 2)
        
        # STANDARD ORDER: Belief first, then reality
        story = (
            f"{agent} put the {obj} in the {belief_loc}. "
            f"While {agent} was away, someone moved the {obj} to the {reality_loc}. "
            f"{agent} does not know about the move. "
            f"{agent} will look for the {obj} in the"
        )
        
        scenarios.append({
            "id": f"standard_{i}",
            "story": story,
            "agent": agent,
            "belief_loc": belief_loc,     # First mentioned
            "reality_loc": reality_loc,   # Second mentioned
            "correct_completion": f" {belief_loc}",   # ToM = belief = first
            "wrong_completion": f" {reality_loc}",    # Reality = second
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("HEURISTIC VALIDATION TEST")
    print("=" * 60)
    print("\nHypothesis: Model uses 'first-mentioned location' heuristic")
    print("Test: Swap order so first-mentioned = reality, not belief\n")
    
    # Load model
    print("[1/5] Loading model...", flush=True)
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
    
    # Generate scenarios
    print("\n[2/5] Generating scenarios...", flush=True)
    standard = generate_standard_order_scenarios(100)
    swapped = generate_swapped_order_scenarios(100)
    print(f"  Standard (belief first): {len(standard)}")
    print(f"  Swapped (reality first): {len(swapped)}")
    
    results = {
        "standard": {"correct": 0, "total": 0, "details": []},
        "swapped": {"correct": 0, "total": 0, "details": []},
    }
    
    # Test STANDARD
    print("\n[3/5] Testing STANDARD order (belief first)...", flush=True)
    for i, s in enumerate(standard):
        if i % 20 == 0:
            print(f"  [{i}/{len(standard)}]", flush=True)
        
        logp_belief = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_reality = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_belief > logp_reality
        results["standard"]["total"] += 1
        if is_correct:
            results["standard"]["correct"] += 1
        
        results["standard"]["details"].append({
            "id": s["id"],
            "correct": is_correct,
            "margin": logp_belief - logp_reality,
        })
    
    # Test SWAPPED
    print("\n[4/5] Testing SWAPPED order (reality first)...", flush=True)
    for i, s in enumerate(swapped):
        if i % 20 == 0:
            print(f"  [{i}/{len(swapped)}]", flush=True)
        
        logp_belief = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_reality = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_belief > logp_reality
        results["swapped"]["total"] += 1
        if is_correct:
            results["swapped"]["correct"] += 1
        
        results["swapped"]["details"].append({
            "id": s["id"],
            "correct": is_correct,
            "margin": logp_belief - logp_reality,
        })
    
    # Statistics
    print("\n[5/5] Statistical analysis...", flush=True)
    
    for condition in ["standard", "swapped"]:
        n = results[condition]["total"]
        k = results[condition]["correct"]
        acc = k / n
        binom = stats.binomtest(k, n, p=0.5, alternative='two-sided')
        
        results[condition]["statistics"] = {
            "accuracy": float(acc),
            "n_samples": int(n),
            "p_value": float(binom.pvalue),
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    std_acc = results["standard"]["statistics"]["accuracy"]
    swap_acc = results["swapped"]["statistics"]["accuracy"]
    
    print(f"\n  STANDARD (belief first): {std_acc:.1%}")
    print(f"  SWAPPED (reality first): {swap_acc:.1%}")
    print(f"  Difference: {std_acc - swap_acc:.1%}")
    
    print("\n  INTERPRETATION:")
    if std_acc > 0.90 and swap_acc < 0.20:
        print("  [!!!] CONFIRMED: Model uses first-mentioned heuristic!")
        print("        -> Standard high (heuristic = ToM)")
        print("        -> Swapped low (heuristic != ToM)")
    elif std_acc > 0.70 and swap_acc < 0.40:
        print("  [!] PARTIAL CONFIRMATION of heuristic")
    elif abs(std_acc - swap_acc) < 0.15:
        print("  [?] No order effect - model may use genuine ToM")
    else:
        print("  [?] Mixed results - needs more investigation")
    
    # Save
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
    
    output = {
        "standard": {k: v for k, v in results["standard"].items() if k != "details"},
        "swapped": {k: v for k, v in results["swapped"].items() if k != "details"},
    }
    output = convert_to_native(output)
    
    with open(RESULTS_DIR / "heuristic_validation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'heuristic_validation_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


