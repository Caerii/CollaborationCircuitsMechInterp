"""
Fix Second-Order Test: Remove Linguistic Shortcut
==================================================

Problem: Original prompt "Where does A think B will look? A thinks B will look in the"
explicitly telegraphs the answer pattern.

Fix: Test second-order ToM with neutral question that doesn't hint at structure.
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
    
    total = 0
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1
        if pos >= 0:
            total += log_probs[pos, token_id].item()
    
    return total


def generate_fixed_second_order_scenarios(n: int = 100) -> list:
    """
    Generate second-order scenarios without telegraphing the answer.
    
    Key fix: Don't use "A thinks B will look in the" pattern.
    Instead use neutral question: "Where will B look?"
    
    The test is whether model uses B's belief (correct) or reality/A's belief (wrong).
    """
    import random
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 3)
        a, b, c = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # FIXED: Neutral question that tests understanding of B's belief
        # without explicitly saying "A thinks B will..."
        story = (
            f"{a} tells {b} that the {obj} is in the {loc1}. "
            f"Later, {c} tells {a} (privately) that the {obj} was moved to the {loc2}. "
            f"{b} was not informed of this change. "
            f"Now {b} wants to find the {obj}. "
            f"{b} will search in the"
        )
        
        scenarios.append({
            "id": f"fixed_second_order_{i}",
            "story": story,
            "agent_a": a,
            "agent_b": b,
            "b_belief": loc1,      # What B believes (original info)
            "reality": loc2,       # Where it actually is now
            "correct_completion": f" {loc1}",   # B should search where B thinks it is
            "wrong_completion": f" {loc2}",     # Wrong: using reality/A's updated belief
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("FIXED SECOND-ORDER TOM TEST")
    print("=" * 60)
    print("\nRemoving linguistic shortcut from prompts...")
    
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
    scenarios = generate_fixed_second_order_scenarios(100)
    print(f"  Generated {len(scenarios)} scenarios")
    
    # Run test
    print("\n[3/4] Testing...", flush=True)
    results = {"correct": 0, "incorrect": 0, "details": []}
    
    for i, s in enumerate(scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        logp_correct = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_wrong = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_correct > logp_wrong
        
        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
        
        results["details"].append({
            "id": s["id"],
            "correct": is_correct,
            "logp_belief": logp_correct,
            "logp_reality": logp_wrong,
            "margin": logp_correct - logp_wrong,
            "b_belief": s["b_belief"],
            "reality": s["reality"],
        })
    
    # Statistics
    print("\n[4/4] Statistical analysis...", flush=True)
    n = len(scenarios)
    k = results["correct"]
    accuracy = k / n
    
    binom_result = stats.binomtest(k, n, p=0.5, alternative='greater')
    cohens_h = 2 * (np.arcsin(np.sqrt(accuracy)) - np.arcsin(np.sqrt(0.5)))
    
    results["statistics"] = {
        "n_samples": n,
        "correct": k,
        "accuracy": accuracy,
        "p_value": float(binom_result.pvalue),
        "significant": binom_result.pvalue < 0.05,
        "cohens_h": cohens_h,
        "95_ci": [float(x) for x in binom_result.proportion_ci()],
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy: {accuracy:.1%} ({k}/{n})")
    print(f"  p-value: {binom_result.pvalue:.2e}")
    print(f"  Cohen's h: {cohens_h:.2f}")
    
    if accuracy >= 0.95:
        print("\n  [!] STILL SUSPICIOUSLY HIGH - may have other shortcuts")
    elif accuracy >= 0.70:
        print("\n  [+] GENUINE ToM EVIDENCE")
    elif accuracy >= 0.55:
        print("\n  [~] WEAK EVIDENCE - small effect")
    else:
        print("\n  [-] NO ToM EVIDENCE")
    
    # Compare to original
    print("\n  COMPARISON TO ORIGINAL:")
    print("  Original (telegraphed): 100%")
    print(f"  Fixed (neutral):        {accuracy:.1%}")
    
    # Save
    with open(RESULTS_DIR / "fixed_second_order_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "details"}, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'fixed_second_order_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

