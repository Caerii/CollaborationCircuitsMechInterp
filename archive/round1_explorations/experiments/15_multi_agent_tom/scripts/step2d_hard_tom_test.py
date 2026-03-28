"""
HARD ToM Test: No Explicit Cues
================================

The problem with previous tests:
- "Bob was not informed" -> Directly tells model who knows what
- "A thinks B will look" -> Pattern matching, not reasoning

This test uses IMPLICIT information asymmetry that requires actual ToM:
- We never say who knows what
- Model must track information flow from narrative structure alone
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


def generate_hard_scenarios(n: int = 100) -> list:
    """
    Generate scenarios where information asymmetry is IMPLICIT.
    
    Key: Never explicitly state who knows what.
    The model must infer from the narrative structure.
    """
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 2)
        a, b = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # HARD VERSION: No explicit "was not informed" or "doesn't know"
        # Just narrative events that IMPLY information asymmetry
        story = (
            f"{a} put the {obj} in the {loc1}. "
            f"{a} then went to work. "
            f"While {a} was at work, {b} moved the {obj} to the {loc2}. "
            f"When {a} came home, {a} looked for the {obj}. "
            f"{a} searched in the"
        )
        
        scenarios.append({
            "id": f"hard_{i}",
            "story": story,
            "agent_a": a,
            "agent_b": b,
            "a_belief": loc1,      # A thinks it's where A left it
            "reality": loc2,       # Actually moved by B
            "correct_completion": f" {loc1}",   # A should search original location
            "wrong_completion": f" {loc2}",     # Wrong: using reality
        })
    
    return scenarios


def generate_control_scenarios(n: int = 100) -> list:
    """
    Control: EASY scenarios with explicit information.
    
    This tests whether model can track beliefs when told directly.
    """
    random.seed(43)  # Different seed
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "closet", "bed"]
    
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 2)
        a, b = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # EASY VERSION: Explicitly say who knows what (control)
        story = (
            f"{a} believes the {obj} is in the {loc1}. "
            f"In reality, the {obj} is in the {loc2}. "
            f"{a} does not know it has been moved. "
            f"{a} searched for the {obj} in the"
        )
        
        scenarios.append({
            "id": f"easy_{i}",
            "story": story,
            "agent_a": a,
            "a_belief": loc1,
            "reality": loc2,
            "correct_completion": f" {loc1}",
            "wrong_completion": f" {loc2}",
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("HARD VS EASY TOM TEST")
    print("=" * 60)
    print("\nComparing IMPLICIT (hard) vs EXPLICIT (easy) information asymmetry...")
    
    # Load model
    print("\n[1/5] Loading model...", flush=True)
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
    hard_scenarios = generate_hard_scenarios(100)
    easy_scenarios = generate_control_scenarios(100)
    print(f"  Hard (implicit): {len(hard_scenarios)}")
    print(f"  Easy (explicit): {len(easy_scenarios)}")
    
    results = {
        "hard": {"correct": 0, "total": 0, "details": []},
        "easy": {"correct": 0, "total": 0, "details": []},
    }
    
    # Test HARD scenarios
    print("\n[3/5] Testing HARD scenarios (implicit cues)...", flush=True)
    for i, s in enumerate(hard_scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(hard_scenarios)}]", flush=True)
        
        logp_belief = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_reality = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_belief > logp_reality
        results["hard"]["total"] += 1
        if is_correct:
            results["hard"]["correct"] += 1
        
        results["hard"]["details"].append({
            "id": s["id"],
            "correct": is_correct,
            "margin": logp_belief - logp_reality,
        })
    
    # Test EASY scenarios  
    print("\n[4/5] Testing EASY scenarios (explicit cues)...", flush=True)
    for i, s in enumerate(easy_scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(easy_scenarios)}]", flush=True)
        
        logp_belief = get_token_logprob(model, tokenizer, s["story"], s["correct_completion"])
        logp_reality = get_token_logprob(model, tokenizer, s["story"], s["wrong_completion"])
        
        is_correct = logp_belief > logp_reality
        results["easy"]["total"] += 1
        if is_correct:
            results["easy"]["correct"] += 1
        
        results["easy"]["details"].append({
            "id": s["id"],
            "correct": is_correct,
            "margin": logp_belief - logp_reality,
        })
    
    # Statistics
    print("\n[5/5] Statistical analysis...", flush=True)
    
    for condition in ["hard", "easy"]:
        n = results[condition]["total"]
        k = results[condition]["correct"]
        acc = k / n
        binom = stats.binomtest(k, n, p=0.5, alternative='greater')
        
        results[condition]["statistics"] = {
            "accuracy": float(acc),
            "n_samples": int(n),
            "p_value": float(binom.pvalue),
            "significant": bool(binom.pvalue < 0.05),
        }
    
    # Compare hard vs easy
    hard_correct = [d["correct"] for d in results["hard"]["details"]]
    easy_correct = [d["correct"] for d in results["easy"]["details"]]
    chi2, chi2_p = stats.chisquare([sum(hard_correct), sum(easy_correct)])
    
    results["comparison"] = {
        "chi2": float(chi2),
        "chi2_p": float(chi2_p),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    hard_acc = results["hard"]["statistics"]["accuracy"]
    easy_acc = results["easy"]["statistics"]["accuracy"]
    
    print(f"\n  HARD (implicit cues): {hard_acc:.1%}")
    print(f"  EASY (explicit cues): {easy_acc:.1%}")
    print(f"  Difference: {easy_acc - hard_acc:.1%}")
    
    print("\n  INTERPRETATION:")
    if hard_acc >= 0.90:
        print("  [+] Model uses IMPLICIT ToM - genuine understanding!")
    elif hard_acc >= 0.60 and easy_acc >= 0.90:
        print("  [~] Model relies on EXPLICIT cues - shallow ToM")
    elif hard_acc < 0.60 and easy_acc >= 0.80:
        print("  [-] Model ONLY uses explicit cues - no real ToM!")
    else:
        print("  [?] Unclear pattern - needs investigation")
    
    # Save - convert numpy types to Python native
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
        "hard": {k: v for k, v in results["hard"].items() if k != "details"},
        "easy": {k: v for k, v in results["easy"].items() if k != "details"},
        "comparison": results["comparison"],
    }
    output = convert_to_native(output)
    
    with open(RESULTS_DIR / "hard_tom_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'hard_tom_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

